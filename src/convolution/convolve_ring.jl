"""
    RingConvolution{T}

The convolution result of one HEALPix ring.

`accumulated[sky, beam, active_pixel, scan_weight]` holds the two scan weights
`(s=0, s=2)`; converting them to `(I, Q, U)` is a later stage. `positions` are
the ascending hit positions within the ring and `pixels` the corresponding
RING-ordered HEALPix indices, so the third axis of `accumulated` can be mapped
back onto the sphere.
"""
struct RingConvolution{T<:_HARMONIC_PRECISIONS}
    ring::Int
    first_pixel::Int
    ring_pixel_count::Int
    positions::Vector{Int}
    pixels::Vector{Int}
    accumulated::CuArray{Complex{T},4}
end

Base.length(result::RingConvolution) = length(result.positions)
Base.eltype(::RingConvolution{T}) where {T} = Complex{T}

function Base.show(io::IO, result::RingConvolution{T}) where {T}
    print(
        io,
        "RingConvolution{$T}(ring=$(result.ring), ",
        "$(length(result.positions))/$(result.ring_pixel_count) ",
        "pixels, size=$(size(result.accumulated)))",
    )
end

"""
    free_ring_convolution!(result)

Release the device accumulator of one ring result.
"""
function free_ring_convolution!(result::RingConvolution)
    CUDA.unsafe_free!(result.accumulated)
    return nothing
end

function _check_ring_convolution_inputs(
    cp::ConvolutionParams,
    geometry::RingGeometry,
    active::ActiveRingPixels,
    angles::AbstractVector{<:Angles},
    table::_CuWignerChebyshevTable{T},
    scan_plan::ScanMomentPlan{T},
    l_tile::Int,
) where {T}
    cp.HWP == false || throw(ArgumentError(
        "convolve_ring_gpu currently supports HWP=false",
    ))
    l_tile >= 1 || throw(ArgumentError(
        "l_tile must be positive, got $l_tile",
    ))
    active.ring_pixel_count == geometry.pixel_count ||
        throw(DimensionMismatch(
            "ActiveRingPixels describes a ring of " *
            "$(active.ring_pixel_count) pixels; the geometry has " *
            "$(geometry.pixel_count)",
        ))
    length(angles) == active.active_count || throw(DimensionMismatch(
        "got $(length(angles)) Angles for $(active.active_count) " *
        "active pixels; make_ring_angles returns them in the same order",
    ))

    # The Chebyshev variable x(beta) is only shared by the table and the scan
    # moments if both come from the same plan. A mismatch here would otherwise
    # produce a silently wrong result rather than an error.
    table.beta_max == scan_plan.beta_max || throw(ArgumentError(
        "Wigner table beta_max=$(table.beta_max) does not match " *
        "scan_plan.beta_max=$(scan_plan.beta_max); build both from the " *
        "same ScanMomentPlan",
    ))
    Int(table.beta_order) == scan_plan.beta_order || throw(ArgumentError(
        "Wigner table beta_order=$(Int(table.beta_order)) does not match " *
        "scan_plan.beta_order=$(scan_plan.beta_order)",
    ))
    Int(table.kappa) == scan_plan.kappa || throw(ArgumentError(
        "Wigner table kappa=$(Int(table.kappa)) does not match " *
        "scan_plan.kappa=$(scan_plan.kappa)",
    ))
    Int(table.nmax) == scan_plan.nmax || throw(ArgumentError(
        "Wigner table nmax=$(Int(table.nmax)) does not match " *
        "scan_plan.nmax=$(scan_plan.nmax)",
    ))
    Int(table.lstart) == cp.lstart && Int(table.lstop) == cp.lstop ||
        throw(ArgumentError(
            "Wigner table covers l=$(table.lstart):$(table.lstop); " *
            "ConvolutionParams covers $(cp.lstart):$(cp.lstop)",
        ))
    scan_plan.kappa == cp.κ || throw(ArgumentError(
        "scan_plan.kappa=$(scan_plan.kappa) does not match cp.κ=$(cp.κ)",
    ))
    scan_plan.nmax == calc_nmax(cp, cp.lstop) || throw(ArgumentError(
        "scan_plan.nmax=$(scan_plan.nmax) does not match " *
        "calc_nmax(cp, cp.lstop)=$(calc_nmax(cp, cp.lstop))",
    ))
    return nothing
end

function _profile_cuda_phase(work, timing_callback)
    timing_callback === nothing && return work(), 0.0
    CUDA.synchronize()
    started = time_ns()
    value = work()
    CUDA.synchronize()
    return value, Float64(time_ns() - started) / 1.0e9
end

"""
    convolve_ring_gpu(
        cp, geometry, active, angles, alm, blm, table, scan_plan;
        l_tile, moment_method=:gemm, device_id=nothing,
    ) -> RingConvolution

Convolve one HEALPix ring by combining the existing single-purpose entry
points. The scan-dependent objects are built once for the ring and reused by
every `l` tile:

1. the ringD stack at `geometry.theta`;
2. the scan moments of the active pixels.

Each `l` tile then builds its effective beam and its sky ring, gathers the sky
ring onto the hit pixels, contracts the two into the ring accumulator, and
releases its temporaries before the next tile. One cuFFT plan is kept for the
ring and destroyed on exit, so full-width tiles do not create a new plan each
time.

`alm`, `blm`, and `table` are scan-independent and are neither modified nor
released here, so one upload serves every ring.

`l_tile` defaults to `:auto`, which asks `suggest_l_tile` for the widest tile
that both indexes with `Int32` and fits `memory_fraction` of the device memory
still free once the ringD stack and the scan moments are resident. Pass an
integer to pin it. The result does not depend on the choice, only the memory
profile does.

`ringd_context` shares one seed plan and one ringD buffer across rings. Without
it both are built per ring, which at lmax 1200 means uploading about 627k seed
entries and allocating 5.6 GB every time.

Pass `timing_callback` to obtain a NamedTuple of synchronized, per-ring stage
times. Timing is deliberately opt-in: it synchronizes the current device at
stage boundaries, so it is suited to bottleneck measurement rather than a
zero-perturbation throughput benchmark.
"""
function convolve_ring_gpu(
    cp::ConvolutionParams,
    geometry::RingGeometry,
    active::ActiveRingPixels,
    angles::AbstractVector{<:Angles},
    alm::GPUAlm{T},
    blm::PreparedBlm{T},
    table::_CuWignerChebyshevTable{T},
    scan_plan::ScanMomentPlan{T};
    l_tile::Union{Symbol,Integer}=:auto,
    moment_method::Symbol=:gemm,
    memory_fraction::Real=0.8,
    ringd_context::Union{Nothing,RingDContext}=nothing,
    device_id::Union{Nothing,Integer}=nothing,
    timing_callback=nothing,
) where {T<:_HARMONIC_PRECISIONS}
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    automatic_tile = l_tile isa Symbol
    if automatic_tile
        l_tile === :auto || throw(ArgumentError(
            "l_tile must be :auto or a positive integer, got $l_tile",
        ))
    end
    _check_ring_convolution_inputs(
        cp,
        geometry,
        active,
        angles,
        table,
        scan_plan,
        automatic_tile ? 1 : Int(l_tile),
    )
    ringd_context === nothing ||
        _check_ringd_context(ringd_context, cp)
    device_id === nothing || CUDA.device!(Int(device_id))

    positions = Int.(Array(active.positions))
    owns_ringd = ringd_context === nothing
    ringd = nothing
    moments = nothing
    accumulated = nothing
    plan_cache = _SkyRingFFTPlanCache()
    succeeded = false
    profiling = timing_callback !== nothing
    profile_started = profiling ? time_ns() : zero(UInt64)
    ringd_seconds = 0.0
    scan_moments_seconds = 0.0
    accumulator_seconds = 0.0
    tile_sizing_seconds = 0.0
    effective_beam_seconds = 0.0
    effective_beam_wall_seconds = 0.0
    effective_kernel_seconds = 0.0
    blm_upload_seconds = 0.0
    beam_apply_seconds = 0.0
    sky_ring_seconds = 0.0
    active_sky_ring_seconds = 0.0
    contraction_seconds = 0.0
    tile_count = 0
    try
        ringd, ringd_seconds = _profile_cuda_phase(timing_callback) do
            owns_ringd ?
                ringd_blocks_recursion_gpu(cp, geometry.theta; precision=T) :
                ringd_blocks_recursion_gpu!(
                    ringd_context,
                    cp,
                    geometry.theta,
                )
        end
        moments, scan_moments_seconds = _profile_cuda_phase(timing_callback) do
            scan_moments_gpu(
                scan_plan,
                angles;
                method=moment_method,
            )
        end
        accumulated, accumulator_seconds = _profile_cuda_phase(timing_callback) do
            CUDA.zeros(
                Complex{T},
                cp.numofsky,
                cp.numofbeam,
                active.active_count,
                2,
            )
        end

        # Sized here, with the ringD stack, the moments and the accumulator
        # already resident, so the free memory reported already reflects them.
        tile_sizing_started = profiling ? time_ns() : zero(UInt64)
        tile_width = automatic_tile ?
            suggest_l_tile(
                cp,
                geometry.pixel_count,
                active.active_count;
                precision=T,
                complete_ring=active.complete,
                memory_fraction=memory_fraction,
                subtract_persistent=false,
            ) :
            Int(l_tile)
        profiling && (tile_sizing_seconds =
            Float64(time_ns() - tile_sizing_started) / 1.0e9)

        for l_first in cp.lstart:tile_width:cp.lstop
            l_last = min(l_first + tile_width - 1, cp.lstop)
            effective = nothing
            sky_ring = nothing
            active_sky_ring = nothing
            try
                effective_timing = Ref{Any}(nothing)
                effective, seconds = _profile_cuda_phase(timing_callback) do
                    build_effective_beam_gpu(
                        table,
                        moments,
                        blm,
                        l_first,
                        l_last,
                        timing_callback=profiling ?
                            record -> (effective_timing[] = record) : nothing,
                    )
                end
                effective_beam_wall_seconds += seconds
                if profiling
                    effective_timing[] === nothing && error(
                        "build_effective_beam_gpu did not produce a " *
                        "timing record",
                    )
                    effective_kernel_seconds +=
                        effective_timing[].effective_kernel_seconds
                    blm_upload_seconds +=
                        effective_timing[].blm_upload_seconds
                    beam_apply_seconds +=
                        effective_timing[].beam_apply_seconds
                    effective_beam_seconds +=
                        effective_timing[].effective_beam_seconds
                end
                sky_ring, seconds = _profile_cuda_phase(timing_callback) do
                    build_sky_ring_gpu(
                        cp,
                        ringd,
                        alm,
                        geometry.phi_pixels,
                        l_first,
                        l_last;
                        plan_cache=plan_cache,
                    )
                end
                sky_ring_seconds += seconds
                active_sky_ring, seconds = _profile_cuda_phase(timing_callback) do
                    select_active_sky_ring_gpu(
                        sky_ring,
                        active,
                    )
                end
                active_sky_ring_seconds += seconds
                _, seconds = _profile_cuda_phase(timing_callback) do
                    accumulate_convolution_tile_gemm_gpu!(
                        accumulated,
                        active_sky_ring,
                        effective,
                    )
                end
                contraction_seconds += seconds
                tile_count += 1
            finally
                if active_sky_ring !== nothing && sky_ring !== nothing
                    free_active_sky_ring!(active_sky_ring, sky_ring)
                end
                sky_ring === nothing || CUDA.unsafe_free!(sky_ring)
                effective === nothing || CUDA.unsafe_free!(effective)
            end
        end

        if profiling
            timing_callback((
                ring=geometry.ring,
                tile_count=tile_count,
                ringd_seconds=ringd_seconds,
                scan_moments_seconds=scan_moments_seconds,
                accumulator_seconds=accumulator_seconds,
                tile_sizing_seconds=tile_sizing_seconds,
                effective_beam_seconds=effective_beam_seconds,
                effective_beam_wall_seconds=effective_beam_wall_seconds,
                effective_kernel_seconds=effective_kernel_seconds,
                blm_upload_seconds=blm_upload_seconds,
                beam_apply_seconds=beam_apply_seconds,
                sky_ring_seconds=sky_ring_seconds,
                active_sky_ring_seconds=active_sky_ring_seconds,
                contraction_seconds=contraction_seconds,
                total_seconds=Float64(time_ns() - profile_started) / 1.0e9,
            ))
        end
        succeeded = true
        return RingConvolution{T}(
            geometry.ring,
            geometry.first_pixel,
            geometry.pixel_count,
            positions,
            [geometry.first_pixel + position - 1 for position in positions],
            accumulated,
        )
    finally
        _release_sky_ring_fft_plans!(plan_cache)
        moments === nothing || CUDA.unsafe_free!(moments)
        owns_ringd && ringd !== nothing && CUDA.unsafe_free!(ringd)
        succeeded || accumulated === nothing ||
            CUDA.unsafe_free!(accumulated)
    end
end

"""
    convolve_ring_gpu(
        cp, ring, phi, theta, psi, alm, blm, table, scan_plan;
        time_array=nothing, l_tile, moment_method=:gemm, device_id=nothing,
    ) -> Union{Nothing,RingConvolution}

Bin a scan into one ring and convolve it. Returns `nothing` when the ring has
no hits. The `ActiveRingPixels` built here is released before returning; use
the `RingGeometry`/`ActiveRingPixels` method directly to keep it across calls.
"""
function convolve_ring_gpu(
    cp::ConvolutionParams,
    ring::Integer,
    phi::AbstractVector{<:Real},
    theta::AbstractVector{<:Real},
    psi::AbstractVector{<:Real},
    alm::GPUAlm{T},
    blm::PreparedBlm{T},
    table::_CuWignerChebyshevTable{T},
    scan_plan::ScanMomentPlan{T};
    time_array::Union{Nothing,AbstractVector{<:Real}}=nothing,
    l_tile::Union{Symbol,Integer}=:auto,
    moment_method::Symbol=:gemm,
    memory_fraction::Real=0.8,
    ringd_context::Union{Nothing,RingDContext}=nothing,
    device_id::Union{Nothing,Integer}=nothing,
    timing_callback=nothing,
) where {T<:_HARMONIC_PRECISIONS}
    geometry = ring_geometry(cp, ring)
    active, angles = make_ring_angles(
        cp,
        geometry,
        phi,
        theta,
        psi;
        time_array=time_array,
    )
    active === nothing && return nothing
    try
        return convolve_ring_gpu(
            cp,
            geometry,
            active,
            angles,
            alm,
            blm,
            table,
            scan_plan;
            l_tile=l_tile,
            moment_method=moment_method,
            memory_fraction=memory_fraction,
            ringd_context=ringd_context,
            device_id=device_id,
            timing_callback=timing_callback,
        )
    finally
        _free_active_ring_pixels!(active)
    end
end
