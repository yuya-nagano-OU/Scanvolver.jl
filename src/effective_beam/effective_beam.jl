const _EFFECTIVE_BEAM_THREADS = 256
const _EFFECTIVE_BEAM_CUBLAS = isdefined(CUDA, :cuBLAS) ?
    CUDA.cuBLAS : CUDA.CUBLAS

@inline function _effective_beam_contract_add(
    total::ComplexF64,
    correction::ComplexF64,
    term::ComplexF64,
)
    return total + term, correction
end

@inline function _effective_beam_contract_add(
    total::ComplexF32,
    correction::ComplexF32,
    term::ComplexF32,
)
    corrected_term = term - correction
    updated = total + corrected_term
    updated_correction = (updated - total) - corrected_term
    return updated, updated_correction
end

function _check_effective_kernel_inputs(
    table::_CuWignerChebyshevTable,
    moments::CuArray,
    l_first::Int,
    l_last::Int,
)
    Int(table.lstart) <= l_first <= l_last <= Int(table.lstop) ||
        throw(ArgumentError(
            "l tile must lie in $(table.lstart):$(table.lstop)",
        ))
    expected_prefix = (
        2,
        Int(table.beta_order),
        2 * Int(table.kappa) + 1,
        2 * Int(table.nmax) + 1,
    )
    size(moments)[1:4] == expected_prefix ||
        throw(DimensionMismatch(
            "moments prefix $(size(moments)[1:4]) does not match " *
            "$expected_prefix",
        ))
    size(moments, 5) > 0 || throw(ArgumentError(
        "moments must contain at least one pixel",
    ))
    return nothing
end

function _build_effective_kernel_kernel!(
    output,
    moments,
    coefficients,
    pair_lookup,
    l_first::Int32,
    lstart::Int32,
    kappa::Int32,
    nmax_global::Int32,
    Mmax_global::Int32,
    beta_order::Int32,
    tile_count::Int32,
    pixel_count::Int32,
)
    linear_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    delta_count = Int32(2) * kappa + Int32(1)
    M_count = Int32(2) * Mmax_global + Int32(1)
    total =
        Int32(2) * delta_count * M_count * pixel_count * tile_count
    linear_index > total && return nothing
    index = linear_index - Int32(1)
    scan_weight = index % Int32(2) + Int32(1)
    index ÷= Int32(2)
    delta_index = index % delta_count + Int32(1)
    index ÷= delta_count
    M_index = index % M_count + Int32(1)
    index ÷= M_count
    pixel_index = index % pixel_count + Int32(1)
    tile_index = index ÷ pixel_count + Int32(1)

    l = l_first + tile_index - Int32(1)
    M = M_index - Mmax_global - Int32(1)
    abs(M) <= min(Mmax_global, l) || return nothing
    delta = delta_index - kappa - Int32(1)
    n = M - delta
    abs(n) <= min(nmax_global, l) || return nothing
    @inbounds pair_index = pair_lookup[
        M + Mmax_global + Int32(1),
        n + nmax_global + Int32(1),
    ]
    pair_index == Int32(0) && return nothing

    l_index = l - lstart + Int32(1)
    n_index = n + nmax_global + Int32(1)
    total_value = zero(eltype(output))
    correction = zero(eltype(output))
    @inbounds for beta_index in Int32(1):beta_order
        term =
            coefficients[beta_index, pair_index, l_index] *
            moments[
                scan_weight,
                beta_index,
                delta_index,
                n_index,
                pixel_index,
            ]
        total_value, correction = _effective_beam_contract_add(
            total_value,
            correction,
            term,
        )
    end
    batch_index = M_index + M_count * (tile_index - Int32(1))
    @inbounds output[
        delta_index,
        pixel_index,
        batch_index,
        scan_weight,
    ] = total_value
    return nothing
end

"""
    build_effective_kernel_gpu(table, moments, l_first, l_last)

Build the beam-independent compact effective kernel for one resident `l`
tile:

```math
K_{w,p,M,\\delta,l}=\\sum_r
c^l_{M,M-\\delta,r}H_{w,r,\\delta,M-\\delta}(p).
```

The returned `CuArray` has layout
`K[delta, pixel, (M,l), scan_weight]`. `delta=M-n` contains only
`2*table.kappa+1` modes. The kernel can be reused with multiple prepared
beam sets before being released.
"""
function build_effective_kernel_gpu(
    table::_CuWignerChebyshevTable{T},
    moments::CuArray{Complex{T},5},
    l_first::Integer,
    l_last::Integer,
) where {T<:_GPU_RECURSION_PRECISIONS}
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    first_l = Int(l_first)
    last_l = Int(l_last)
    _check_effective_kernel_inputs(table, moments, first_l, last_l)

    pixel_count = size(moments, 5)
    tile_count = last_l - first_l + 1
    delta_count = 2 * Int(table.kappa) + 1
    M_count = 2 * Int(table.Mmax) + 1
    batch_count = M_count * tile_count
    total =
        Int128(2) * delta_count * pixel_count * batch_count
    total <= typemax(Int32) || throw(ArgumentError(
        "compact effective-kernel tile has $total elements, exceeding " *
        "the Int32-indexed kernel limit; reduce the l tile",
    ))

    output = CUDA.zeros(
        Complex{T},
        delta_count,
        pixel_count,
        batch_count,
        2,
    )
    succeeded = false
    try
        @cuda threads=_EFFECTIVE_BEAM_THREADS blocks=cld(
            Int(total),
            _EFFECTIVE_BEAM_THREADS,
        ) _build_effective_kernel_kernel!(
            output,
            moments,
            table.coefficients,
            table.pair_lookup,
            Int32(first_l),
            table.lstart,
            table.kappa,
            table.nmax,
            table.Mmax,
            table.beta_order,
            Int32(tile_count),
            Int32(pixel_count),
        )
        CUDA.synchronize()
        succeeded = true
        return output
    finally
        succeeded || CUDA.unsafe_free!(output)
    end
end

function _check_prepared_blm(
    table::_CuWignerChebyshevTable,
    blm::PreparedBlm,
    l_first::Int,
    l_last::Int,
)
    blm.lstart == Int(table.lstart) &&
        blm.lstop == Int(table.lstop) || throw(ArgumentError(
        "PreparedBlm l bounds $(blm.lstart):$(blm.lstop) do not match " *
        "the Wigner table bounds $(table.lstart):$(table.lstop)",
    ))
    min(blm.bnmax, blm.lstop) == Int(table.nmax) ||
        throw(ArgumentError(
            "PreparedBlm nmax does not match the Wigner table nmax",
        ))
    length(blm.slices) == blm.lstop - blm.lstart + 1 ||
        throw(DimensionMismatch(
            "PreparedBlm slice count does not match its l bounds",
        ))
    blm.numofbeam > 0 || throw(ArgumentError(
        "PreparedBlm must contain at least one beam",
    ))
    for l in l_first:l_last
        nmax_l = min(Int(table.nmax), l)
        blocks = blm_slice(blm, l)
        for spin_index in 1:3
            size(blocks[spin_index]) == (
                2 * nmax_l + 1,
                blm.numofbeam,
            ) || throw(DimensionMismatch(
                "PreparedBlm spin block has the wrong shape for l=$l",
            ))
        end
    end
    return nothing
end

function _upload_blm_tile_gpu(
    table::_CuWignerChebyshevTable{T},
    blm::PreparedBlm{T},
    l_first::Int,
    l_last::Int,
) where {T<:_GPU_RECURSION_PRECISIONS}
    nmax_global = Int(table.nmax)
    tile_count = l_last - l_first + 1
    packed = zeros(
        Complex{T},
        2 * nmax_global + 1,
        3,
        blm.numofbeam,
        tile_count,
    )
    for (tile_index, l) in enumerate(l_first:l_last)
        nmax_l = min(nmax_global, l)
        destination =
            (nmax_global - nmax_l + 1):(nmax_global + nmax_l + 1)
        blocks = blm_slice(blm, l)
        for spin_index in 1:3
            @views packed[
                destination,
                spin_index,
                :,
                tile_index,
            ] .= blocks[spin_index]
        end
    end
    return CuArray(packed)
end

function _pack_beam_gemm_matrices_kernel!(
    beam_matrices,
    packed_blm,
    kappa::Int32,
    nmax_global::Int32,
    Mmax_global::Int32,
    tile_count::Int32,
    beam_count::Int32,
    l_first::Int32,
)
    linear_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    spin_beam_count = Int32(3) * beam_count
    delta_count = Int32(2) * kappa + Int32(1)
    M_count = Int32(2) * Mmax_global + Int32(1)
    total = spin_beam_count * delta_count * M_count * tile_count
    linear_index > total && return nothing
    index = linear_index - Int32(1)
    spin_beam_index = index % spin_beam_count + Int32(1)
    index ÷= spin_beam_count
    delta_index = index % delta_count + Int32(1)
    index ÷= delta_count
    M_index = index % M_count + Int32(1)
    tile_index = index ÷ M_count + Int32(1)

    spin_index =
        (spin_beam_index - Int32(1)) % Int32(3) + Int32(1)
    beam_index =
        (spin_beam_index - Int32(1)) ÷ Int32(3) + Int32(1)
    l = l_first + tile_index - Int32(1)
    M = M_index - Mmax_global - Int32(1)
    abs(M) <= min(Mmax_global, l) || return nothing
    delta = delta_index - kappa - Int32(1)
    n = M - delta
    abs(n) <= min(nmax_global, l) || return nothing
    n_index = n + nmax_global + Int32(1)
    batch_index = M_index + M_count * (tile_index - Int32(1))
    @inbounds beam_matrices[
        spin_beam_index,
        delta_index,
        batch_index,
    ] = packed_blm[
        n_index,
        spin_index,
        beam_index,
        tile_index,
    ]
    return nothing
end

function _pack_effective_beam_output_kernel!(
    output,
    gemm_output,
    scan_weight::Int32,
    M_count::Int32,
    pixel_count::Int32,
    beam_count::Int32,
    batch_count::Int32,
)
    linear_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    spin_beam_count = Int32(3) * beam_count
    total = spin_beam_count * pixel_count * batch_count
    linear_index > total && return nothing
    index = linear_index - Int32(1)
    spin_beam_index = index % spin_beam_count + Int32(1)
    index ÷= spin_beam_count
    pixel_index = index % pixel_count + Int32(1)
    batch_index = index ÷ pixel_count + Int32(1)

    spin_index =
        (spin_beam_index - Int32(1)) % Int32(3) + Int32(1)
    beam_index =
        (spin_beam_index - Int32(1)) ÷ Int32(3) + Int32(1)
    M_index = (batch_index - Int32(1)) % M_count + Int32(1)
    tile_index =
        (batch_index - Int32(1)) ÷ M_count + Int32(1)
    term_index = spin_index + Int32(3) * (
        M_index - Int32(1) + M_count * (tile_index - Int32(1))
    )
    @inbounds output[
        term_index,
        beam_index,
        pixel_index,
        scan_weight,
    ] = gemm_output[
        spin_beam_index,
        pixel_index,
        batch_index,
    ]
    return nothing
end

"""
    apply_beams_gemm_gpu(
        table, effective_kernel, blm, l_first, l_last,
    )

Apply every beam and beam-spin column in `blm` to a compact effective
kernel. For each `(scan_weight,M,l)` batch this evaluates

`(3*beam, 2*kappa+1) * (2*kappa+1, pixel)`

with a non-conjugating strided batched GEMM. `PreparedBlm` already contains
the conjugation required by the convolution convention.

The returned layout is
`E[(spin,M,l), beam, pixel, scan_weight]`. Within the first axis spin varies
fastest in the order `(0,+2,-2)`, followed by signed `M`, then `l`.
"""
function apply_beams_gemm_gpu(
    table::_CuWignerChebyshevTable{T},
    effective_kernel::CuArray{Complex{T},4},
    blm::PreparedBlm{T},
    l_first::Integer,
    l_last::Integer,
    ;
    timing_callback=nothing,
) where {T<:_GPU_RECURSION_PRECISIONS}
    first_l = Int(l_first)
    last_l = Int(l_last)
    Int(table.lstart) <= first_l <= last_l <= Int(table.lstop) ||
        throw(ArgumentError(
            "l tile must lie in $(table.lstart):$(table.lstop)",
        ))
    _check_prepared_blm(table, blm, first_l, last_l)

    tile_count = last_l - first_l + 1
    delta_count = 2 * Int(table.kappa) + 1
    M_count = 2 * Int(table.Mmax) + 1
    batch_count = M_count * tile_count
    pixel_count = size(effective_kernel, 2)
    size(effective_kernel) == (
        delta_count,
        pixel_count,
        batch_count,
        2,
    ) || throw(DimensionMismatch(
        "effective_kernel has incompatible size $(size(effective_kernel))",
    ))
    pixel_count > 0 || throw(ArgumentError(
        "effective_kernel must contain at least one pixel",
    ))

    beam_count = blm.numofbeam
    spin_beam_count = 3 * beam_count
    beam_total =
        Int128(spin_beam_count) * delta_count * batch_count
    beam_total <= typemax(Int32) || throw(ArgumentError(
        "beam GEMM matrix tile has $beam_total elements, exceeding the " *
        "Int32-indexed packing limit; reduce the l tile",
    ))
    term_count = 3 * M_count * tile_count
    output_total =
        Int128(2) * term_count * beam_count * pixel_count
    output_total <= typemax(Int32) || throw(ArgumentError(
        "effective-beam tile has $output_total elements, exceeding the " *
        "Int32-indexed packing limit; reduce the l tile",
    ))
    per_scan_total =
        Int128(spin_beam_count) * pixel_count * batch_count
    per_scan_total <= typemax(Int32) || throw(ArgumentError(
        "beam GEMM output has $per_scan_total elements per scan weight; " *
        "reduce the l tile",
    ))

    packed_blm = nothing
    beam_matrices = nothing
    output = nothing
    succeeded = false
    profiling = timing_callback !== nothing
    blm_upload_seconds = 0.0
    beam_apply_seconds = 0.0
    apply_started = zero(UInt64)
    try
        if profiling
            CUDA.synchronize()
        end
        upload_started = profiling ? time_ns() : zero(UInt64)
        packed_blm = _upload_blm_tile_gpu(
            table,
            blm,
            first_l,
            last_l,
        )
        if profiling
            CUDA.synchronize()
            blm_upload_seconds =
                Float64(time_ns() - upload_started) / 1.0e9
        end
        apply_started = profiling ? time_ns() : zero(UInt64)
        beam_matrices = CUDA.zeros(
            Complex{T},
            spin_beam_count,
            delta_count,
            batch_count,
        )
        @cuda threads=_EFFECTIVE_BEAM_THREADS blocks=cld(
            Int(beam_total),
            _EFFECTIVE_BEAM_THREADS,
        ) _pack_beam_gemm_matrices_kernel!(
            beam_matrices,
            packed_blm,
            table.kappa,
            table.nmax,
            table.Mmax,
            Int32(tile_count),
            Int32(beam_count),
            Int32(first_l),
        )
        CUDA.synchronize()
        CUDA.unsafe_free!(packed_blm)
        packed_blm = nothing

        output = CUDA.zeros(
            Complex{T},
            term_count,
            beam_count,
            pixel_count,
            2,
        )
        alpha = one(Complex{T})
        beta = zero(Complex{T})
        for scan_weight in 1:2
            gemm_output = nothing
            try
                gemm_output = CUDA.zeros(
                    Complex{T},
                    spin_beam_count,
                    pixel_count,
                    batch_count,
                )
                kernel_view = view(
                    effective_kernel,
                    :, :, :, scan_weight,
                )
                _EFFECTIVE_BEAM_CUBLAS.gemm_strided_batched!(
                    'N',
                    'N',
                    alpha,
                    beam_matrices,
                    kernel_view,
                    beta,
                    gemm_output,
                )
                @cuda threads=_EFFECTIVE_BEAM_THREADS blocks=cld(
                    Int(per_scan_total),
                    _EFFECTIVE_BEAM_THREADS,
                ) _pack_effective_beam_output_kernel!(
                    output,
                    gemm_output,
                    Int32(scan_weight),
                    Int32(M_count),
                    Int32(pixel_count),
                    Int32(beam_count),
                    Int32(batch_count),
                )
                CUDA.synchronize()
            finally
                gemm_output === nothing || begin
                    CUDA.synchronize()
                    CUDA.unsafe_free!(gemm_output)
                end
            end
        end
        succeeded = true
    finally
        CUDA.synchronize()
        packed_blm === nothing || CUDA.unsafe_free!(packed_blm)
        beam_matrices === nothing || CUDA.unsafe_free!(beam_matrices)
        succeeded || output === nothing || CUDA.unsafe_free!(output)
        if profiling && succeeded
            CUDA.synchronize()
            beam_apply_seconds =
                Float64(time_ns() - apply_started) / 1.0e9
        end
    end
    if profiling
        try
            timing_callback((
                blm_upload_seconds=blm_upload_seconds,
                beam_apply_seconds=beam_apply_seconds,
            ))
        catch
            output === nothing || CUDA.unsafe_free!(output)
            output = nothing
            rethrow()
        end
    end
    return output
end

"""
    build_effective_beam_gpu(
        table, moments, blm, l_first, l_last,
    )

Build one bounded effective-beam tile. This is the production convenience
path: construct the beam-independent compact kernel once, apply all beams by
batched GEMM, and release the compact kernel before returning
`E[(spin,M,l),beam,pixel,scan_weight]`.
"""
function build_effective_beam_gpu(
    table::_CuWignerChebyshevTable{T},
    moments::CuArray{Complex{T},5},
    blm::PreparedBlm{T},
    l_first::Integer,
    l_last::Integer,
    ;
    timing_callback=nothing,
) where {T<:_GPU_RECURSION_PRECISIONS}
    profiling = timing_callback !== nothing
    profiling && CUDA.synchronize()
    profile_started = profiling ? time_ns() : zero(UInt64)
    kernel_started = profile_started
    effective_kernel = build_effective_kernel_gpu(
        table,
        moments,
        l_first,
        l_last,
    )
    effective_kernel_seconds = profiling ?
        Float64(time_ns() - kernel_started) / 1.0e9 : 0.0
    apply_timing = Ref{Any}(nothing)
    effective = nothing
    try
        effective = apply_beams_gemm_gpu(
            table,
            effective_kernel,
            blm,
            l_first,
            l_last,
            timing_callback=profiling ?
                record -> (apply_timing[] = record) : nothing,
        )
    finally
        CUDA.synchronize()
        CUDA.unsafe_free!(effective_kernel)
    end
    if profiling
        apply_timing[] === nothing && error(
            "apply_beams_gemm_gpu did not produce a timing record",
        )
        CUDA.synchronize()
        effective_beam_seconds =
            effective_kernel_seconds +
            apply_timing[].blm_upload_seconds +
            apply_timing[].beam_apply_seconds
        try
            timing_callback((
                effective_kernel_seconds=effective_kernel_seconds,
                blm_upload_seconds=apply_timing[].blm_upload_seconds,
                beam_apply_seconds=apply_timing[].beam_apply_seconds,
                effective_beam_seconds=effective_beam_seconds,
                effective_beam_wall_seconds=
                    Float64(time_ns() - profile_started) / 1.0e9,
            ))
        catch
            effective === nothing || CUDA.unsafe_free!(effective)
            effective = nothing
            rethrow()
        end
    end
    return effective
end
