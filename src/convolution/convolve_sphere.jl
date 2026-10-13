const _SPHERE_OUTPUT_MODES = (:full_map, :ring_callback, :discard)

# The ringd_context keyword shadows the constructor of the same name inside
# convolve_sphere_gpu, so reach it through an alias bound at include time.
const _new_ringd_context = ringd_context

_locked(body, ::Nothing) = body()
_locked(body, lock::ReentrantLock) = Base.lock(body, lock)

function _run_sphere_workers(body, worker_ids)
    @sync for worker_id in worker_ids
        Threads.@spawn body(worker_id)
    end
    return nothing
end

function _check_sphere_inputs(
    cp::ConvolutionParams,
    buffer::RingScanBuffer,
    rings::AbstractUnitRange{<:Integer},
    output::Symbol,
    ring_callback,
    reclaim_interval::Int,
    map_precision::Type,
    checkpoint_path,
    checkpoint_id,
    checkpoint_interval::Int,
)
    output in _SPHERE_OUTPUT_MODES || throw(ArgumentError(
        "output must be one of $_SPHERE_OUTPUT_MODES, got $output",
    ))
    if output === :ring_callback
        ring_callback === nothing && throw(ArgumentError(
            "output=:ring_callback requires a ring_callback",
        ))
    elseif ring_callback !== nothing
        throw(ArgumentError(
            "ring_callback is only used with output=:ring_callback, but " *
            "output=$output was given",
        ))
    end
    map_precision in (Float32, Float64) || throw(ArgumentError(
        "map_precision must be Float32 or Float64, got $map_precision",
    ))
    reclaim_interval >= 0 || throw(ArgumentError(
        "reclaim_interval must be non-negative, got $reclaim_interval",
    ))
    checkpoint_interval >= 0 || throw(ArgumentError(
        "checkpoint_interval must be non-negative, got $checkpoint_interval",
    ))
    checkpoint_path === nothing || output === :full_map || throw(ArgumentError(
        "checkpointing is only supported with output=:full_map; under " *
        "$output the result belongs to the caller",
    ))
    if checkpoint_path === nothing
        checkpoint_id === nothing || throw(ArgumentError(
            "checkpoint_id is only used when checkpoint_path is set",
        ))
        checkpoint_interval == 0 || throw(ArgumentError(
            "checkpoint_interval requires checkpoint_path",
        ))
    else
        checkpoint_id === nothing && throw(ArgumentError(
            "checkpoint_id is required when checkpoint_path is set; change " *
            "it whenever the alm, blm, or pointing input changes",
        ))
        isempty(strip(checkpoint_id)) && throw(ArgumentError(
            "checkpoint_id must not be empty",
        ))
    end
    buffer.nside == cp.nside_output || throw(ArgumentError(
        "buffer was built at nside=$(buffer.nside); cp has " *
        "nside=$(cp.nside_output)",
    ))
    isempty(rings) && throw(ArgumentError("rings must be non-empty"))
    buffer.first_ring <= first(rings) && last(rings) <= buffer.last_ring ||
        throw(ArgumentError(
            "rings $(first(rings)):$(last(rings)) exceed the buffered range " *
            "$(buffer.first_ring):$(buffer.last_ring)",
        ))
    return nothing
end

function _restore_sphere_state(
    cp::ConvolutionParams,
    rings::AbstractUnitRange{<:Integer},
    output::Symbol,
    unseen::Real,
    map_precision::Type,
    checkpoint_path,
    checkpoint_signature,
    resume::Bool,
)
    ring_total = length(rings)
    completed = fill(false, ring_total)
    if output !== :full_map
        return nothing, completed
    end
    if checkpoint_path !== nothing && resume && isfile(checkpoint_path)
        checkpoint = load_sphere_checkpoint(
            checkpoint_path,
            cp;
            expected_signature=checkpoint_signature,
        )
        checkpoint.first_ring == first(rings) &&
            checkpoint.last_ring == last(rings) || throw(ArgumentError(
            "checkpoint covers rings $(checkpoint.first_ring):" *
            "$(checkpoint.last_ring); this run covers " *
            "$(first(rings)):$(last(rings))",
        ))
        eltype(checkpoint.stokes_map) === map_precision || throw(ArgumentError(
            "checkpoint holds a $(eltype(checkpoint.stokes_map)) map; this " *
            "run asks for $map_precision",
        ))
        return checkpoint.stokes_map, copy(checkpoint.completed)
    end
    return allocate_stokes_map(
        cp;
        unseen=unseen,
        precision=map_precision,
    ), completed
end

function _write_sphere_checkpoint(
    checkpoint_path,
    cp::ConvolutionParams,
    rings::AbstractUnitRange{<:Integer},
    completed::Vector{Bool},
    stokes_map,
    checkpoint_signature,
)
    checkpoint_path === nothing && return nothing
    stokes_map === nothing && return nothing
    save_sphere_checkpoint(
        checkpoint_path,
        cp,
        SphereCheckpoint(
            first(rings),
            last(rings),
            copy(completed),
            stokes_map,
            checkpoint_signature,
        ),
    )
    return nothing
end

"""
    _convolve_one_ring!(...) -> (active_pixels, hit_count, processed)

The body both drivers share. Reads one ring out of the buffer, convolves it,
routes it to the chosen output, and releases everything it allocated before
returning. `processed` is false for a ring with no hits.

Under `:full_map` the write goes straight into the shared map: rings occupy
disjoint pixels, so concurrent workers never touch the same element. The ring
callback is serialized instead, since nothing is known about what it does.
"""
function _convolve_one_ring!(
    cp,
    buffer,
    ring,
    alm,
    blm,
    table,
    scan_plan,
    context,
    stokes_map,
    output,
    ring_callback,
    callback_lock,
    l_tile,
    moment_method,
    memory_fraction,
    rcond,
    unseen,
    map_precision,
    timing_callback,
)
    geometry = ring_geometry(cp, ring)
    active, angles = make_ring_angles(cp, geometry, buffer)
    active === nothing && return 0, 0, false
    active_pixels = length(active)
    hit_count = sum(angle.num for angle in angles)

    result = nothing
    try
        result = convolve_ring_gpu(
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
            ringd_context=context,
            timing_callback=timing_callback,
        )
        if output === :full_map
            binning_mapmake!(
                stokes_map,
                result,
                angles;
                rcond=rcond,
                unseen=unseen,
            )
        elseif output === :ring_callback
            payload = (
                ring=ring,
                pixels=result.pixels,
                stokes=ring_stokes(
                    result,
                    angles;
                    rcond=rcond,
                    unseen=unseen,
                    precision=map_precision,
                ),
            )
            _locked(callback_lock) do
                ring_callback(payload)
            end
        end
    finally
        result === nothing || free_ring_convolution!(result)
        _free_active_ring_pixels!(active)
    end
    return active_pixels, hit_count, true
end

"""
    convolve_sphere_gpu(
        cp, buffer, alm, blm, table, scan_plan;
        rings, output=:full_map, ring_callback=nothing,
        progress_callback=nothing, reclaim_interval=16,
        checkpoint_path=nothing, checkpoint_id=nothing,
        checkpoint_interval=0, resume=true,
        l_tile=:auto, memory_fraction=0.8, moment_method=:gemm,
        rcond=1e-2, unseen=NaN, map_precision=Float64,
        ringd_context=nothing, device_id=nothing,
    )

Convolve a whole sphere on one GPU, ring by ring, from a pixel-ordered scan
buffer.

Everything scan-independent is built once and shared by every ring: the ringD
seed plan and buffer through a `RingDContext`, and the caller's `alm`, `blm` and
Wigner table, which are neither modified nor released here. Everything else is
released before the next ring starts, so peak memory is one ring's worth
regardless of how many rings are processed; `ring_memory_bytes` gives that
figure.

Three output modes, because the full map is not always affordable: at nside 512
a Float32 map for 22 skies and 100 beams is about 83 GB.

- `:full_map` allocates a full-sky `(sky, beam, stokes, pixel)` map, fills the
  rings that were processed, and returns it. Pixels no ring writes stay
  `unseen`.
- `:ring_callback` hands each ring to `ring_callback` as a NamedTuple with
  `ring`, `pixels` and a bounded `stokes` array of
  `(sky, beam, stokes, active_pixel)`, then releases it. Returns `nothing`.
- `:discard` computes each ring and drops it, for benchmarking the GPU path
  without paying for the result. Returns `nothing`.

`progress_callback` receives a NamedTuple after every ring with `ring`,
`ring_index`, `ring_total`, `active_pixels`, `hit_count`, `elapsed_seconds` and
`processed_rings`. Rings with no hits are reported with `active_pixels = 0` and
are otherwise skipped.

`reclaim_interval` calls `CUDA.reclaim()` every that many rings, which returns
cached blocks of now-unused sizes to the driver; a sphere sweeps through many
ring lengths and therefore many allocation sizes. Zero disables it.

With `checkpoint_path` set and `output=:full_map`, the partial map and the set
of finished rings are written every `checkpoint_interval` rings and once at the
end, and `resume=true` picks up an existing compatible checkpoint and skips the
rings it already has. `checkpoint_id` is then required and must change whenever
the alm, blm, or pointing content changes. It is combined with the calculation
configuration into the signature stored in the checkpoint.
"""
function convolve_sphere_gpu(
    cp::ConvolutionParams,
    buffer::RingScanBuffer,
    alm::GPUAlm{T},
    blm::PreparedBlm{T},
    table::_CuWignerChebyshevTable{T},
    scan_plan::ScanMomentPlan{T};
    rings::AbstractUnitRange{<:Integer}=buffer.first_ring:buffer.last_ring,
    output::Symbol=:full_map,
    ring_callback=nothing,
    progress_callback=nothing,
    reclaim_interval::Integer=16,
    checkpoint_path::Union{Nothing,AbstractString}=nothing,
    checkpoint_id::Union{Nothing,AbstractString}=nothing,
    checkpoint_interval::Integer=0,
    resume::Bool=true,
    l_tile::Union{Symbol,Integer}=:auto,
    memory_fraction::Real=0.8,
    moment_method::Symbol=:gemm,
    rcond::Real=1e-2,
    unseen::Real=NaN,
    map_precision::Type=Float64,
    ringd_context::Union{Nothing,RingDContext}=nothing,
    device_id::Union{Nothing,Integer}=nothing,
    timing_callback=nothing,
) where {T<:_HARMONIC_PRECISIONS}
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    interval = Int(reclaim_interval)
    checkpoint_every = Int(checkpoint_interval)
    _check_sphere_inputs(
        cp,
        buffer,
        rings,
        output,
        ring_callback,
        interval,
        map_precision,
        checkpoint_path,
        checkpoint_id,
        checkpoint_every,
    )
    checkpoint_signature = checkpoint_path === nothing ? nothing :
        _sphere_checkpoint_signature(
            checkpoint_id,
            cp,
            buffer,
            alm,
            blm,
            table,
            scan_plan,
            rings;
            compute_precision=T,
            map_precision=map_precision,
            rcond=rcond,
            unseen=unseen,
            moment_method=moment_method,
            l_tile=l_tile,
            memory_fraction=memory_fraction,
        )
    device_id === nothing || CUDA.device!(Int(device_id))

    stokes_map, completed = _restore_sphere_state(
        cp,
        rings,
        output,
        unseen,
        map_precision,
        checkpoint_path,
        checkpoint_signature,
        resume,
    )
    owns_context = ringd_context === nothing
    context = ringd_context
    ring_total = length(rings)
    processed_rings = 0
    since_checkpoint = 0
    started = time()
    try
        if owns_context
            context = _new_ringd_context(cp; precision=T)
        else
            _check_ringd_context(context, cp)
        end

        for (ring_index, ring) in enumerate(rings)
            if completed[ring_index]
                progress_callback === nothing || progress_callback((
                    ring=ring,
                    ring_index=ring_index,
                    ring_total=ring_total,
                    active_pixels=0,
                    hit_count=0,
                    elapsed_seconds=time() - started,
                    processed_rings=processed_rings,
                    skipped=true,
                ))
                continue
            end
            active_pixels, hit_count, processed = _convolve_one_ring!(
                cp, buffer, ring, alm, blm, table, scan_plan, context,
                stokes_map, output, ring_callback, nothing,
                l_tile, moment_method, memory_fraction,
                rcond, unseen, map_precision, timing_callback,
            )
            completed[ring_index] = true
            processed && (processed_rings += 1)
            since_checkpoint += 1

            if interval > 0 && ring_index % interval == 0
                CUDA.reclaim()
            end
            if checkpoint_every > 0 && since_checkpoint >= checkpoint_every
                _write_sphere_checkpoint(
                    checkpoint_path, cp, rings, completed, stokes_map,
                    checkpoint_signature,
                )
                since_checkpoint = 0
            end
            progress_callback === nothing || progress_callback((
                ring=ring,
                ring_index=ring_index,
                ring_total=ring_total,
                active_pixels=active_pixels,
                hit_count=hit_count,
                elapsed_seconds=time() - started,
                processed_rings=processed_rings,
                skipped=false,
            ))
        end
        _write_sphere_checkpoint(
            checkpoint_path, cp, rings, completed, stokes_map,
            checkpoint_signature,
        )
    finally
        owns_context && context !== nothing && free_ringd_context!(context)
    end

    return stokes_map
end

"""
    convolve_sphere_multigpu(
        cp, buffer, prepared_alm, blm, table, scan_plan;
        devices=0:(length(CUDA.devices()) - 1), <as convolve_sphere_gpu>,
    )

Convolve a sphere across several GPUs with a shared ring queue.

The harmonic inputs are taken **host-side** here, unlike the single-GPU driver:
a `GPUAlm` and an uploaded Wigner table live on one device, so each worker
uploads its own copy along with its own `RingDContext`. `blm` is host-side in
both drivers already.

Rings are handed out one at a time from an atomic counter rather than split up
front, because ring cost varies by more than an order of magnitude between the
poles and the equator and a static split would leave workers idle.

Under `:full_map` the workers write straight into the shared map, since rings
occupy disjoint pixels. `ring_callback` and `progress_callback` are serialized,
so they need not be thread-safe, but they are called from worker tasks and
therefore in completion order rather than ring order.

Julia must be started with at least as many threads as devices for the workers
to run in parallel; with fewer, they interleave on the threads available and a
warning says so.
"""
function convolve_sphere_multigpu(
    cp::ConvolutionParams,
    buffer::RingScanBuffer,
    prepared_alm::PreparedAlm{T},
    blm::PreparedBlm{T},
    table::WignerChebyshevTable{T},
    scan_plan::ScanMomentPlan{T};
    devices=0:(length(CUDA.devices()) - 1),
    rings::AbstractUnitRange{<:Integer}=buffer.first_ring:buffer.last_ring,
    output::Symbol=:full_map,
    ring_callback=nothing,
    progress_callback=nothing,
    reclaim_interval::Integer=16,
    checkpoint_path::Union{Nothing,AbstractString}=nothing,
    checkpoint_id::Union{Nothing,AbstractString}=nothing,
    checkpoint_interval::Integer=0,
    resume::Bool=true,
    l_tile::Union{Symbol,Integer}=:auto,
    memory_fraction::Real=0.8,
    moment_method::Symbol=:gemm,
    rcond::Real=1e-2,
    unseen::Real=NaN,
    map_precision::Type=Float64,
    timing_callback=nothing,
) where {T<:_HARMONIC_PRECISIONS}
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    device_list = collect(Int, devices)
    isempty(device_list) && throw(ArgumentError("devices must be non-empty"))
    available = length(CUDA.devices())
    all(device -> 0 <= device < available, device_list) || throw(ArgumentError(
        "devices must be in 0:$(available - 1), got $device_list",
    ))
    interval = Int(reclaim_interval)
    checkpoint_every = Int(checkpoint_interval)
    _check_sphere_inputs(
        cp,
        buffer,
        rings,
        output,
        ring_callback,
        interval,
        map_precision,
        checkpoint_path,
        checkpoint_id,
        checkpoint_every,
    )
    checkpoint_signature = checkpoint_path === nothing ? nothing :
        _sphere_checkpoint_signature(
            checkpoint_id,
            cp,
            buffer,
            prepared_alm,
            blm,
            table,
            scan_plan,
            rings;
            compute_precision=T,
            map_precision=map_precision,
            rcond=rcond,
            unseen=unseen,
            moment_method=moment_method,
            l_tile=l_tile,
            memory_fraction=memory_fraction,
        )
    if Threads.nthreads() < length(device_list)
        @warn "fewer Julia threads than devices; workers will interleave" threads =
            Threads.nthreads() devices = length(device_list)
    end

    stokes_map, completed = _restore_sphere_state(
        cp,
        rings,
        output,
        unseen,
        map_precision,
        checkpoint_path,
        checkpoint_signature,
        resume,
    )
    ring_list = collect(rings)
    ring_total = length(ring_list)
    next_index = Threads.Atomic{Int}(1)
    state_lock = ReentrantLock()
    callback_lock = ReentrantLock()
    processed_rings = 0
    since_checkpoint = 0
    started = time()

    _run_sphere_workers(device_list) do device
        begin
            CUDA.device!(device)
            alm = nothing
            device_table = nothing
            context = nothing
            try
                alm = upload_alm_gpu(prepared_alm)
                device_table = upload_wigner_table_gpu(table)
                context = _new_ringd_context(cp; precision=T)
                local_index = 0
                while true
                    ring_index = Threads.atomic_add!(next_index, 1)
                    ring_index > ring_total && break
                    ring = ring_list[ring_index]
                    if completed[ring_index]
                        continue
                    end
                    local_timing_callback = timing_callback === nothing ?
                        nothing : record -> _locked(callback_lock) do
                            timing_callback((; record..., device=device))
                        end
                    active_pixels, hit_count, processed = _convolve_one_ring!(
                            cp, buffer, ring, alm, blm, device_table,
                            scan_plan, context, stokes_map, output,
                            ring_callback, callback_lock,
                            l_tile, moment_method, memory_fraction,
                            rcond, unseen, map_precision, local_timing_callback,
                        )
                    local_index += 1
                    if interval > 0 && local_index % interval == 0
                        CUDA.reclaim()
                    end
                    Base.lock(state_lock) do
                        completed[ring_index] = true
                        processed && (processed_rings += 1)
                        since_checkpoint += 1
                        if checkpoint_every > 0 &&
                            since_checkpoint >= checkpoint_every
                            _write_sphere_checkpoint(
                                checkpoint_path, cp, rings, completed,
                                stokes_map, checkpoint_signature,
                            )
                            since_checkpoint = 0
                        end
                        progress_callback === nothing ||
                            progress_callback((
                                ring=ring,
                                ring_index=ring_index,
                                ring_total=ring_total,
                                active_pixels=active_pixels,
                                hit_count=hit_count,
                                elapsed_seconds=time() - started,
                                processed_rings=processed_rings,
                                device=device,
                            ))
                    end
                end
            finally
                context === nothing || free_ringd_context!(context)
                if device_table !== nothing
                    CUDA.unsafe_free!(device_table.coefficients)
                    CUDA.unsafe_free!(device_table.pair_lookup)
                end
                alm === nothing || CUDA.unsafe_free!(alm.values)
            end
        end
    end

    _write_sphere_checkpoint(
        checkpoint_path, cp, rings, completed, stokes_map,
        checkpoint_signature,
    )
    return stokes_map
end
