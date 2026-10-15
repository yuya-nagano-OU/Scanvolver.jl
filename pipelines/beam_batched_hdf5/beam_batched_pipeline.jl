module BeamBatchedHDF5

using CUDA
using Dates
using Scanvolver
using HDF5
using HDF5: attributes
using Healpix: Resolution, max_pixrad

export automatic_beta_max, automatic_scan_plan, beam_batches
export streamed_tile_bytes, convolve_to_hdf5, read_ring_hdf5

"""
    automatic_beta_max(nside; precision=Float32)

HEALPix's global centre-to-corner radius, rounded upward with a small
allowance for Float64 rotation arithmetic and conversion to compute precision.
This is the output-map resolution, not the beam-generation resolution.
The allowance is 64 Float64 epsilons (radians) plus 8 relative compute epsilons.
Actual local beta is still checked on every ring before constructing moments.
"""
function automatic_beta_max(nside::Integer; precision::Type=Float32)
    precision in (Float32, Float64) || throw(ArgumentError("precision must be Float32 or Float64"))
    radius = Float64(max_pixrad(Resolution(Int(nside))))
    allowance = 64eps(Float64) + 8Float64(eps(precision)) * radius
    return nextfloat(precision(radius + allowance))
end

function automatic_scan_plan(cp; beta_order::Integer=8, precision::Type=Float32)
    return ScanMomentPlan(
        kappa=cp.κ, nmax=calc_nmax(cp, cp.lstop), beta_order=beta_order,
        beta_max=automatic_beta_max(cp.nside_output; precision), precision=precision,
    )
end

"""Split already-prepared, conjugated beams without conjugating them again."""
function beam_batches(blm::PreparedBlm{T}, batch_size::Integer) where {T}
    batch_size > 0 || throw(ArgumentError("beam_batch_size must be positive"))
    return [
        (indices=first:min(first + Int(batch_size) - 1, blm.numofbeam),
         blm=PreparedBlm{T}(
             [ntuple(spin -> block[spin][:, first:min(first + Int(batch_size) - 1, blm.numofbeam)], 3)
              for block in blm.slices],
             blm.lstart, blm.lstop, blm.lmax, blm.bnmax,
             min(Int(batch_size), blm.numofbeam - first + 1),
         ))
        for first in 1:Int(batch_size):blm.numofbeam
    ]
end

"""
    streamed_tile_bytes(cp_batch, ring_pixels, active_pixels, l_tile; precision, complete_ring)

Peak explicit device buffers for one tile, including simultaneous K + full U
+ gathered U during beam application. K and U both remain resident across beam
batches. Scan moments, ringD, accumulators,
the uploaded alm/table and cuFFT's plan-dependent workspace are excluded.
"""
function streamed_tile_bytes(cp, ring_pixels, active_pixels, tile;
                             precision=Float32, complete_ring=false)
    sizes = ring_memory_bytes(cp, ring_pixels, active_pixels, tile;
                              precision, complete_ring)
    bytes = Int128(sizeof(Complex{precision}))
    modes = Int128(2calc_Mmax(cp, cp.lstop) + 1)
    delta = Int128(2cp.κ + 1)
    beams = Int128(cp.numofbeam)
    packed = Int128(2calc_nmax(cp, cp.lstop) + 1) * 3 * beams * tile * bytes
    matrices = 3 * beams * delta * modes * tile * bytes
    gemm = 3 * beams * Int128(active_pixels) * modes * tile * bytes
    beam_peak = max(packed + matrices, matrices + gemm + sizes.effective_beam)
    return sizes.effective_kernel + sizes.sky_ring + sizes.active_sky_ring + beam_peak
end

available_bytes() = min(Int128(CUDA.total_memory()),
                        Int128(CUDA.free_memory()) + Int128(CUDA.cached_memory()))

function select_tile(cp_batch, geometry, active, T, l_tile, max_l_tile, fraction)
    unit = streamed_tile_bytes(cp_batch, geometry.pixel_count, length(active), 1;
                               precision=T, complete_ring=active.complete)
    budget = floor(Int128, fraction * Float64(available_bytes()))
    index_limit = max_indexable_l_tile(cp_batch, length(active))
    ceiling = min(cp_batch.lstop - cp_batch.lstart + 1, index_limit)
    width = l_tile === :auto ?
        Int(min(Int128(min(ceiling, max_l_tile)), budget ÷ unit)) :
        min(Int(l_tile), cp_batch.lstop - cp_batch.lstart + 1)
    width >= 1 && width <= ceiling && unit * width <= budget ||
        throw(ArgumentError("ring $(geometry.ring): l tile does not fit the device/index budget " *
                            "(one-l bytes=$unit, budget=$budget, index_limit=$index_limit)"))
    return width, unit * width
end

function check_beta(angles, plan, ring)
    observed = 0.0
    for angle in angles, beta in angle.β
        isfinite(beta) && 0 <= beta <= Float64(plan.beta_max) ||
            throw(DomainError(beta, "ring $ring: beta outside automatic interval [0,$(plan.beta_max)]"))
        observed = max(observed, beta)
    end
    return observed
end

function prepare_context(cp, alm, table, device, T)
    CUDA.device!(device)
    device_alm = device_table = context = nothing
    try
        device_alm = upload_alm_gpu(alm)
        device_table = upload_wigner_table_gpu(table)
        context = ringd_context(cp; precision=T)
        CUDA.synchronize()
        return (device=device, alm=device_alm, table=device_table, ringd=context)
    catch
        context === nothing || free_ringd_context!(context)
        if device_table !== nothing
            CUDA.unsafe_free!(device_table.coefficients)
            CUDA.unsafe_free!(device_table.pair_lookup)
        end
        device_alm === nothing || CUDA.unsafe_free!(device_alm.values)
        rethrow()
    end
end

function free_context(context)
    CUDA.device!(context.device)
    CUDA.synchronize()
    free_ringd_context!(context.ringd)
    CUDA.unsafe_free!(context.table.coefficients)
    CUDA.unsafe_free!(context.table.pair_lookup)
    CUDA.unsafe_free!(context.alm.values)
    return nothing
end

"""
    MapLayout

What a run writes for each batch of beams. `BeamMaps` gives one `(I, Q, U)`
map per beam; `DetectorGroupedHDF5` adds one map per detector group. The ring
queue, the tiling, the HDF5 bookkeeping and the reader are shared; a layout
only says how a batch is solved and what the file records about it.
"""
abstract type MapLayout end

"""One `(I, Q, U)` map per input beam."""
struct BeamMaps <: MapLayout
    batch_size::Int
    beam_ids
    beam_metadata
end

output_count(::BeamMaps, cp) = cp.numofbeam
output_batch_size(layout::BeamMaps) = layout.batch_size
output_indices(::BeamMaps, batch) = batch.indices
output_name(::BeamMaps) = :beams

"""`(sky, beam, I/Q/U, active_pixel)` of one beam batch on one ring."""
function batch_stokes(::BeamMaps, accumulator, geometry, positions, pixels, angles;
                      rcond, unseen, map_precision)
    T = real(eltype(accumulator))
    result = RingConvolution{T}(geometry.ring, geometry.first_pixel, geometry.pixel_count,
                                positions, pixels, accumulator)
    return ring_stokes(result, angles; rcond, unseen, precision=map_precision)
end

"""Device bytes of one ring's accumulators across all beams."""
accumulator_bytes(cp, T, active_count) =
    Int128(sizeof(Complex{T})) * cp.numofsky * cp.numofbeam * active_count * 2

extra_record(::BeamMaps, cp, T, active_count, batches) =
    (; accumulator_bytes=Int64(accumulator_bytes(cp, T, active_count)))

"""Write the layout's attributes and datasets; return its extra ring records."""
function write_layout!(file, layout::BeamMaps, cp, devices)
    attrs = attributes(file)
    attrs["schema"] = "EBCoDeCMB.beam_batched_hdf5.v1"
    # HDF5.jl reverses dimensions on disk. This deliberately gives h5py the
    # conventional (sky, beam, stokes, pixel), with pixels contiguous on disk.
    attrs["disk_axes"] = "sky,beam,stokes,pixel"
    attrs["julia_axes"] = "pixel,stokes,beam,sky"
    attrs["numofbeam"] = cp.numofbeam
    attrs["beam_batch_size"] = layout.batch_size
    file["beam_ids"] = string.(layout.beam_ids)
    file["devices"] = devices
    write_metadata_group!(file, "beam_metadata", layout.beam_metadata)
    return ("accumulator_bytes",)
end

function write_metadata_group!(file, name, values)
    group = create_group(file, name)
    try
        for (key, value) in values
            group[string(key)] = value
        end
    finally
        close(group)
    end
end

# All HDF5 access (including callbacks and completion records) uses this one
# lock. The file stays open until @sync has joined every worker, even on error.
struct RingWriter
    file::HDF5.File
    map::HDF5.Dataset
    records::Dict{String,HDF5.Dataset}
    state::HDF5.Dataset
    lock::ReentrantLock
end

# Per-ring records every layout keeps, besides completed/device/beta/elapsed.
const RING_RECORDS = ("hit_count", "active_pixels", "l_tile", "tile_count",
                      "scan_moment_builds", "ringd_builds", "sky_fft_builds",
                      "kernel_builds", "beam_applications", "estimated_tile_bytes")

function make_writer(file, layout, cp, plan, rings, T, map_precision, devices,
                     sky_ids, metadata, settings)
    attrs = attributes(file)
    attrs["created_utc"] = string(now(UTC))
    attrs["pixel_ordering"] = "RING"
    attrs["pixel_index_base"] = 1
    attrs["stokes_order"] = "I,Q,U"
    attrs["compute_precision"] = string(T)
    attrs["map_precision"] = string(map_precision)
    attrs["ebcodecmb_source"] = pathof(Scanvolver)
    for key in (:nside_output, :lstart, :lstop, :lmax, :bnmax, :κ, :numofsky)
        attrs[String(key)] = getproperty(cp, key)
    end
    for (key, value) in ("beta_max" => plan.beta_max, "beta_order" => plan.beta_order,
                         "rcond" => settings.rcond, "unseen" => settings.unseen,
                         "memory_fraction" => settings.memory_fraction,
                         "l_tile" => string(settings.l_tile),
                         "max_l_tile" => settings.max_l_tile,
                         "moment_method" => string(settings.moment_method))
        attrs[key] = value
    end
    file["sky_ids"] = string.(sky_ids)
    write_metadata_group!(file, "metadata", metadata)
    extra_records = write_layout!(file, layout, cp, devices)
    group = create_group(file, "rings")
    try
        count = ring_count(cp)
        group["pixel_first"] = [first(ring_pixel_range(cp, r)) for r in 1:count]
        group["pixel_last"] = [last(ring_pixel_range(cp, r)) for r in 1:count]
        requested = zeros(UInt8, count)
        requested[rings] .= 1
        group["requested"] = requested
        group["completed"] = zeros(UInt8, count)
        group["device"] = fill(Int32(-1), count)
        group["observed_beta_max"] = fill(NaN, count)
        group["elapsed_seconds"] = zeros(Float64, count)
        for key in (RING_RECORDS..., extra_records...)
            group[key] = zeros(Int64, count)
        end
    finally
        close(group)
    end
    file["run_state"] = UInt8[0]
    attrs["run_state_codes"] = "0=running,1=complete,2=failed"
    records = Dict(key => file["rings/$key"] for key in
        ("completed", "device", "observed_beta_max", "elapsed_seconds",
         RING_RECORDS..., extra_records...))
    outputs = output_count(layout, cp)
    dims = (cp.resol.numOfPixels, 3, outputs, cp.numofsky)
    chunk = (min(4cp.nside_output, 1024), 3, min(output_batch_size(layout), outputs),
             min(cp.numofsky, 8))
    map = create_dataset(file, "map", map_precision, dims;
                         chunk, fill_value=map_precision(settings.unseen))
    return RingWriter(file, map, records, file["run_state"], ReentrantLock())
end

function close_writer(writer)
    close(writer.map)
    foreach(close, values(writer.records))
    close(writer.state)
end

function write_batch!(writer, layout, geometry, indices, stokes, batch_callback)
    pixels = geometry.first_pixel:(geometry.first_pixel + geometry.pixel_count - 1)
    lock(writer.lock) do
        writer.map[pixels, :, indices, :] = permutedims(stokes, (4, 3, 2, 1))
        batch_callback === nothing || batch_callback(
            NamedTuple{(:ring, :pixels, output_name(layout), :stokes)}(
                (geometry.ring, pixels, indices, stokes)))
    end
end

function finish_ring!(writer, record, progress_callback; started_ns)
    lock(writer.lock) do
        # Flush all map slices before publishing the completion flag.
        flush(writer.file)
        for key in keys(record)
            key === :ring && continue
            writer.records[String(key)][record.ring:record.ring] = [getproperty(record, key)]
        end
        elapsed_seconds = (time_ns() - started_ns) / 1.0e9
        writer.records["elapsed_seconds"][record.ring:record.ring] = [elapsed_seconds]
        writer.records["completed"][record.ring:record.ring] = UInt8[1]
        flush(writer.file)
        progress_callback === nothing || progress_callback(merge(record, (; elapsed_seconds)))
    end
end

function process_ring!(writer, layout, context, cp, buffer, batches, plan, ring;
                       l_tile, max_l_tile, memory_fraction, moment_method,
                       rcond, unseen, map_precision, batch_callback, progress_callback)
    T = typeof(plan.beta_max)
    started_ns = time_ns()
    geometry = ring_geometry(cp, ring)
    active = moments = nothing
    accumulators = CuArray{Complex{T},4}[]
    plan_cache = Scanvolver._SkyRingFFTPlanCache()
    try
        active, angles = make_ring_angles(cp, geometry, buffer)
        hits = ring_hit_count(buffer, ring)
        observed = check_beta(angles, plan, ring)
        if active === nothing
            finish_ring!(writer, merge((
                ring=ring, device=context.device, observed_beta_max=observed,
                hit_count=hits, active_pixels=0, l_tile=0, tile_count=0,
                scan_moment_builds=0, ringd_builds=0, sky_fft_builds=0,
                kernel_builds=0, beam_applications=0, estimated_tile_bytes=0,
            ), extra_record(layout, cp, T, 0, ())), progress_callback; started_ns)
            return nothing
        end
        positions = Int.(Array(active.positions))
        pixels = geometry.first_pixel .+ positions .- 1
        ringd = ringd_blocks_recursion_gpu!(context.ringd, cp, geometry.theta)
        moments = scan_moments_gpu(plan, angles; method=moment_method,
                                   memory_fraction=memory_fraction)
        accumulator_bytes(cp, T, length(active)) <= memory_fraction * Float64(available_bytes()) ||
            throw(ArgumentError("ring $ring: all-beam ring accumulators exceed the memory budget"))
        for batch in batches
            push!(accumulators, CUDA.zeros(Complex{T}, cp.numofsky,
                  length(batch.indices), length(active), 2))
        end
        cp_batch = ConvolutionParams(cp; numofbeam=maximum(length(b.indices) for b in batches))
        width, tile_bytes = select_tile(cp_batch, geometry, active, T, l_tile,
                                        max_l_tile, memory_fraction)
        tile_count = 0
        for l_first in cp.lstart:width:cp.lstop
            l_last = min(l_first + width - 1, cp.lstop)
            kernel = sky = active_sky = nothing
            try
                kernel = build_effective_kernel_gpu(context.table, moments, l_first, l_last)
                sky = build_sky_ring_gpu(
                    cp, ringd, context.alm, geometry.phi_pixels, l_first, l_last;
                    plan_cache=plan_cache,
                )
                active_sky = select_active_sky_ring_gpu(sky, active)
                for (batch, accumulator) in zip(batches, accumulators)
                    effective = nothing
                    try
                        effective = apply_beams_gemm_gpu(context.table, kernel, batch.blm, l_first, l_last)
                        accumulate_convolution_tile_gemm_gpu!(accumulator, active_sky, effective)
                    finally
                        effective === nothing || CUDA.unsafe_free!(effective)
                    end
                end
                tile_count += 1
            finally
                CUDA.synchronize()
                active_sky === nothing || free_active_sky_ring!(active_sky, sky)
                sky === nothing || CUDA.unsafe_free!(sky)
                kernel === nothing || CUDA.unsafe_free!(kernel)
            end
        end
        for (batch, accumulator) in zip(batches, accumulators)
            compact = batch_stokes(layout, accumulator, geometry, positions, pixels, angles;
                                   rcond, unseen, map_precision)
            indices = output_indices(layout, batch)
            stokes = fill(map_precision(unseen), cp.numofsky, length(indices), 3, geometry.pixel_count)
            stokes[:, :, :, positions] = compact
            write_batch!(writer, layout, geometry, indices, stokes, batch_callback)
        end
        finish_ring!(writer, merge((
            ring=ring, device=context.device, observed_beta_max=observed,
            hit_count=hits, active_pixels=length(active), l_tile=width, tile_count=tile_count,
            scan_moment_builds=1, ringd_builds=1, sky_fft_builds=tile_count,
            kernel_builds=tile_count, beam_applications=tile_count * length(batches),
            estimated_tile_bytes=Int64(tile_bytes),
        ), extra_record(layout, cp, T, length(active), batches)), progress_callback; started_ns)
    finally
        CUDA.synchronize()
        Scanvolver._release_sky_ring_fft_plans!(plan_cache)
        foreach(CUDA.unsafe_free!, accumulators)
        moments === nothing || CUDA.unsafe_free!(moments)
        active === nothing || CUDA.unsafe_free!(active.positions)
    end
    return nothing
end

"""
Check what every layout needs; return the ring list, the device list and the
absolute output path.
"""
function check_run_inputs(cp, buffer, alm::PreparedAlm{T}, blm::PreparedBlm{T};
                          map_precision, rings, devices, output_path, l_tile, max_l_tile,
                          memory_fraction, rcond, unseen, reclaim_interval) where {T}
    cp.HWP && throw(ArgumentError("HWP is not supported"))
    cp.noise && throw(ArgumentError("noise is not supported"))
    T in (Float32, Float64) && map_precision in (Float32, Float64) ||
        throw(ArgumentError("compute/map precision must be Float32 or Float64"))
    for prepared in (alm, blm)
        (prepared.lstart, prepared.lstop, prepared.lmax) == (cp.lstart, cp.lstop, cp.lmax) ||
            throw(ArgumentError("prepared harmonics do not match cp multipoles"))
    end
    alm.numofsky == cp.numofsky && blm.numofbeam == cp.numofbeam && blm.bnmax == cp.bnmax ||
        throw(ArgumentError("prepared sky/beam dimensions do not match cp"))
    buffer.nside == cp.nside_output || throw(ArgumentError("scan-buffer nside does not match cp"))
    max_l_tile > 0 || throw(ArgumentError("batch and tile sizes must be positive"))
    l_tile === :auto || (l_tile isa Integer && l_tile > 0) || throw(ArgumentError("invalid l_tile"))
    isfinite(memory_fraction) && 0 < memory_fraction <= 1 || throw(ArgumentError("invalid memory_fraction"))
    isfinite(rcond) && rcond >= 0 || throw(ArgumentError("invalid rcond"))
    isinf(unseen) && throw(ArgumentError("unseen must be finite or NaN"))
    reclaim_interval >= 0 || throw(ArgumentError("invalid reclaim_interval"))
    ring_list = Int.(collect(rings))
    !isempty(ring_list) && allunique(ring_list) &&
        all(r -> buffer.first_ring <= r <= buffer.last_ring, ring_list) ||
        throw(ArgumentError("rings must be unique and inside the scan buffer"))
    device_list = Int.(collect(devices))
    !isempty(device_list) && allunique(device_list) &&
        all(d -> 0 <= d < length(CUDA.devices()), device_list) || throw(ArgumentError("invalid devices"))
    Threads.nthreads() >= length(device_list) || throw(ArgumentError("start Julia with at least one thread per GPU"))
    path = abspath(output_path)
    ispath(path) && throw(ArgumentError("output already exists: $path"))
    isdir(dirname(path)) || throw(ArgumentError("output directory does not exist"))
    return ring_list, device_list, path
end

"""
Run every requested ring of `layout` and write the HDF5 file at `path`.
Persistent device contexts consume a dynamic ring queue.
"""
function run_rings(layout, cp, buffer, alm::PreparedAlm{T}, blm, batches, path, ring_list,
                   device_list; beta_order, l_tile, max_l_tile, memory_fraction,
                   moment_method, map_precision, rcond, unseen, reclaim_interval,
                   sky_ids, metadata, batch_callback, progress_callback) where {T}
    plan = automatic_scan_plan(cp; beta_order, precision=T)
    table = build_wigner_chebyshev_table_cpu(cp, plan)
    contexts = []
    try
        # Serial first-use CUDA setup avoids concurrent context initialization.
        for device in device_list
            push!(contexts, prepare_context(cp, alm, table, device, T))
        end
        h5open(path, "w") do file
            writer = make_writer(file, layout, cp, plan, ring_list, T, map_precision,
                                 device_list, sky_ids, metadata,
                                 (; rcond, unseen, l_tile, max_l_tile, memory_fraction,
                                    moment_method))
            next_ring = Threads.Atomic{Int}(1)
            failed = Threads.Atomic{Bool}(false)
            try
                @sync for context in contexts
                    Threads.@spawn begin
                        CUDA.device!(context.device)
                        processed = 0
                        try
                            while !failed[]
                                index = Threads.atomic_add!(next_ring, 1)
                                index > length(ring_list) && break
                                process_ring!(writer, layout, context, cp, buffer, batches, plan,
                                              ring_list[index]; l_tile, max_l_tile,
                                              memory_fraction, moment_method, rcond, unseen,
                                              map_precision, batch_callback, progress_callback)
                                processed += 1
                                if reclaim_interval > 0 && processed % reclaim_interval == 0
                                    CUDA.synchronize()
                                    CUDA.reclaim()
                                end
                            end
                        catch
                            failed[] = true
                            rethrow()
                        end
                    end
                end
                writer.state[:] = UInt8[1]
                flush(file)
            catch
                writer.state[:] = UInt8[2]
                flush(file)
                rethrow()
            finally
                close_writer(writer)
            end
        end
    finally
        foreach(free_context, contexts)
    end
    return path
end

"""
    convolve_to_hdf5(cp, buffer, prepared_alm, prepared_blm; output_path, ...)

Persistent device contexts consume a dynamic ring queue. All skies remain on
each GPU. Beam intermediates are batch-sized; only one ring's accumulators
span all beams. Shared K/U are built once per l tile, not per beam batch.

The output file is created only when `output_path` does not exist. Failures
propagate after every GPU worker has joined; completed rings remain recorded
in the partial file. This version records progress but does not auto-resume.
Callbacks run serially; batch_callback receives a borrowed, ring-local CPU
array after writing it, before that ring is marked complete. No full map is
allocated or returned. Returns the absolute output path.
"""
function convolve_to_hdf5(cp::ConvolutionParams, buffer::RingScanBuffer,
                         alm::PreparedAlm{T}, blm::PreparedBlm{T};
                         output_path::AbstractString, beam_batch_size::Integer=9,
                         devices=[0], rings=buffer.first_ring:buffer.last_ring,
                         beta_order::Integer=8, l_tile=:auto, max_l_tile::Integer=32,
                         memory_fraction::Real=0.7, moment_method::Symbol=:gemm,
                         map_precision::Type=T, rcond::Real=1e-2, unseen::Real=NaN,
                         reclaim_interval::Integer=16,
                         sky_ids=string.(1:cp.numofsky), beam_ids=string.(1:cp.numofbeam),
                         metadata=Dict{String,Any}(), beam_metadata=Dict{String,Any}(),
                         batch_callback=nothing, progress_callback=nothing) where {T}
    ring_list, device_list, path = check_run_inputs(
        cp, buffer, alm, blm; map_precision, rings, devices, output_path, l_tile,
        max_l_tile, memory_fraction, rcond, unseen, reclaim_interval)
    beam_batch_size > 0 || throw(ArgumentError("batch and tile sizes must be positive"))
    length(sky_ids) == cp.numofsky && length(beam_ids) == cp.numofbeam || throw(ArgumentError("ID counts do not match cp"))
    for (key, values) in beam_metadata
        values isa AbstractVector && length(values) == cp.numofbeam ||
            throw(ArgumentError("beam metadata $key must have one value per beam"))
    end
    layout = BeamMaps(Int(beam_batch_size), beam_ids, beam_metadata)
    return run_rings(layout, cp, buffer, alm, blm, beam_batches(blm, beam_batch_size), path,
                     ring_list, device_list; beta_order, l_tile, max_l_tile, memory_fraction,
                     moment_method, map_precision, rcond, unseen, reclaim_interval, sky_ids,
                     metadata, batch_callback, progress_callback)
end

"""Read a completed ring as `(sky, output, I/Q/U, ring_pixel)`; selections are ranges."""
function read_ring_maps(path, ring::Integer, skies, outputs, label)
    return h5open(path, "r") do file
        completed = read(file["rings/completed"])
        1 <= ring <= length(completed) && completed[ring] == 1 ||
            throw(ArgumentError("ring $ring has not been completely saved"))
        first_pixel = file["rings/pixel_first"][ring]
        last_pixel = file["rings/pixel_last"][ring]
        dataset = file["map"]
        try
            sky_range = skies === nothing ? (1:size(dataset, 4)) : skies
            output_range = outputs === nothing ? (1:size(dataset, 3)) : outputs
            sky_range isa AbstractUnitRange && output_range isa AbstractUnitRange ||
                throw(ArgumentError("sky/$label selections must be ranges"))
            return permutedims(dataset[first_pixel:last_pixel, :, output_range, sky_range], (4, 3, 2, 1))
        finally
            close(dataset)
        end
    end
end

"""Read a completed ring as `(sky, beam, I/Q/U, ring_pixel)`; selections are ranges."""
read_ring_hdf5(path, ring::Integer; skies=nothing, beams=nothing) =
    read_ring_maps(path, ring, skies, beams, "beam")

end # module
