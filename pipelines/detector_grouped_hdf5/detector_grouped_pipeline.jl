module DetectorGroupedHDF5

using CUDA
using Dates
using Scanvolver
using HDF5
using HDF5: attributes
using LinearAlgebra

# Reuse the validated scan planning, device-context, tiling, and beam batching
# implementation.  This module owns a separate writer and mapmaking path.
include(joinpath(@__DIR__, "..", "beam_batched_hdf5", "beam_batched_pipeline.jl"))
using .BeamBatchedHDF5

export convolve_detector_groups_to_hdf5, detector_group_batches
export grouped_ring_stokes, reduce_detector_accumulators, read_group_ring_hdf5

const DEFAULT_DETECTOR_LABELS = ("Qp", "Qm", "Up", "Um")
const DEFAULT_DETECTOR_OFFSETS = (0.0, pi / 2, pi / 4, 3pi / 4)

function normalized_detector_weights(offsets, weights)
    length(offsets) == length(weights) || throw(DimensionMismatch(
        "detector_offsets and detector_weights must have the same length",
    ))
    !isempty(offsets) || throw(ArgumentError("at least one detector is required"))
    all(isfinite, offsets) || throw(ArgumentError("detector offsets must be finite"))
    all(weight -> isfinite(weight) && weight > 0, weights) || throw(ArgumentError(
        "detector weights must be finite and positive",
    ))
    return Float64.(weights) ./ sum(weights)
end

"""Split prepared beams into batches containing only complete detector groups."""
function detector_group_batches(blm::PreparedBlm, detectors_per_group::Integer,
                                group_batch_size::Integer)
    detectors_per_group > 0 || throw(ArgumentError("detectors_per_group must be positive"))
    group_batch_size > 0 || throw(ArgumentError("group_batch_size must be positive"))
    blm.numofbeam % detectors_per_group == 0 || throw(ArgumentError(
        "$(blm.numofbeam) beams cannot be divided into groups of $detectors_per_group",
    ))
    source_batches = BeamBatchedHDF5.beam_batches(
        blm, Int(detectors_per_group) * Int(group_batch_size),
    )
    return [(
        indices=batch.indices,
        groups=((first(batch.indices) - 1) ÷ detectors_per_group + 1):
               (last(batch.indices) ÷ detectors_per_group),
        blm=batch.blm,
    ) for batch in source_batches]
end

"""
    reduce_detector_accumulators(values, detector_offsets, detector_weights)

Combine consecutive detector accumulators before mapmaking.  `values` has axes
`(sky, input_beam, active_pixel, scan_weight)`.  The first scan weight is
averaged directly; the spin-2 scan weight is averaged after multiplication by
`exp(2im * detector_offset)`.  The returned axes are
`(sky, detector_group, active_pixel, scan_weight)`.
"""
function reduce_detector_accumulators(values::AbstractArray{Complex{T},4},
                                      detector_offsets, detector_weights) where {T}
    detector_count = length(detector_offsets)
    normalized = normalized_detector_weights(detector_offsets, detector_weights)
    size(values, 4) == 2 || throw(DimensionMismatch("the scan-weight axis must have length 2"))
    size(values, 2) % detector_count == 0 || throw(DimensionMismatch(
        "the beam axis must contain complete detector groups",
    ))
    group_count = size(values, 2) ÷ detector_count
    grouped = zeros(Complex{T}, size(values, 1), group_count, size(values, 3), 2)
    for group in 1:group_count, detector in 1:detector_count
        source = (group - 1) * detector_count + detector
        weight0 = T(normalized[detector])
        weight2 = Complex{T}(normalized[detector] * cis(2 * detector_offsets[detector]))
        @views grouped[:, group, :, 1] .+= weight0 .* values[:, source, :, 1]
        @views grouped[:, group, :, 2] .+= weight2 .* values[:, source, :, 2]
    end
    return grouped
end

"""
    grouped_ring_stokes(grouped, angles, detector_offsets, detector_weights; ...)

Mapmake detector-grouped sufficient statistics using Scanvolver's existing
`Q+iU` convention.  Detector offsets rotate both the spin-2 data moment and
the scan moments.  The standard Qp/Qm/Up/Um quartet makes the joint `h2` and
`h4` vanish, but they are evaluated explicitly so other offset sets remain
well-defined.
"""
function grouped_ring_stokes(grouped::AbstractArray{Complex{T},4}, angles,
                             detector_offsets, detector_weights;
                             rcond::Real=1e-2, unseen::Real=NaN,
                             precision::Type=Float64) where {T}
    precision in (Float32, Float64) || throw(ArgumentError(
        "precision must be Float32 or Float64",
    ))
    isfinite(rcond) && rcond >= 0 || throw(ArgumentError("rcond must be finite and non-negative"))
    active_count = size(grouped, 3)
    size(grouped, 4) == 2 || throw(DimensionMismatch("the scan-weight axis must have length 2"))
    length(angles) == active_count || throw(DimensionMismatch(
        "angles must have one entry per active pixel",
    ))
    normalized = normalized_detector_weights(detector_offsets, detector_weights)
    phase2_mean = sum(normalized .* cis.(2 .* Float64.(detector_offsets)))
    phase4_mean = sum(normalized .* cis.(4 .* Float64.(detector_offsets)))
    output = fill(precision(unseen), size(grouped, 1), size(grouped, 2), 3, active_count)

    for active_index in 1:active_count
        angle = angles[active_index]
        angle.num == 0 && continue
        base_h2 = sum(cis(2 * psi) for psi in angle.ψ) / angle.num
        base_h4 = sum(cis(4 * psi) for psi in angle.ψ) / angle.num
        h2 = base_h2 * phase2_mean
        h4 = base_h4 * phase4_mean
        h_matrix = ComplexF64[
            1.0              0.5*conj(h2)       0.5*h2
            0.5*h2           0.25               0.25*h4
            0.5*conj(h2)     0.25*conj(h4)      0.25
        ]
        condition = cond(h_matrix)
        (!isfinite(condition) || inv(condition) <= rcond) && continue
        factorization = lu(h_matrix)
        for group in axes(grouped, 2), sky in axes(grouped, 1)
            spin0 = ComplexF64(grouped[sky, group, active_index, 1])
            half_spin2 = ComplexF64(grouped[sky, group, active_index, 2]) / 2
            solution = factorization \
                ComplexF64[spin0, half_spin2, conj(half_spin2)]
            output[sky, group, 1, active_index] = precision(real(solution[1]))
            output[sky, group, 2, active_index] = precision(real(solution[2]))
            output[sky, group, 3, active_index] = precision(imag(solution[2]))
        end
    end
    return output
end

struct GroupRingWriter
    file::HDF5.File
    map::HDF5.Dataset
    records::Dict{String,HDF5.Dataset}
    state::HDF5.Dataset
    lock::ReentrantLock
end

function make_writer(file, cp, plan, rings, group_batch_size, T, map_precision,
                     devices, sky_ids, input_beam_ids, group_ids, detector_labels,
                     detector_offsets, detector_weights, metadata,
                     input_beam_metadata, group_metadata, rcond, unseen, l_tile,
                     max_l_tile, memory_fraction, moment_method)
    group_count = cp.numofbeam ÷ length(detector_offsets)
    attrs = attributes(file)
    attrs["schema"] = "EBCoDeCMB.detector_grouped_hdf5.v1"
    attrs["created_utc"] = string(now(UTC))
    attrs["disk_axes"] = "sky,detector_group,stokes,pixel"
    attrs["julia_axes"] = "pixel,stokes,detector_group,sky"
    attrs["pixel_ordering"] = "RING"
    attrs["pixel_index_base"] = 1
    attrs["stokes_order"] = "I,Q,U"
    attrs["polarization_convention"] = "x2=Q+iU"
    attrs["compute_precision"] = string(T)
    attrs["map_precision"] = string(map_precision)
    attrs["ebcodecmb_source"] = pathof(Scanvolver)
    for key in (:nside_output, :lstart, :lstop, :lmax, :bnmax, :κ, :numofsky)
        attrs[String(key)] = getproperty(cp, key)
    end
    attrs["input_numofbeam"] = cp.numofbeam
    attrs["numofdetector_groups"] = group_count
    attrs["detectors_per_group"] = length(detector_offsets)
    for (key, value) in (
        "beta_max" => plan.beta_max, "beta_order" => plan.beta_order,
        "group_batch_size" => group_batch_size,
        "input_beam_batch_size" => group_batch_size * length(detector_offsets),
        "rcond" => rcond, "unseen" => unseen,
        "memory_fraction" => memory_fraction, "l_tile" => string(l_tile),
        "max_l_tile" => max_l_tile, "moment_method" => string(moment_method),
    )
        attrs[key] = value
    end
    file["sky_ids"] = string.(sky_ids)
    file["input_beam_ids"] = string.(input_beam_ids)
    file["detector_group_ids"] = string.(group_ids)
    file["detector_labels"] = collect(string.(detector_labels))
    file["detector_offsets_rad"] = collect(Float64.(detector_offsets))
    file["detector_weights"] = collect(Float64.(detector_weights))
    for (name, values) in (("metadata", metadata),
                           ("input_beam_metadata", input_beam_metadata),
                           ("group_metadata", group_metadata))
        group = create_group(file, name)
        try
            for (key, value) in values
                group[string(key)] = value
            end
        finally
            close(group)
        end
    end
    rings_group = create_group(file, "rings")
    try
        count = ring_count(cp)
        rings_group["pixel_first"] = [first(ring_pixel_range(cp, r)) for r in 1:count]
        rings_group["pixel_last"] = [last(ring_pixel_range(cp, r)) for r in 1:count]
        requested = zeros(UInt8, count)
        requested[rings] .= 1
        rings_group["requested"] = requested
        rings_group["completed"] = zeros(UInt8, count)
        rings_group["device"] = fill(Int32(-1), count)
        rings_group["observed_beta_max"] = fill(NaN, count)
        rings_group["elapsed_seconds"] = zeros(Float64, count)
        for key in ("hit_count", "active_pixels", "l_tile", "tile_count",
                    "scan_moment_builds", "ringd_builds", "sky_fft_builds",
                    "kernel_builds", "beam_applications", "group_reductions",
                    "estimated_tile_bytes", "input_accumulator_bytes",
                    "grouped_accumulator_bytes")
            rings_group[key] = zeros(Int64, count)
        end
    finally
        close(rings_group)
    end
    file["run_state"] = UInt8[0]
    attrs["run_state_codes"] = "0=running,1=complete,2=failed"
    record_names = ("completed", "device", "observed_beta_max", "elapsed_seconds",
                    "hit_count", "active_pixels", "l_tile", "tile_count",
                    "scan_moment_builds", "ringd_builds", "sky_fft_builds",
                    "kernel_builds", "beam_applications", "group_reductions",
                    "estimated_tile_bytes", "input_accumulator_bytes",
                    "grouped_accumulator_bytes")
    records = Dict(name => file["rings/$name"] for name in record_names)
    dims = (cp.resol.numOfPixels, 3, group_count, cp.numofsky)
    chunk = (min(4cp.nside_output, 1024), 3, min(group_batch_size, group_count),
             min(cp.numofsky, 8))
    map = create_dataset(file, "map", map_precision, dims;
                         chunk, fill_value=map_precision(unseen))
    return GroupRingWriter(file, map, records, file["run_state"], ReentrantLock())
end

function close_writer(writer)
    close(writer.map)
    foreach(close, values(writer.records))
    close(writer.state)
end

function write_batch!(writer, geometry, group_indices, stokes, batch_callback)
    pixels = geometry.first_pixel:(geometry.first_pixel + geometry.pixel_count - 1)
    lock(writer.lock) do
        writer.map[pixels, :, group_indices, :] = permutedims(stokes, (4, 3, 2, 1))
        batch_callback === nothing || batch_callback((
            ring=geometry.ring, pixels=pixels, groups=group_indices, stokes=stokes,
        ))
    end
end

function finish_ring!(writer, record, progress_callback; started_ns)
    lock(writer.lock) do
        flush(writer.file)
        for key in keys(record)
            key === :ring && continue
            writer.records[String(key)][record.ring:record.ring] = [getproperty(record, key)]
        end
        elapsed = (time_ns() - started_ns) / 1e9
        writer.records["elapsed_seconds"][record.ring:record.ring] = [elapsed]
        writer.records["completed"][record.ring:record.ring] = UInt8[1]
        flush(writer.file)
        progress_callback === nothing || progress_callback(merge(record, (; elapsed_seconds=elapsed)))
    end
end

function process_ring!(writer, context, cp, buffer, batches, plan, ring,
                       detector_offsets, detector_weights;
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
        observed = BeamBatchedHDF5.check_beta(angles, plan, ring)
        if active === nothing
            finish_ring!(writer, (
                ring=ring, device=context.device, observed_beta_max=observed,
                hit_count=hits, active_pixels=0, l_tile=0, tile_count=0,
                scan_moment_builds=0, ringd_builds=0, sky_fft_builds=0,
                kernel_builds=0, beam_applications=0, group_reductions=0,
                estimated_tile_bytes=0, input_accumulator_bytes=0,
                grouped_accumulator_bytes=0,
            ), progress_callback; started_ns)
            return nothing
        end
        positions = Int.(Array(active.positions))
        pixels = geometry.first_pixel .+ positions .- 1
        ringd = ringd_blocks_recursion_gpu!(context.ringd, cp, geometry.theta)
        moments = scan_moments_gpu(plan, angles; method=moment_method,
                                   memory_fraction=memory_fraction)
        input_bytes = Int128(sizeof(Complex{T})) * cp.numofsky * cp.numofbeam * length(active) * 2
        input_bytes <= memory_fraction * Float64(BeamBatchedHDF5.available_bytes()) ||
            throw(ArgumentError("ring $ring: input-beam accumulators exceed the memory budget"))
        for batch in batches
            push!(accumulators, CUDA.zeros(Complex{T}, cp.numofsky,
                  length(batch.indices), length(active), 2))
        end
        cp_batch = ConvolutionParams(cp; numofbeam=maximum(length(batch.indices) for batch in batches))
        width, tile_bytes = BeamBatchedHDF5.select_tile(
            cp_batch, geometry, active, T, l_tile, max_l_tile, memory_fraction,
        )
        tile_count = 0
        for l_first in cp.lstart:width:cp.lstop
            l_last = min(l_first + width - 1, cp.lstop)
            kernel = sky = active_sky = nothing
            try
                kernel = build_effective_kernel_gpu(context.table, moments, l_first, l_last)
                sky = build_sky_ring_gpu(cp, ringd, context.alm, geometry.phi_pixels,
                                         l_first, l_last; plan_cache=plan_cache)
                active_sky = select_active_sky_ring_gpu(sky, active)
                for (batch, accumulator) in zip(batches, accumulators)
                    effective = nothing
                    try
                        effective = apply_beams_gemm_gpu(
                            context.table, kernel, batch.blm, l_first, l_last,
                        )
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
            grouped = reduce_detector_accumulators(
                Array(accumulator), detector_offsets, detector_weights,
            )
            compact = grouped_ring_stokes(
                grouped, angles, detector_offsets, detector_weights;
                rcond, unseen, precision=map_precision,
            )
            stokes = fill(map_precision(unseen), cp.numofsky, length(batch.groups),
                          3, geometry.pixel_count)
            stokes[:, :, :, positions] = compact
            write_batch!(writer, geometry, batch.groups, stokes, batch_callback)
        end
        group_count = cp.numofbeam ÷ length(detector_offsets)
        grouped_bytes = Int128(sizeof(Complex{T})) * cp.numofsky * group_count * length(active) * 2
        finish_ring!(writer, (
            ring=ring, device=context.device, observed_beta_max=observed,
            hit_count=hits, active_pixels=length(active), l_tile=width,
            tile_count=tile_count, scan_moment_builds=1, ringd_builds=1,
            sky_fft_builds=tile_count, kernel_builds=tile_count,
            beam_applications=tile_count * length(batches),
            group_reductions=length(batches), estimated_tile_bytes=Int64(tile_bytes),
            input_accumulator_bytes=Int64(input_bytes),
            grouped_accumulator_bytes=Int64(grouped_bytes),
        ), progress_callback; started_ns)
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
    convolve_detector_groups_to_hdf5(cp, buffer, alm, blm; output_path, ...)

Convolve every input detector beam, combine consecutive detector groups at the
sufficient-statistic level, then perform one Scanvolver mapmaking solve per
group.  This is deliberately separate from `BeamBatchedHDF5.convolve_to_hdf5`.
The default group order is `Qp,Qm,Up,Um` with offsets
`0, pi/2, pi/4, 3pi/4`.
"""
function convolve_detector_groups_to_hdf5(
    cp::ConvolutionParams, buffer::RingScanBuffer,
    alm::PreparedAlm{T}, blm::PreparedBlm{T};
    output_path::AbstractString, group_batch_size::Integer=4,
    detector_labels=DEFAULT_DETECTOR_LABELS,
    detector_offsets=DEFAULT_DETECTOR_OFFSETS,
    detector_weights=ones(length(detector_offsets)),
    devices=[0], rings=buffer.first_ring:buffer.last_ring,
    beta_order::Integer=8, l_tile=:auto, max_l_tile::Integer=32,
    memory_fraction::Real=0.7, moment_method::Symbol=:gemm,
    map_precision::Type=T, rcond::Real=1e-2, unseen::Real=NaN,
    reclaim_interval::Integer=16,
    sky_ids=string.(1:cp.numofsky),
    input_beam_ids=string.(1:cp.numofbeam),
    group_ids=string.(1:(cp.numofbeam ÷ length(detector_offsets))),
    metadata=Dict{String,Any}(), input_beam_metadata=Dict{String,Any}(),
    group_metadata=Dict{String,Any}(), batch_callback=nothing,
    progress_callback=nothing,
) where {T}
    cp.HWP && throw(ArgumentError("HWP is not supported"))
    cp.noise && throw(ArgumentError("noise is not supported"))
    detector_count = length(detector_offsets)
    length(detector_labels) == detector_count || throw(DimensionMismatch(
        "detector_labels and detector_offsets must have the same length",
    ))
    normalized_detector_weights(detector_offsets, detector_weights)
    cp.numofbeam % detector_count == 0 || throw(ArgumentError(
        "cp.numofbeam must be divisible by the detector count",
    ))
    group_count = cp.numofbeam ÷ detector_count
    T in (Float32, Float64) && map_precision in (Float32, Float64) ||
        throw(ArgumentError("compute/map precision must be Float32 or Float64"))
    for prepared in (alm, blm)
        (prepared.lstart, prepared.lstop, prepared.lmax) ==
            (cp.lstart, cp.lstop, cp.lmax) || throw(ArgumentError(
                "prepared harmonics do not match cp multipoles",
            ))
    end
    alm.numofsky == cp.numofsky && blm.numofbeam == cp.numofbeam &&
        blm.bnmax == cp.bnmax || throw(ArgumentError(
            "prepared sky/beam dimensions do not match cp",
        ))
    buffer.nside == cp.nside_output || throw(ArgumentError("scan-buffer nside does not match cp"))
    group_batch_size > 0 && max_l_tile > 0 || throw(ArgumentError(
        "batch and tile sizes must be positive",
    ))
    l_tile === :auto || (l_tile isa Integer && l_tile > 0) ||
        throw(ArgumentError("invalid l_tile"))
    isfinite(memory_fraction) && 0 < memory_fraction <= 1 ||
        throw(ArgumentError("invalid memory_fraction"))
    isfinite(rcond) && rcond >= 0 || throw(ArgumentError("invalid rcond"))
    isinf(unseen) && throw(ArgumentError("unseen must be finite or NaN"))
    reclaim_interval >= 0 || throw(ArgumentError("invalid reclaim_interval"))
    length(sky_ids) == cp.numofsky || throw(ArgumentError("sky ID count does not match cp"))
    length(input_beam_ids) == cp.numofbeam || throw(ArgumentError(
        "input beam ID count does not match cp",
    ))
    length(group_ids) == group_count || throw(ArgumentError(
        "detector group ID count does not match cp",
    ))
    for (name, values, count) in (("input beam", input_beam_metadata, cp.numofbeam),
                                  ("group", group_metadata, group_count))
        for (key, value) in values
            value isa AbstractVector && length(value) == count || throw(ArgumentError(
                "$name metadata $key must have one value per $name",
            ))
        end
    end
    ring_list = Int.(collect(rings))
    !isempty(ring_list) && allunique(ring_list) &&
        all(ring -> buffer.first_ring <= ring <= buffer.last_ring, ring_list) ||
        throw(ArgumentError("rings must be unique and inside the scan buffer"))
    device_list = Int.(collect(devices))
    !isempty(device_list) && allunique(device_list) &&
        all(device -> 0 <= device < length(CUDA.devices()), device_list) ||
        throw(ArgumentError("invalid devices"))
    Threads.nthreads() >= length(device_list) || throw(ArgumentError(
        "start Julia with at least one thread per GPU",
    ))
    path = abspath(output_path)
    ispath(path) && throw(ArgumentError("output already exists: $path"))
    isdir(dirname(path)) || throw(ArgumentError("output directory does not exist"))

    batches = detector_group_batches(blm, detector_count, group_batch_size)
    plan = BeamBatchedHDF5.automatic_scan_plan(cp; beta_order, precision=T)
    table = build_wigner_chebyshev_table_cpu(cp, plan)
    contexts = []
    try
        for device in device_list
            push!(contexts, BeamBatchedHDF5.prepare_context(cp, alm, table, device, T))
        end
        h5open(path, "w") do file
            writer = make_writer(
                file, cp, plan, ring_list, Int(group_batch_size), T, map_precision,
                device_list, sky_ids, input_beam_ids, group_ids, detector_labels,
                detector_offsets, detector_weights, metadata, input_beam_metadata,
                group_metadata, rcond, unseen, l_tile, max_l_tile,
                memory_fraction, moment_method,
            )
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
                                process_ring!(
                                    writer, context, cp, buffer, batches, plan,
                                    ring_list[index], detector_offsets, detector_weights;
                                    l_tile, max_l_tile, memory_fraction, moment_method,
                                    rcond, unseen, map_precision, batch_callback,
                                    progress_callback,
                                )
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
        foreach(BeamBatchedHDF5.free_context, contexts)
    end
    return path
end

"""Read a completed ring as `(sky, detector_group, I/Q/U, ring_pixel)`."""
function read_group_ring_hdf5(path, ring::Integer; skies=nothing, groups=nothing)
    return h5open(path, "r") do file
        completed = read(file["rings/completed"])
        1 <= ring <= length(completed) && completed[ring] == 1 || throw(ArgumentError(
            "ring $ring has not been completely saved",
        ))
        first_pixel = file["rings/pixel_first"][ring]
        last_pixel = file["rings/pixel_last"][ring]
        dataset = file["map"]
        try
            sky_range = skies === nothing ? (1:size(dataset, 4)) : skies
            group_range = groups === nothing ? (1:size(dataset, 3)) : groups
            sky_range isa AbstractUnitRange && group_range isa AbstractUnitRange ||
                throw(ArgumentError("sky/group selections must be ranges"))
            return permutedims(
                dataset[first_pixel:last_pixel, :, group_range, sky_range],
                (4, 3, 2, 1),
            )
        finally
            close(dataset)
        end
    end
end

end # module
