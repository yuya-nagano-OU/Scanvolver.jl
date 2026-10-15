# The HDF5 file a run writes, shared by both drivers: what a map layout must
# provide, the writer (attributes, ring records, chunked map, one lock for all
# HDF5 access), and the ring reader.

"""
    MapLayout

What a run writes for each batch of beams: `BeamMaps` one `(I, Q, U)` map per
beam (`convolve_to_hdf5`), `DetectorGroupMaps` one per group of detectors
(`convolve_detector_groups_to_hdf5`). The ring queue, the tiling, the HDF5
bookkeeping and the reader are shared; a layout only says how a batch is solved
and what the file records about it, through `output_count`, `output_batch_size`,
`output_indices`, `output_name`, `batch_stokes`, `extra_record` and
`write_layout!`.
"""
abstract type MapLayout end

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
