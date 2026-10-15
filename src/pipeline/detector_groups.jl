# Detector groups: consecutive detector beams combined at the sufficient-
# statistic level and solved once per group. A map layout of the same machinery
# as convolve_to_hdf5 (DetectorGroupMaps).

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
    source_batches = beam_batches(
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

"""One `(I, Q, U)` map per group of consecutive detectors."""
struct DetectorGroupMaps{O,W} <: MapLayout
    group_batch_size::Int
    labels
    offsets::O
    weights::W
    input_beam_ids
    group_ids
    input_beam_metadata
    group_metadata
end

output_count(layout::DetectorGroupMaps, cp) = cp.numofbeam ÷ length(layout.offsets)
output_batch_size(layout::DetectorGroupMaps) = layout.group_batch_size
output_indices(::DetectorGroupMaps, batch) = batch.groups
output_name(::DetectorGroupMaps) = :groups

function batch_stokes(layout::DetectorGroupMaps, accumulator, geometry, positions, pixels,
                      angles; rcond, unseen, map_precision)
    grouped = reduce_detector_accumulators(Array(accumulator), layout.offsets, layout.weights)
    return grouped_ring_stokes(grouped, angles, layout.offsets, layout.weights;
                               rcond, unseen, precision=map_precision)
end

function extra_record(layout::DetectorGroupMaps, cp, T, active_count, batches)
    grouped_bytes = Int128(sizeof(Complex{T})) * cp.numofsky * output_count(layout, cp) *
                    active_count * 2
    return (group_reductions=length(batches),
            input_accumulator_bytes=Int64(accumulator_bytes(cp, T, active_count)),
            grouped_accumulator_bytes=Int64(grouped_bytes))
end

function write_layout!(file, layout::DetectorGroupMaps, cp, devices)
    attrs = attributes(file)
    attrs["schema"] = "EBCoDeCMB.detector_grouped_hdf5.v1"
    attrs["disk_axes"] = "sky,detector_group,stokes,pixel"
    attrs["julia_axes"] = "pixel,stokes,detector_group,sky"
    attrs["polarization_convention"] = "x2=Q+iU"
    attrs["input_numofbeam"] = cp.numofbeam
    attrs["numofdetector_groups"] = output_count(layout, cp)
    attrs["detectors_per_group"] = length(layout.offsets)
    attrs["group_batch_size"] = layout.group_batch_size
    attrs["input_beam_batch_size"] = layout.group_batch_size * length(layout.offsets)
    file["input_beam_ids"] = string.(layout.input_beam_ids)
    file["detector_group_ids"] = string.(layout.group_ids)
    file["detector_labels"] = collect(string.(layout.labels))
    file["detector_offsets_rad"] = collect(Float64.(layout.offsets))
    file["detector_weights"] = collect(Float64.(layout.weights))
    write_metadata_group!(file, "input_beam_metadata", layout.input_beam_metadata)
    write_metadata_group!(file, "group_metadata", layout.group_metadata)
    return ("group_reductions", "input_accumulator_bytes", "grouped_accumulator_bytes")
end

"""
    convolve_detector_groups_to_hdf5(cp, buffer, alm, blm; output_path, ...)

Convolve every input detector beam, combine consecutive detector groups at the
sufficient-statistic level, then perform one Scanvolver mapmaking solve per
group. The ring queue, tiling and HDF5 bookkeeping are those of
`convolve_to_hdf5`; only the per-batch solve and the file
metadata differ. The default group order is `Qp,Qm,Up,Um` with offsets
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
    detector_count = length(detector_offsets)
    length(detector_labels) == detector_count || throw(DimensionMismatch(
        "detector_labels and detector_offsets must have the same length",
    ))
    normalized_detector_weights(detector_offsets, detector_weights)
    cp.numofbeam % detector_count == 0 || throw(ArgumentError(
        "cp.numofbeam must be divisible by the detector count",
    ))
    group_count = cp.numofbeam ÷ detector_count
    ring_list, device_list, path = check_run_inputs(
        cp, buffer, alm, blm; map_precision, rings, devices, output_path, l_tile,
        max_l_tile, memory_fraction, rcond, unseen, reclaim_interval)
    group_batch_size > 0 || throw(ArgumentError("batch and tile sizes must be positive"))
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
    layout = DetectorGroupMaps(Int(group_batch_size), detector_labels, detector_offsets,
                               detector_weights, input_beam_ids, group_ids,
                               input_beam_metadata, group_metadata)
    batches = detector_group_batches(blm, detector_count, group_batch_size)
    return run_rings(
        layout, cp, buffer, alm, blm, batches, path, ring_list, device_list;
        beta_order, l_tile, max_l_tile, memory_fraction, moment_method, map_precision,
        rcond, unseen, reclaim_interval, sky_ids, metadata, batch_callback,
        progress_callback)
end

"""Read a completed ring as `(sky, detector_group, I/Q/U, ring_pixel)`."""
read_group_ring_hdf5(path, ring::Integer; skies=nothing, groups=nothing) =
    read_ring_maps(path, ring, skies, groups, "group")
