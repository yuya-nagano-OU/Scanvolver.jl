"""
    RingScanBuffer

A pointing store ordered by HEALPix pixel, so that one ring is a contiguous
slice and one pixel a contiguous sub-slice of it.

`make_ring_angles(cp, geometry, phi, theta, psi)` walks the whole time stream
once per ring, which is fine for a single ring and hopeless for a sphere: a
year of samples would be traversed `4*nside - 1` times. This bins every sample
once instead.

Only `phi`, `theta` and `psi` are retained, which is what the HWP-free
convolution needs; supporting `HWP=true` would mean carrying sample times as a
fourth array.

`pixel_offsets` is indexed relative to `first_pixel`, so a ring range much
narrower than the sphere does not pay for the whole map. Samples inside one
pixel are in arbitrary order: the reordering is not stable, and nothing
downstream depends on the order because the scan moments are plain sums over
hits.
"""
struct RingScanBuffer
    nside::Int
    first_ring::Int
    last_ring::Int
    first_pixel::Int
    last_pixel::Int
    processed_samples::Int
    stored_samples::Int
    phi::Vector{Float64}
    theta::Vector{Float64}
    psi::Vector{Float64}
    pixel_offsets::Vector{Int}
end

Base.length(buffer::RingScanBuffer) = buffer.stored_samples

function Base.show(io::IO, buffer::RingScanBuffer)
    print(
        io,
        "RingScanBuffer(nside=$(buffer.nside), ",
        "rings=$(buffer.first_ring):$(buffer.last_ring), ",
        "$(buffer.stored_samples)/$(buffer.processed_samples) samples, ",
        "$(round(ring_scan_buffer_bytes(buffer) / 2^20; digits=1)) MiB)",
    )
end

"""
    ring_scan_buffer_bytes(buffer)

Bytes held by the retained pointing arrays and the pixel offsets. Julia object
headers and any transient ingestion chunk are not counted.
"""
function ring_scan_buffer_bytes(buffer::RingScanBuffer)
    return sizeof(buffer.phi) +
        sizeof(buffer.theta) +
        sizeof(buffer.psi) +
        sizeof(buffer.pixel_offsets)
end

"""
    ring_hit_count(buffer, ring)

Number of retained samples that fall in `ring`.
"""
function ring_hit_count(buffer::RingScanBuffer, ring::Integer)
    ring_index = Int(ring)
    buffer.first_ring <= ring_index <= buffer.last_ring ||
        throw(ArgumentError(
            "ring $ring_index is outside the buffered range " *
            "$(buffer.first_ring):$(buffer.last_ring)",
        ))
    info = getringinfo(Resolution(buffer.nside), ring_index)
    first_pixel = Int(info.firstPixIdx)
    last_pixel = first_pixel + Int(info.numOfPixels) - 1
    return buffer.pixel_offsets[last_pixel - buffer.first_pixel + 2] -
        buffer.pixel_offsets[first_pixel - buffer.first_pixel + 1]
end

function _grow_scan_arrays!(phi, theta, psi, pixels, required)
    required <= length(phi) && return nothing
    capacity = length(phi)
    new_capacity = max(required, max(1024, capacity + capacity ÷ 2))
    resize!(phi, new_capacity)
    resize!(theta, new_capacity)
    resize!(psi, new_capacity)
    resize!(pixels, new_capacity)
    return nothing
end

"""
    _sort_scan_by_pixel!(phi, theta, psi, pixels, counts) -> offsets

In-place American-flag sort into pixel buckets. Doing this in place rather than
scattering into a second set of arrays halves the peak memory: at 19 Hz for a
year the pointing arrays alone are about 14 GB.

`pixels` holds 1-based bucket indices, not HEALPix pixel indices.
"""
function _sort_scan_by_pixel!(
    phi::Vector{Float64},
    theta::Vector{Float64},
    psi::Vector{Float64},
    pixels::Vector{Int32},
    counts::Vector{Int},
)
    bucket_count = length(counts)
    offsets = Vector{Int}(undef, bucket_count + 1)
    offsets[1] = 1
    @inbounds for bucket in 1:bucket_count
        offsets[bucket + 1] = offsets[bucket] + counts[bucket]
    end
    next_positions = copy(@view(offsets[1:bucket_count]))

    @inbounds for target in 1:bucket_count
        position = next_positions[target]
        bucket_stop = offsets[target + 1] - 1
        while position <= bucket_stop
            bucket = Int(pixels[position])
            if bucket == target
                position += 1
                next_positions[target] = position
                continue
            end
            destination = next_positions[bucket]
            phi[position], phi[destination] =
                phi[destination], phi[position]
            theta[position], theta[destination] =
                theta[destination], theta[position]
            psi[position], psi[destination] =
                psi[destination], psi[position]
            pixels[position], pixels[destination] =
                pixels[destination], pixels[position]
            next_positions[bucket] = destination + 1
        end
    end
    return offsets
end

function _check_scan_chunk(phi, theta, psi)
    sample_count = length(phi)
    length(theta) == sample_count || throw(DimensionMismatch(
        "theta has $(length(theta)) samples; phi has $sample_count",
    ))
    length(psi) == sample_count || throw(DimensionMismatch(
        "psi has $(length(psi)) samples; phi has $sample_count",
    ))
    return sample_count
end

"""
    build_ring_scan_buffer(
        cp, chunks;
        rings=1:ring_count(cp), capacity_hint=nothing,
        progress_callback=nothing,
    ) -> RingScanBuffer

Bin a scan into pixel order in one pass.

`chunks` is any iterable of `(phi, theta, psi)` pointing chunks, so the caller
decides how the time stream is produced and how large a chunk is; nothing about
a scan strategy enters here. Samples outside `rings` are counted as processed
and then discarded.

`capacity_hint` presizes the retained arrays. Without it the arrays grow by
half each time, which copies roughly three times the final size in total --
noticeable, though not fatal, for a year of samples.

`progress_callback` receives a NamedTuple after every chunk with
`chunk_index`, `processed_samples`, `stored_samples` and `capacity`.
"""
function build_ring_scan_buffer(
    cp::ConvolutionParams,
    chunks;
    rings::AbstractUnitRange{<:Integer}=1:ring_count(cp),
    capacity_hint::Union{Nothing,Integer}=nothing,
    progress_callback=nothing,
)
    cp.HWP == false || throw(ArgumentError(
        "build_ring_scan_buffer currently supports HWP=false; sample times " *
        "would have to be retained as a fourth array",
    ))
    isempty(rings) && throw(ArgumentError("rings must be non-empty"))
    first_ring = Int(first(rings))
    last_ring = Int(last(rings))
    first_pixel, _ = ring_pixel_range(cp, first_ring)
    _, last_pixel = ring_pixel_range(cp, last_ring)
    bucket_count = last_pixel - first_pixel + 1

    capacity = capacity_hint === nothing ? 1024 : max(1, Int(capacity_hint))
    phi_store = Vector{Float64}(undef, capacity)
    theta_store = Vector{Float64}(undef, capacity)
    psi_store = Vector{Float64}(undef, capacity)
    pixel_store = Vector{Int32}(undef, capacity)
    counts = zeros(Int, bucket_count)

    processed_samples = 0
    stored_samples = 0
    chunk_index = 0
    for chunk in chunks
        phi, theta, psi = chunk
        sample_count = _check_scan_chunk(phi, theta, psi)
        _grow_scan_arrays!(
            phi_store,
            theta_store,
            psi_store,
            pixel_store,
            stored_samples + sample_count,
        )
        next_position = stored_samples + 1
        @inbounds for sample_index in eachindex(phi, theta, psi)
            pixel = ang2pixRing(
                cp.resol,
                Float64(theta[sample_index]),
                Float64(phi[sample_index]),
            )
            first_pixel <= pixel <= last_pixel || continue
            phi_store[next_position] = Float64(phi[sample_index])
            theta_store[next_position] = Float64(theta[sample_index])
            psi_store[next_position] = Float64(psi[sample_index])
            bucket = pixel - first_pixel + 1
            pixel_store[next_position] = Int32(bucket)
            counts[bucket] += 1
            next_position += 1
        end
        processed_samples += sample_count
        stored_samples = next_position - 1
        chunk_index += 1
        progress_callback === nothing || progress_callback((
            chunk_index=chunk_index,
            processed_samples=processed_samples,
            stored_samples=stored_samples,
            capacity=length(phi_store),
        ))
    end

    resize!(phi_store, stored_samples)
    resize!(theta_store, stored_samples)
    resize!(psi_store, stored_samples)
    resize!(pixel_store, stored_samples)
    offsets = _sort_scan_by_pixel!(
        phi_store,
        theta_store,
        psi_store,
        pixel_store,
        counts,
    )

    return RingScanBuffer(
        cp.nside_output,
        first_ring,
        last_ring,
        first_pixel,
        last_pixel,
        processed_samples,
        stored_samples,
        phi_store,
        theta_store,
        psi_store,
        offsets,
    )
end

"""
    build_ring_scan_buffer(cp, phi, theta, psi; kwargs...)

Convenience method for a scan that is already in memory as one chunk.
"""
function build_ring_scan_buffer(
    cp::ConvolutionParams,
    phi::AbstractVector{<:Real},
    theta::AbstractVector{<:Real},
    psi::AbstractVector{<:Real};
    kwargs...,
)
    return build_ring_scan_buffer(
        cp,
        ((phi, theta, psi),);
        capacity_hint=length(phi),
        kwargs...,
    )
end

"""
    make_ring_angles(cp, geometry, buffer) -> (active, angles)

Read one ring out of a pixel-ordered scan buffer. Returns exactly what the
time-stream method returns, in the same ascending order, so it drops straight
into `convolve_ring_gpu`: `(nothing, Angles[])` for a ring with no hits.

Samples within a pixel come out in the buffer's order rather than in time
order, which changes nothing downstream but does reorder the floating-point
sums by a few ulp.
"""
function make_ring_angles(
    cp::ConvolutionParams,
    geometry::RingGeometry,
    buffer::RingScanBuffer,
)
    buffer.nside == cp.nside_output || throw(ArgumentError(
        "buffer was built at nside=$(buffer.nside); cp has " *
        "nside=$(cp.nside_output)",
    ))
    buffer.first_ring <= geometry.ring <= buffer.last_ring ||
        throw(ArgumentError(
            "ring $(geometry.ring) is outside the buffered range " *
            "$(buffer.first_ring):$(buffer.last_ring)",
        ))

    positions = Int[]
    angles = Angles[]
    for position in 1:geometry.pixel_count
        pixel = geometry.first_pixel + position - 1
        bucket = pixel - buffer.first_pixel + 1
        first_sample = buffer.pixel_offsets[bucket]
        last_sample = buffer.pixel_offsets[bucket + 1] - 1
        last_sample >= first_sample || continue
        span = first_sample:last_sample
        push!(positions, position)
        push!(
            angles,
            Angles(
                cp;
                pix=pixel,
                φ=buffer.phi[span],
                θ=buffer.theta[span],
                ψ=buffer.psi[span],
            ),
        )
    end
    isempty(positions) && return nothing, Angles[]
    return ActiveRingPixels(positions, geometry.pixel_count), angles
end
