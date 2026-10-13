"""
    RingGeometry

The spatial geometry of one HEALPix ring in RING ordering.

`phi_pixels` is ascending and equally spaced, which is what makes the inverse
FFT in `build_sky_ring_gpu` exact. `theta` is constant across the ring, so one
ringD stack serves every pixel of it.
"""
struct RingGeometry
    ring::Int
    first_pixel::Int
    pixel_count::Int
    theta::Float64
    phi_pixels::Vector{Float64}
end

Base.length(geometry::RingGeometry) = geometry.pixel_count

function Base.show(io::IO, geometry::RingGeometry)
    print(
        io,
        "RingGeometry(ring=$(geometry.ring), ",
        "pixels=$(geometry.first_pixel):",
        "$(geometry.first_pixel + geometry.pixel_count - 1), ",
        "theta=$(round(geometry.theta; digits=6)))",
    )
end

function _check_ring_index(cp::ConvolutionParams, ring::Integer)
    ring_count = numOfRings(cp.resol)
    1 <= ring <= ring_count || throw(ArgumentError(
        "ring must be in 1:$ring_count for nside=$(cp.nside_output), " *
        "got $ring",
    ))
    return Int(ring)
end

"""
    ring_count(cp) -> Int

Number of HEALPix rings, `4*nside - 1`.
"""
ring_count(cp::ConvolutionParams) = numOfRings(cp.resol)

"""
    ring_pixel_range(cp, ring) -> (first_pixel, last_pixel)

Inclusive 1-based RING-ordered pixel range of one ring.
"""
function ring_pixel_range(cp::ConvolutionParams, ring::Integer)
    ring_index = _check_ring_index(cp, ring)
    info = getringinfo(cp.resol, ring_index)
    first_pixel = Int(info.firstPixIdx)
    return first_pixel, first_pixel + Int(info.numOfPixels) - 1
end

"""
    ring_geometry(cp, ring) -> RingGeometry

Build the colatitude and the ascending ring longitudes of one ring. The
longitudes come from `pix2angRing`, so they are the exact pixel centres rather
than a reconstructed grid.
"""
function ring_geometry(cp::ConvolutionParams, ring::Integer)
    ring_index = _check_ring_index(cp, ring)
    info = getringinfo(cp.resol, ring_index)
    first_pixel = Int(info.firstPixIdx)
    pixel_count = Int(info.numOfPixels)
    phi_pixels = Vector{Float64}(undef, pixel_count)
    @inbounds for pixel_index in 1:pixel_count
        _, phi = pix2angRing(
            cp.resol,
            first_pixel + pixel_index - 1,
        )
        phi_pixels[pixel_index] = phi
    end
    return RingGeometry(
        ring_index,
        first_pixel,
        pixel_count,
        Float64(info.colatitude_rad),
        phi_pixels,
    )
end

function _check_ring_scan_inputs(
    cp::ConvolutionParams,
    phi::AbstractVector{<:Real},
    theta::AbstractVector{<:Real},
    psi::AbstractVector{<:Real},
    time_array::Union{Nothing,AbstractVector{<:Real}},
)
    sample_count = length(phi)
    length(theta) == sample_count || throw(DimensionMismatch(
        "theta must have length $sample_count, got $(length(theta))",
    ))
    length(psi) == sample_count || throw(DimensionMismatch(
        "psi must have length $sample_count, got $(length(psi))",
    ))
    if time_array === nothing
        cp.HWP && throw(ArgumentError(
            "time_array must be provided when HWP=true",
        ))
    else
        length(time_array) == sample_count || throw(DimensionMismatch(
            "time_array must have length $sample_count, got " *
            "$(length(time_array))",
        ))
    end
    return sample_count
end

"""
    make_ring_angles(cp, geometry, phi, theta, psi; time_array=nothing)
        -> (active, angles)

Bin one scan into the pixels of a single ring and build the per-pixel scan
geometry.

Every sample is assigned to a pixel with `ang2pixRing`; samples outside this
ring are ignored. Ring positions with at least one hit become `active`, and
`angles[k]` is the `Angles` of the pixel at `active.positions[k]`. **The two are
returned in the same ascending order**, which is what aligns the gathered sky
ring with the effective beam along the pixel axis.

Returns `(nothing, Angles[])` for a ring with no hits.

This scans the whole time stream once per ring, so it is a correctness-first
entry point for a single ring rather than the way to process a full sphere; a
ring-bucketed scan buffer belongs to a later stage.
"""
function make_ring_angles(
    cp::ConvolutionParams,
    geometry::RingGeometry,
    phi::AbstractVector{<:Real},
    theta::AbstractVector{<:Real},
    psi::AbstractVector{<:Real};
    time_array::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    _check_ring_scan_inputs(cp, phi, theta, psi, time_array)
    first_pixel = geometry.first_pixel
    last_pixel = first_pixel + geometry.pixel_count - 1

    samples_by_position = [Int[] for _ in 1:geometry.pixel_count]
    @inbounds for sample_index in eachindex(phi, theta, psi)
        pixel = ang2pixRing(
            cp.resol,
            Float64(theta[sample_index]),
            Float64(phi[sample_index]),
        )
        first_pixel <= pixel <= last_pixel || continue
        push!(
            samples_by_position[pixel - first_pixel + 1],
            sample_index,
        )
    end

    positions = findall(
        position -> !isempty(samples_by_position[position]),
        1:geometry.pixel_count,
    )
    isempty(positions) && return nothing, Angles[]

    angles = Vector{Angles}(undef, length(positions))
    for (active_index, position) in pairs(positions)
        hits = samples_by_position[position]
        angles[active_index] = Angles(
            cp;
            pix=first_pixel + position - 1,
            φ=Float64[Float64(phi[hit]) for hit in hits],
            θ=Float64[Float64(theta[hit]) for hit in hits],
            ψ=Float64[Float64(psi[hit]) for hit in hits],
            time_array=time_array === nothing ?
                nothing :
                Float64[Float64(time_array[hit]) for hit in hits],
        )
    end

    return ActiveRingPixels(positions, geometry.pixel_count), angles
end
