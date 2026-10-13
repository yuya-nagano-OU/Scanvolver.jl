"""
    ActiveRingPixels

The scan-hit pixels of one HEALPix ring, resident on the device.

`positions` holds strictly ascending, 1-based positions within the ring. It is
built once per ring and reused by every `l` tile, so `select_active_sky_ring_gpu`
never transfers indices back to the host.

`complete` records whether the ring is fully hit. It is derived on the host at
construction: strictly ascending, unique positions inside `1:ring_pixel_count`
whose count equals `ring_pixel_count` can only be `1:ring_pixel_count`. A
complete ring therefore needs no gather at all.
"""
struct ActiveRingPixels
    positions::CuVector{Int32}
    ring_pixel_count::Int
    active_count::Int
    complete::Bool
end

Base.length(active::ActiveRingPixels) = active.active_count

function Base.show(io::IO, active::ActiveRingPixels)
    print(
        io,
        "ActiveRingPixels($(active.active_count)/",
        "$(active.ring_pixel_count) pixels",
        active.complete ? ", complete)" : ")",
    )
end

"""
    ActiveRingPixels(positions, ring_pixel_count; device_id=nothing)

Upload the hit positions of one ring. `positions` must be strictly ascending,
1-based, and contained in `1:ring_pixel_count`.
"""
function ActiveRingPixels(
    positions::AbstractVector{<:Integer},
    ring_pixel_count::Integer;
    device_id::Union{Nothing,Integer}=nothing,
)
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    total_pixels = Int(ring_pixel_count)
    total_pixels > 0 || throw(ArgumentError(
        "ring_pixel_count must be positive, got $total_pixels",
    ))
    total_pixels <= typemax(Int32) || throw(ArgumentError(
        "ring_pixel_count must fit in Int32, got $total_pixels",
    ))
    active_count = length(positions)
    active_count > 0 || throw(ArgumentError(
        "positions must contain at least one active pixel",
    ))
    active_count <= total_pixels || throw(ArgumentError(
        "positions has $active_count entries, more than the " *
        "$total_pixels pixels in the ring",
    ))

    previous = 0
    @inbounds for position in positions
        value = Int(position)
        1 <= value <= total_pixels || throw(ArgumentError(
            "ring position $value is outside 1:$total_pixels",
        ))
        value > previous || throw(ArgumentError(
            "positions must be strictly ascending; $value follows " *
            "$previous",
        ))
        previous = value
    end

    device_id === nothing || CUDA.device!(Int(device_id))
    return ActiveRingPixels(
        CuArray(Int32[Int32(position) for position in positions]),
        total_pixels,
        active_count,
        active_count == total_pixels,
    )
end

function _free_active_ring_pixels!(active::ActiveRingPixels)
    CUDA.unsafe_free!(active.positions)
    return nothing
end

function _gather_active_sky_ring_kernel!(
    active,
    full,
    positions,
    sky_count::Int64,
    term_count::Int64,
    active_count::Int64,
)
    linear_index =
        (Int64(blockIdx().x) - Int64(1)) * Int64(blockDim().x) +
        Int64(threadIdx().x)
    total = sky_count * term_count * active_count
    linear_index > total && return nothing
    index = linear_index - Int64(1)
    sky_index = index % sky_count + Int64(1)
    index ÷= sky_count
    term_index = index % term_count + Int64(1)
    active_index = index ÷ term_count + Int64(1)
    @inbounds ring_position = Int64(positions[active_index])
    @inbounds active[sky_index, term_index, active_index] =
        full[sky_index, term_index, ring_position]
    return nothing
end

"""
    select_active_sky_ring_gpu(sky_ring, active) -> CuArray

Select the scan-hit columns of one sky-ring tile:

`U_active[sky, term, active_pixel] = U_full[sky, term, positions[active_pixel]]`.

`sky_ring` covers every pixel of the ring because the inverse FFT is defined on
the complete longitude grid, while the effective beam exists only for hit
pixels. This gather is what makes the two operands of
`accumulate_convolution_tile_gemm_gpu!` share one pixel axis. Only the pixel
axis is touched: the `(spin, M, l)` term axis is copied verbatim, so one
`ActiveRingPixels` is valid for every `l` tile of the ring, including a final
tile of different width.

**When `active.complete` is true the input is returned unchanged**, with no
kernel launch and no allocation. The result then aliases `sky_ring`, so release
it with `free_active_sky_ring!` rather than `CUDA.unsafe_free!`.
"""
function select_active_sky_ring_gpu(
    sky_ring::CuArray{Complex{T},3},
    active::ActiveRingPixels,
) where {T<:_HARMONIC_PRECISIONS}
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    sky_count, term_count, pixel_count = size(sky_ring)
    pixel_count == active.ring_pixel_count || throw(DimensionMismatch(
        "sky_ring has $pixel_count pixels; ActiveRingPixels describes a " *
        "ring of $(active.ring_pixel_count)",
    ))
    sky_count > 0 || throw(ArgumentError(
        "sky_ring must contain at least one sky",
    ))
    term_count > 0 || throw(ArgumentError(
        "the (spin, M, l) term axis must be non-empty",
    ))

    active.complete && return sky_ring

    total =
        Int128(sky_count) * term_count * active.active_count
    total <= typemax(Int64) || throw(ArgumentError(
        "active sky-ring tile has $total elements, exceeding Int64 " *
        "indexing; reduce the l tile",
    ))

    gathered = CUDA.zeros(
        Complex{T},
        sky_count,
        term_count,
        active.active_count,
    )
    succeeded = false
    try
        @cuda threads=_SKY_RING_THREADS blocks=cld(
            Int(total),
            _SKY_RING_THREADS,
        ) _gather_active_sky_ring_kernel!(
            gathered,
            sky_ring,
            active.positions,
            Int64(sky_count),
            Int64(term_count),
            Int64(active.active_count),
        )
        CUDA.synchronize()
        succeeded = true
        return gathered
    finally
        succeeded || CUDA.unsafe_free!(gathered)
    end
end

"""
    free_active_sky_ring!(active_sky_ring, sky_ring)

Release the result of `select_active_sky_ring_gpu` without double-freeing the
full-ring tile it may alias. On a complete ring the two arguments are the same
object and nothing is freed; the caller still owns `sky_ring`.
"""
function free_active_sky_ring!(
    active_sky_ring::CuArray,
    sky_ring::CuArray,
)
    active_sky_ring === sky_ring && return nothing
    CUDA.unsafe_free!(active_sky_ring)
    return nothing
end
