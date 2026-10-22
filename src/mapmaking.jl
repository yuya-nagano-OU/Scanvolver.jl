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
    ring_stokes(result, angles; rcond=1e-2, unseen=NaN, precision=Float64)

Solve one ring for `(I, Q, U)` into a ring-local
`(sky, beam, stokes, active_pixel)` array, in the order of `result.positions`.

`result.accumulated[:, :, k, 1]` and `[:, :, k, 2]` are the hit-averaged
`<S>` and `<S e^{2i psi}>` of the pixel at `result.positions[k]`, and `angles[k]`
must be that pixel's scan geometry, in the order `make_ring_angles` returns.

With `h2 = <e^{2i psi}>` and `h4 = <e^{4i psi}>` the per-pixel system is

```
[ 1             0.5*conj(h2)   0.5*h2     ] [ x1 ]   [ <S>                    ]
[ 0.5*h2        0.25           0.25*h4    ] [ x2 ] = [ <S e^{2i psi}>/2       ]
[ 0.5*conj(h2)  0.25*conj(h4)  0.25       ] [ x3 ]   [ conj(<S e^{2i psi}>/2) ]
```

which is the ideal-detector model `S = I + Q cos(2 psi) + U sin(2 psi)` written
with `x1 = I`, `x2 = Q + iU` and `x3 = conj(x2)`. Hence `I = real(x1)`,
`Q = real(x2)` and `U = imag(x2)`. The matrix, the placement of the two
conjugations, and the reciprocal-condition rejection are kept exactly as in the
reference CPU implementation.

A pixel is written as `unseen` when it has no hits or when the reciprocal
condition number of the system is at most `rcond`, that is when the scan does
not cover enough polarization angles to separate `Q` from `U`.
The default `1e-2` is intentionally conservative for production maps; pass a
smaller value explicitly only when retaining poorly conditioned pixels is
useful for diagnostics.

The system depends only on `psi`, so it is factorized once per pixel and reused
by every sky and beam. One ring is the bounded form of the result: at nside 512
a full-sky map for 22 skies and 100 beams is about 83 GB in Float32, while one
ring of it is a few MB, so the HDF5 pipelines write each ring as it is solved.
"""
function ring_stokes(
    result::RingConvolution{T},
    angles::AbstractVector{<:Angles};
    rcond::Real=1e-2,
    unseen::Real=NaN,
    precision::Type=Float64,
) where {T<:_HARMONIC_PRECISIONS}
    precision in (Float32, Float64) || throw(ArgumentError(
        "precision must be Float32 or Float64, got $precision",
    ))
    active_count = length(result.positions)
    length(angles) == active_count || throw(DimensionMismatch(
        "got $(length(angles)) Angles for $active_count active pixels; " *
        "they must be in the order make_ring_angles returns",
    ))
    rcond >= 0 || throw(ArgumentError(
        "rcond must be non-negative, got $rcond",
    ))

    sky_count = size(result.accumulated, 1)
    beam_count = size(result.accumulated, 2)
    stokes = fill(
        precision(unseen),
        sky_count,
        beam_count,
        3,
        active_count,
    )
    values = Array(result.accumulated)

    for active_index in 1:active_count
        angle = angles[active_index]
        angle.num == 0 && continue

        inverse_hits = 1.0 / angle.num
        h2 = inverse_hits * sum(cis(2 * psi) for psi in angle.ψ)
        h4 = inverse_hits * sum(cis(4 * psi) for psi in angle.ψ)
        h_matrix = ComplexF64[
            1.0              0.5*conj(h2)     0.5*h2
            0.5*h2           0.25             0.25*h4
            0.5*conj(h2)     0.25*conj(h4)    0.25
        ]
        condition = cond(h_matrix)
        (!isfinite(condition) || inv(condition) <= rcond) && continue

        factorization = lu(h_matrix)
        for beam_index in 1:beam_count, sky_index in 1:sky_count
            spin0 = ComplexF64(
                values[sky_index, beam_index, active_index, 1],
            )
            half_spin2 = ComplexF64(
                values[sky_index, beam_index, active_index, 2],
            ) / 2
            solution = factorization \
                ComplexF64[spin0, half_spin2, conj(half_spin2)]
            stokes[sky_index, beam_index, 1, active_index] =
                precision(real(solution[1]))
            stokes[sky_index, beam_index, 2, active_index] =
                precision(real(solution[2]))
            stokes[sky_index, beam_index, 3, active_index] =
                precision(imag(solution[2]))
        end
    end

    return stokes
end
