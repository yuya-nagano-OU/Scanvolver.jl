# cuBLAS is reached through the same guarded alias used by the effective-beam
# GEMM path, so a future CUDA.jl submodule rename is handled in one place per
# file. Unifying the three current spellings is a separate cleanup.
const _CONVOLUTION_CUBLAS = isdefined(CUDA, :cuBLAS) ?
    CUDA.cuBLAS : CUDA.CUBLAS

# The HWP=false convolution retains exactly the two scan weights s=0 and s=2.
const _CONVOLUTION_SCAN_WEIGHTS = 2

function _check_convolution_tile_inputs(
    accumulated::CuArray{Complex{T},4},
    sky_ring::CuArray{Complex{T},3},
    effective_beam::CuArray{Complex{T},4},
) where {T<:_HARMONIC_PRECISIONS}
    sky_count, term_count, pixel_count = size(sky_ring)
    beam_count = size(effective_beam, 2)

    size(effective_beam, 1) == term_count || throw(DimensionMismatch(
        "effective_beam has $(size(effective_beam, 1)) terms; sky_ring " *
        "has $term_count. Both must describe the same (spin, M, l) tile",
    ))
    size(effective_beam, 3) == pixel_count || throw(DimensionMismatch(
        "effective_beam has $(size(effective_beam, 3)) pixels; sky_ring " *
        "has $pixel_count",
    ))
    size(effective_beam, 4) == _CONVOLUTION_SCAN_WEIGHTS ||
        throw(DimensionMismatch(
            "effective_beam has $(size(effective_beam, 4)) scan weights; " *
            "expected $_CONVOLUTION_SCAN_WEIGHTS",
        ))
    size(accumulated) == (
        sky_count,
        beam_count,
        pixel_count,
        _CONVOLUTION_SCAN_WEIGHTS,
    ) || throw(DimensionMismatch(
        "accumulated has size $(size(accumulated)); expected " *
        "($sky_count, $beam_count, $pixel_count, " *
        "$_CONVOLUTION_SCAN_WEIGHTS)",
    ))

    sky_count > 0 || throw(ArgumentError(
        "sky_ring must contain at least one sky",
    ))
    term_count > 0 || throw(ArgumentError(
        "the contracted (spin, M, l) axis must be non-empty",
    ))
    pixel_count > 0 || throw(ArgumentError(
        "the tile must contain at least one pixel",
    ))
    beam_count > 0 || throw(ArgumentError(
        "effective_beam must contain at least one beam",
    ))

    all(
        dimension -> dimension <= typemax(Int32),
        (sky_count, term_count, pixel_count, beam_count),
    ) || throw(ArgumentError(
        "sky, term, pixel, and beam counts must fit in Int32 for the " *
        "cuBLAS batched GEMM; got ($sky_count, $term_count, " *
        "$pixel_count, $beam_count)",
    ))

    return sky_count, term_count, pixel_count, beam_count
end

"""
    accumulate_convolution_tile_gemm_gpu!(
        accumulated, sky_ring, effective_beam,
    )

Contract one resident `l` tile over its shared `(spin, M, l)` axis and add the
result into a caller-owned accumulator:

```math
C_{i,j,p,w}\\mathrel{+}=\\sum_\\mu U_{i,\\mu,p}E_{\\mu,j,p,w}.
```

Layouts:

- `sky_ring[sky, term, pixel]` — from `build_sky_ring_gpu`
- `effective_beam[term, beam, pixel, scan_weight]` — from `apply_beams_gemm_gpu`
- `accumulated[sky, beam, pixel, scan_weight]` — allocated by the caller

`term` is the flattened `(spin, M, l)` index shared by both inputs: spin varies
fastest in `(0,+2,-2)` order, then signed `M`, then `l` within the tile.
`scan_weight` is `(s=0, s=2)`.

One non-conjugating (`'N'`,`'N'`) strided batched GEMM per scan weight is
issued, batched over pixels: both inputs are already in GEMM layout, so no
packing kernel and no temporary device array are needed. `alpha=1` and
`beta=1`, so successive `l` tiles accumulate into the same buffer, and a
non-zero `accumulated` is added to rather than overwritten.

Both weightings that the convolution needs are already present in the inputs
and **must not be reapplied here**: the polarization weight `(1,1/2,1/2)` is
applied inside `build_sky_ring_gpu`, and the complex conjugation of the beam
coefficients is applied inside `prepare_blm`.

Returns `accumulated`. The device is synchronized before returning, so the
caller may free either input immediately.
"""
function accumulate_convolution_tile_gemm_gpu!(
    accumulated::CuArray{Complex{T},4},
    sky_ring::CuArray{Complex{T},3},
    effective_beam::CuArray{Complex{T},4},
) where {T<:_HARMONIC_PRECISIONS}
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    _check_convolution_tile_inputs(
        accumulated,
        sky_ring,
        effective_beam,
    )

    alpha = one(Complex{T})
    beta = one(Complex{T})
    for scan_weight in 1:_CONVOLUTION_SCAN_WEIGHTS
        _CONVOLUTION_CUBLAS.gemm_strided_batched!(
            'N',
            'N',
            alpha,
            sky_ring,
            view(effective_beam, :, :, :, scan_weight),
            beta,
            view(accumulated, :, :, :, scan_weight),
        )
    end
    CUDA.synchronize()
    return accumulated
end
