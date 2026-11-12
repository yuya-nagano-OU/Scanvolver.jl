# Real Stokes spin reduction

The HDF5 drivers default to `spin_reduction=:auto`. For a sky satisfying the
real-Stokes coefficient relations, they upload `(0,+2)` rather than
`(0,+2,-2)`, transform only the independent ring modes, and use real GEMM for
the final sky/beam contraction. All scan weights, HWP Fourier components and
Mueller leakage terms are retained. No physical cutoff is changed.

Use `spin_reduction=:off` to select the previous full-complex path for
comparison. `:on` requires the exact reality relations and rejects invalid
inputs. `:auto` falls back to the full path when they do not hold, preserving
support for synthetic complex inputs. The checks include `m=0` and every
signed mode; no input is projected or silently repaired.

For the low-level API, `upload_alm_gpu(alm; spin_reduction=true)` explicitly
selects reduction. `build_sky_ring_gpu` then returns an opaque reduced tile,
accepted by `select_active_sky_ring_gpu`, `free_active_sky_ring!`,
`CUDA.unsafe_free!` and `accumulate_convolution_tile_gemm_gpu!`.
Pass `spin_reduction=true` to `apply_beams_gemm_gpu` for the matching real
effective-beam layout. The low-level defaults retain the full-complex layout.

## Independent ring values

With `G=calc_Mmax(cp,cp.lstop)` and `N=2G+1`, the sky symmetries are

~~math
U_0(-M)=(-1)^M U_0(M)^*,\qquad
U_{-2}(-M)=(-1)^M U_{+2}(M)^*.
~~

The existing polarization factor 1/2 remains in `U_+2`. For each multipole:

- FFT only the scalar orders `M=0:G` and polarization orders `M=-G:G`.
  This is `3G+2` complex FFT channels instead of `3(2G+1)`.
- Store `U0(0)`, then `Re U0(M), Im U0(M)` for `M=1:G`, then
  `Re U+2(M), Im U+2(M)` for `M=-G:G`.
- The resident tile has `3N` **real** entries, half the bytes of the former
  `3N` complex entries. GPU alm storage has two spin blocks instead of three.

## Effective beam and contraction

For scalar `M>0`, the coefficients of `Re U0(M), Im U0(M)` are

~~math
E_0(M)+(-1)^M E_0(-M),\quad
i[E_0(M)-(-1)^M E_0(-M)].
~~

For each signed polarization order they are

~~math
E_{+2}(M)+(-1)^M E_{-2}(-M),\quad
i[E_{+2}(M)-(-1)^M E_{-2}(-M)].
~~

Each formula uses the **same scan weight on both E terms**. Consequently it
preserves both `<S>` and `<S exp(i chi)>`, with `chi=2psi+4eta` for HWP.
It makes no assumption that the two weighted averages are conjugates.

`apply_beams_gemm_gpu(...; spin_reduction=true)` folds its complex beam-GEMM
output into real coefficients for real/imaginary output channels:
`E_real[term, (real_beam,imag_beam), pixel, scan_weight]`. It has the same byte
size as the old effective beam. A real strided-batched GEMM and a small
packing/add kernel update the original complex accumulator, so the IQU solve
and HDF5 map schema remain the same.

The real GEMM has half the arithmetic count of the former complex GEMM.
Actual runtime gains depend on the dimensions, scan-moment work and hardware;
they must be measured rather than inferred from the arithmetic count.

## Memory and provenance

`ring_memory_bytes` and `streamed_tile_bytes` accept
`spin_reduction=true`; their low-level default is `false`. The estimators
include the transient FFT spectrum and real contraction workspace. The
automatic pipeline tile selector uses the actual selected representation.
The unit-tile bound remains conservative for wider tiles.

HDF5 attributes record `spin_reduction` (actual Boolean choice),
`spin_reduction_requested` and `sky_ring_representation`.
The stored maps, their axes, NaN/conditioning policy and physical settings do
not change. Floating-point summation order changes, so bitwise equality is
not expected; compare masks and absolute/relative numerical errors.

The existing real-Mueller, V=0 HWP scope is unchanged. Arbitrary real Mueller
values and nonzero E/B skies are supported by the reduction; all beam spin
blocks are still included when constructing the folded coefficients.
