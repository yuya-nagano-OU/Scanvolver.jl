# Rotating HWP / Mueller convolution

The HWP path uses the LBS crossing-angle convention. For an ideal HWP and
zero detector polarization offset its nominal model is

```
d = I + Q cos(2ψ + 4η) + U sin(2ψ + 4η).
```

Here `η` is the mechanical HWP angle in radians in the same focal-plane frame
as the input detector beams. It is distinct from the local Euler angle
`Angles.α`. With detector polarization offset `δ` in that frame, the nominal
phase is `2ψ + 4η - 2δ`. This sign reversal relative to the HWP-free detector
offset is also applied by the detector-group driver.

## Example

```julia
using Scanvolver

cp = ConvolutionParams(
    nside_output=64, lmax=191, lstop=191, bnmax=10, κ=20,
    numofsky=1, numofbeam=1, HWP=true, HWP_rpm=46.0,
)

# phi, theta, psi, eta: per-sample radians. alm_teb and blm_teb use the
# full HEALPix m-major packed layout, with shape (3, (lmax+1)*(lmax+2)÷2).
buffer = build_ring_scan_buffer(cp, phi, theta, psi; hwp_angle=eta)
sky = prepare_alm(cp, alm_teb; precision=Float64)
M = [1.0 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
beam = prepare_hwp_blm(cp, blm_teb; mueller=M, precision=Float64)

convolve_to_hdf5(cp, buffer, sky, beam; output_path="hwp_maps.h5")
```

The default Mueller matrix is the ideal HWP. The matrix is a finite real
4×4 array in IQUV order. This first implementation accepts **TEB sky and beam
inputs, with V=0 on both sides**. It simulates non-ideal transmission and
I/Q/U leakage, and bins into the nominal ideal-HWP IQU basis; it does not
invert the non-ideal Mueller matrix in the map-maker. Nonzero V sky or beam
inputs are not supported.

`HWP_rpm` is required by the configuration even when measured `hwp_angle`
samples are supplied; it is not used to replace those samples. Set it to zero
when only explicit phases are relevant. Alternatively supply absolute times:

```julia
buffer = build_ring_scan_buffer(cp, phi, theta, psi;
    time_array=seconds_since_reference_epoch, hwp_phase_rad=0.2)
```

The generated phase is `mod(2π * HWP_rpm / 60 * time + hwp_phase_rad, 2π)`.
For a chunked scan use named tuples `(phi=..., theta=..., psi=...,
hwp_angle=...)` or `(phi=..., theta=..., psi=..., time_array=...)`. A plain
four-tuple is interpreted as `(phi, theta, psi, time_array)`. Times must share
one reference epoch across all chunks. The buffer stores Float64 phases and
permutes them together with the pointing samples.

Old HWP-free caches contain neither HWP phases nor original sample times;
their pixel-sorted array index cannot reconstruct the missing time stream.
Build a phase-bearing buffer from the original samples for HWP runs. The old
11-argument `RingScanBuffer` constructor is retained for importing HWP-free
caches. Newly serialized buffers have an additional field.

## Fourier beams and scan moments

The implementation follows the shifted-beam, narrow-beam approximation with
fundamental spatial Mueller modes, as used in the project's LBS/SBM
derivation. HWP time harmonics alone do not justify a spatial beam-index
shift for arbitrary spatial Mueller fields.

With normalized complex Stokes coefficients and `C=T*M*T†`, an element
`C[X,Y]` carries harmonic `t=sigma[X]-sigma[Y]`, with `sigma=(0,2,-2,0)`.
The conjugated effective coefficient at Wigner order `n` uses physical beam
order `n-t`. The implementation handles the normalization between the
normalized basis and the existing unnormalized spin blocks explicitly.

Signed harmonics are combined analytically into the five real components
`1, cos(2η), sin(2η), cos(4η), sin(4η)`. Each component is a real Stokes beam,
so it satisfies the existing negative-mode reality relation. Only exactly
zero components are omitted; small physical 2f terms are never thresholded.

`bnmax` remains the physical input cutoff. Effective beams and Wigner tables
extend through `min(bnmax+4,lmax)`, and the retained sky-ring order extends
through `min(κ+effective_bnmax,l)`. Low-level geometric builders continue to
take an HWP-free geometry configuration internally: the HWP physics is in
the effective beams and phase-weighted scan moments, not in Wigner d itself.

The two moments accumulated for map-making are `<d>` and
`<d exp(i(2ψ+4η))>`. Consequently, a 4f beam component can require an 8f phase
inside the scan average. CPU, direct-GPU and GEMM constructors accept
`hwp_mode=:constant/:cos2/:sin2/:cos4/:sin4` and `demodulate_hwp=true`.

One compact kernel is streamed per Fourier component and reused across beam
batches. Each sky tile and ringD is built once. Component contributions are
summed into the existing two accumulators; no extra HWP axis is retained in
the output maps. All active component moments remain resident, and tile
sizing uses the remaining device memory after their allocation. The GEMM
moment memory estimator includes the additional modulation array when used.

Both `convolve_to_hdf5` and `convolve_detector_groups_to_hdf5` accept
`PreparedHWPBlm`. HDF5 files record the matrix, Fourier modes, effective cutoff,
angle frame and demodulation convention. The existing map dataset axes are
unchanged. HDF5.jl reverses dataset dimensions on disk; the Mueller dataset is
`(output,input)` when read in Julia and `(input,output)` in Python. The file
records both axis orders explicitly.

## Optional low-ell intensity response

The default `lowell_hwp=:shifted` keeps the original approximation. In that
model, the I-input ell=0,1 2f terms proportional to `M_QI` and `M_UI` are
absent: the same-ell polarized beam coefficients do not exist. This is a
limitation of the shifted-beam model, not a statement that physical intensity
monopoles and dipoles cannot generate a 2f signal. LBS uses the same shifted
model and therefore cannot independently validate these terms.

To retain the beam multipole sum for just these terms, use:

```julia
# A passive HWP: principal-axis intensity transmissions 0.99 and 0.97,
# with pi relative retardance. Matrix rows are outputs and columns inputs.
g = sqrt(0.99 * 0.97)
M = [0.98 0.01 0 0; 0.01 0.98 0 0; 0 0 -g 0; 0 0 0 -g]
beam = prepare_hwp_blm(cp, blm_teb; mueller=M, lowell_hwp=:full_sum)
```

The spatial assumption is a Mueller matrix uniform in the beam-centred
Ludwig-3 frame. Multiplying a spin-2 beam by the Mueller spatial mode
produces a scalar effective beam with ell=0,1. No spin-2 coefficient below
ell=2 is introduced, and the original spin-2 guards remain in place.

For output scalar multipole `ell`, Wigner order `n`, HWP harmonic `t=+/-2`,
and polarized input beam order `k=n-t`, the scalar projection weight is

```math
W_{\ell n t,\ell_1} = \sum_{\ell_2}
(-1)^{n+\ell_2}\frac{2(2\ell_2+1)}{\ell_2(\ell_2+1)}
\sqrt{(2\ell_1+1)(2\ell+1)}
\begin{pmatrix}\ell_1&\ell_2&\ell\\-t&t&0\end{pmatrix}
\begin{pmatrix}\ell_1&\ell_2&\ell\\n-t&t&-n\end{pmatrix}.
```

The sum uses polarized input beam multipoles `ell1=2:cp.lmax`, with the
physical `bnmax` cutoff. There is no extra fixed multipole cutoff. In
particular, `lstop` limits the sky/output multipoles, not this input beam
sum; a low-ell-only run may set `lstop=1` while retaining the full beam and
its `lmax`-sized input arrays.

The triangle rule leaves one `ell2` for ell=0, and at most three for ell=1.
`WignerFamilies.jl` evaluates these short 3j families. Weights are computed
once per beam-preparation call and reused across detector beams, in O(lmax)
work per output coefficient. Summation uses Float64 before casting to the
requested storage precision. No additional time-domain or GPU operation
is needed; the result uses the existing cos2/sin2 Fourier beams. HDF5 files
record the choice in the `hwp_lowell` attribute, and beam batches retain it.

Only the **I-input ell=0,1 2f terms** are extended. Polarized sky has no
ell<2 coefficients; all other terms, including ell>=2 and shifts with
`abs(n)>ell`, retain the previous treatment. With no `M_QI/M_UI` coupling
the prepared coefficients are unchanged. With no intensity sky monopole or
dipole the additional coefficients do not contribute to the signal.

Nominal I/4f mapmaking can suppress 2f when the HWP-angle coverage is good.
The residual depends on the scan, input sky, and Mueller matrix; neither
the option nor a small toy-beam test establishes its production-map size.

## Validation

Run the package suite, or the focused checks:

```sh
julia --project=. --threads=2 test/test_hwp.jl
julia --project=. --threads=2 test/test_hwp_lowell.jl
```

Checks cover phase transport across chunks and pixel sorting, harmonic
selection without thresholding, beam reality, nominal IQU recovery, detector
offsets, CPU/direct-GPU/GEMM moment agreement, and continuous subpixel
convolution against independent finite Wigner sums for identity, ideal and
general Mueller matrices. GPU checks use explicit skipped tests when CUDA is
unavailable. Optional LBS comparison fixtures are written outside the package
when `SCANVOLVER_HWP_FIXTURES` is set.

The HWP-free regression suite remains required. Production-size throughput,
κ/Chebyshev convergence for a particular experiment, and V coupling need
their own validation; low-l tests do not establish those results.
