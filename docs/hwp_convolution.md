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

## Validation

Run the package suite, or the focused checks:

```sh
julia --project=. --threads=2 test/test_hwp.jl
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
