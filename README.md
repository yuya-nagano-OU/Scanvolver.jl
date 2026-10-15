# Scanvolver.jl

[![CI](https://github.com/yuya-nagano-OU/Scanvolver.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/yuya-nagano-OU/Scanvolver.jl/actions/workflows/CI.yml)

Scanvolver.jl is a Julia package for GPU-accelerated, scan-aware beam
convolution in cosmic microwave background (CMB) sky simulations. It builds
reusable scan-dependent effective kernels and applies them across many sky and
beam realizations without constructing prohibitively large full-sky
intermediate tensors.

The package is under active development. The public API and numerical details
may change before the first stable release.

## Requirements

- Julia 1.12
- CUDA.jl 5 or 6
- An NVIDIA GPU with a functional CUDA setup for convolution routines

The package can be loaded and its CPU preparation/reference routines can be
tested without a GPU. Functions whose names end in `_gpu` and the sphere
convolution drivers require `CUDA.functional() == true`.

## Installation

Scanvolver.jl is not registered in Julia's General registry. Install it directly
from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/yuya-nagano-OU/Scanvolver.jl")
```

For development, clone the repository and activate it:

```julia
using Pkg
Pkg.activate("path/to/Scanvolver.jl")
Pkg.instantiate()
```

## Quick start: prepare harmonic inputs

The following CPU-only example constructs a convolution configuration and
prepares one sky and one beam from HEALPix/healpy packed non-negative-mode
coefficients. The three rows are interpreted as `(T, E, B)`.

```julia
using Scanvolver

lmax = 8
ncoeff = (lmax + 1) * (lmax + 2) ÷ 2

cp = ConvolutionParams(
    nside_output=4,
    lstart=0,
    lstop=lmax,
    lmax=lmax,
    numofsky=1,
    numofbeam=1,
    bnmax=2,
    κ=4,
)

alm_teb = zeros(ComplexF64, 3, ncoeff)
blm_teb = zeros(ComplexF64, 3, ncoeff)

prepared_alm = prepare_alm(cp, alm_teb)
prepared_blm = prepare_blm(cp, blm_teb)
```

The full GPU pipeline additionally builds a scan buffer, Wigner--Chebyshev
table, and scan-moment plan before calling `convolve_sphere_gpu` or
`convolve_sphere_multigpu`. See [`notebook/README.md`](notebook/README.md) for
the validation notebooks.

## Current scope

- Packed harmonic sky and beam preparation
- CPU and GPU scan moments
- CPU and GPU Wigner--Chebyshev tables
- GPU effective-kernel construction and beam application
- Ring and full-sphere convolution
- Multi-GPU ring scheduling
- Binned `(I, Q, U)` mapmaking
- Checkpoint and resume support

Current limitations:

- Production convolution requires CUDA; CUDA is not an optional dependency in
  this development release.
- The current GPU beam preparation and convolution path supports `HWP=false`.
- Full-map output can be very large for many skies and beams. Use
  `output=:ring_callback` for streaming workflows or `output=:discard` for
  benchmarking.
- Julia versions older than 1.12 have not yet been validated.

## Testing

Run the test suite with:

```julia
using Pkg
Pkg.test()
```

GPU tests run when `CUDA.functional()` is true and otherwise skip their GPU
bodies. The release-preparation branch was validated on Julia 1.12.6 with
CUDA.jl 6.0.0; all 4,020 tests passed with a functional CUDA device.

The GitHub Actions job uses a CPU-only runner without a functional CUDA device.
A green GitHub Actions run therefore validates the CPU paths and CUDA-less package loading,
not the GPU kernels.

## License

Scanvolver.jl is released under the MIT License. See [`LICENSE`](LICENSE).
