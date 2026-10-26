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
tested without a GPU. Functions whose names end in `_gpu` and the `*_to_hdf5`
drivers require `CUDA.functional() == true`.

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

Full-sky convolution is `convolve_to_hdf5`: it takes the configuration, a
pixel-ordered scan buffer and the prepared skies and beams, convolves every
ring on one or more GPUs, and writes the `(I, Q, U)` maps of every sky and beam
to an HDF5 file ring by ring.

## Documentation

- [HDF5 convolution and output](docs/hdf5_output.md): complete calls to
  `convolve_to_hdf5` and `convolve_detector_groups_to_hdf5`, runtime parameter
  defaults and tuning guidance, GPU scheduling, and the output-file schema.
- [Tutorial notebook](notebook/tutorial.ipynb): generate a year of Falcons.jl
  pointings in monthly chunks for a few rings, prepare harmonic inputs, run
  GPU convolution, and read and plot the HDF5 results. See the
  [notebook setup](notebook/README.md) for its environment and GPU requirements.

## Current scope

- Packed harmonic sky and beam preparation
- CPU and GPU scan moments
- Wigner--Chebyshev tables built on the CPU and uploaded to the GPU
- GPU effective-kernel construction and beam application
- Full-sky convolution to HDF5, with beams processed in batches
  (`convolve_to_hdf5`) or combined into detector groups
  (`convolve_detector_groups_to_hdf5`)
- Multi-GPU ring scheduling
- Per-pixel `(I, Q, U)` solve for each ring

Current limitations:

- Production convolution requires CUDA; CUDA is not an optional dependency in
  this development release.
- The current GPU beam preparation and convolution path supports `HWP=false`.
- The HDF5 drivers record which rings are complete but do not resume an
  interrupted run.
- Julia versions older than 1.12 have not yet been validated.

## Testing

Run the test suite with:

```julia
using Pkg
Pkg.test()
```

GPU tests run when `CUDA.functional()` is true and otherwise skip their GPU
bodies. With a functional CUDA device (Julia 1.12.6, CUDA.jl 6.3.1) all 3,843
tests pass. The two-GPU cases of the HDF5 tests run only with at least two
devices and two Julia threads, for example `Pkg.test(julia_args=["--threads=2"])`.

The GitHub Actions job uses a CPU-only runner without a functional CUDA device.
A green GitHub Actions run therefore validates the CPU paths and CUDA-less package loading,
not the GPU kernels.

## License

Scanvolver.jl is released under the MIT License. See [`LICENSE`](LICENSE).
