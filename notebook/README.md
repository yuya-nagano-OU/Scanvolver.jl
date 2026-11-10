# Tutorial

[tutorial.ipynb](tutorial.ipynb) walks through a one-year, single-detector scan
generated with Falcons.jl in twelve consecutive observation chunks, retaining
a few HEALPix rings, preparing toy sky/beam harmonics, running GPU convolution,
and reading and plotting the HDF5 maps. The defaults use nside 8, lmax 16, 1 Hz sampling,
and three equatorial rings. A month is one twelfth of a 365.25-day year in this
example, not a calendar month.

## Setup

Use Julia 1.12. The notebook environment keeps Falcons, plotting and Jupyter
dependencies out of the package environment. From the repository root:

```bash
julia --project=notebook -e 'using Pkg; Pkg.instantiate()'
```

The `[sources]` entries in `notebook/Project.toml` resolve Scanvolver from the
repository root and Falcons from commit
`3a1fb5b6933b230423b4f856c40fa7a54b0f8dbc` (master on 2026-11-10). That
Falcons revision uses the new `Observation` API and reports version `1.0.2`.
The commit is pinned because older Falcons code can report the same version.
The notebook environment uses CUDA 5, which this Falcons revision supports.

If this environment already has a Manifest from the old tutorial, run
`julia --project=notebook -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'`.
Restart the notebook kernel after changing dependencies; an already loaded
Falcons module cannot be replaced by activating the new environment.

To register a kernel for this environment, run from the repository root:

```bash
julia --project=notebook -e 'using IJulia; IJulia.installkernel("Scanvolver tutorial", "--project=$(abspath("notebook"))", "--threads=auto")'
```

Open `tutorial.ipynb` in Jupyter or an editor with Julia notebook support, select
the registered kernel, and run the cells in order. The kernel working directory
must be either the repository root or `notebook/`. The setup cell activates the
notebook environment explicitly.

## CPU and GPU execution

Pointing generation defaults to `Falcons.CPUBackend()`. To use CUDA, set
`pointing_backend = Falcons.GPUBackend(gpu_ids=device_ids)` in the configuration
cell. Both backends return host pointing arrays for Scanvolver's scan buffer.
Harmonic preparation, scan-buffer construction, and hit plots run on the CPU.
The generator produces one month only when the buffer builder asks for it;
it does not materialize a year of input chunks. Retained
pointings remain in the buffer, and construction also needs working memory.

The scan is represented by `SpinningScanningStrategy`, `InstrumentInfo`,
`DetectorInfo`, and `Observation`. `split_observation` partitions its sample
grid, preserving absolute times across boundaries. Pointings explicitly use
`Falcons.Ecliptic`, matching the input sky. The previous 45/50 degree geometry,
192.348 minute precession, 0.05 rpm spin, and HWP-free setup are retained.

Convolution requires an NVIDIA GPU for which `CUDA.functional()` is true in
the Julia kernel. On a CPU-only machine, the convolution and dependent HDF5/map
plot cells explicitly skip. One GPU is selected by default. A multi-GPU kernel
needs at least one Julia thread per selected GPU.

Each convolution run writes to a fresh temporary directory. Julia removes that
directory on exit; choose a new path in an existing persistent directory if you
want to keep the output. Existing files are never overwritten.

An earlier revision (`08a861f`, before the Falcons migration) completed the
default full-year workflow on an RTX A6000 with Julia 1.12.6, including GPU
convolution, HDF5 readback, and plotting. That run does not validate this
Falcons migration. The migrated notebook's code cells were run on Julia 1.12.5
with two one-hour CPU chunks at 1 Hz (7,200 samples, 1,239 retained hits).
All 25 integration checks passed, including equality with single-call buffer
ingestion, a non-integer sample-rate split, and the full-year month boundaries.
A second smoke run on an RTX A6000 with Julia 1.12.6 and CUDA.jl 5.11.3
passed all 42 integration checks, including GPU convolution, HDF5 readback,
and plotting with both CPU and GPU Falcons pointings. The maximum CPU/GPU
pointing difference was 5.5e-15 rad, and all compared finite map values were
identical. The short scan has limited angle coverage: only 6, 3, and 6 of the
96 Stokes values in rings 15, 16, and 17 were finite; the remaining values
were NaN for missing or insufficiently conditioned coverage, as expected.
The default full year has not been rerun for this migration.
Its toy harmonics and low sampling rate demonstrate usage, rather than a
scientifically validated simulation. See the [HDF5 reference](../docs/hdf5_output.md)
for runtime settings and output details.

Notebook outputs are cleared in the repository so that local paths, hardware
details, and stale numerical results are not published as source artifacts.
