# Tutorial

[tutorial.ipynb](tutorial.ipynb) walks through a one-year, single-detector scan
generated with Falcons.jl in twelve monthly chunks, retaining a few HEALPix
rings, preparing toy sky/beam harmonics, running GPU convolution, and reading
and plotting the HDF5 maps. The defaults use nside 8, lmax 16, 1 Hz sampling,
and three equatorial rings. A month is one twelfth of a 365.25-day year in this
example, not a calendar month.

## Setup

Use Julia 1.12. The notebook environment keeps Falcons, plotting and Jupyter
dependencies out of the package environment. From the repository root:

```bash
julia --project=notebook -e 'using Pkg; Pkg.instantiate()'
```

The notebook project resolves Scanvolver from the repository root through the
`[sources]` entry in `notebook/Project.toml`.

To register a kernel for this environment, run from the repository root:

```bash
julia --project=notebook -e 'using IJulia; IJulia.installkernel("Scanvolver tutorial", "--project=$(abspath("notebook"))", "--threads=auto")'
```

Open `tutorial.ipynb` in Jupyter or an editor with Julia notebook support, select
the registered kernel, and run the cells in order. The kernel working directory
must be either the repository root or `notebook/`. The setup cell activates the
notebook environment explicitly.

## CPU and GPU execution

Pointing generation, harmonic preparation, scan-buffer construction, and hit
plots run on the CPU. The generator produces one month only when the buffer
builder asks for it; it does not materialize a year of input chunks. Retained
pointings remain in the buffer, and construction also needs working memory.

Convolution requires an NVIDIA GPU for which `CUDA.functional()` is true in
the Julia kernel. On a CPU-only machine, the convolution and dependent HDF5/map
plot cells explicitly skip. One GPU is selected by default. A multi-GPU kernel
needs at least one Julia thread per selected GPU.

Each convolution run writes to a fresh temporary directory. Julia removes that
directory on exit; choose a new path in an existing persistent directory if you
want to keep the output. Existing files are never overwritten.

The CPU workflow was smoke-tested with two one-hour chunks, including a
comparison with single-call ingestion and a separate month-boundary check.
The default full year was not executed. The notebook was authored without a
GPU; GPU convolution and its numerical output have not been executed or
validated as part of this tutorial change.
Its toy harmonics and low sampling rate demonstrate usage, rather than a
scientifically validated simulation. See the [HDF5 reference](../docs/hdf5_output.md)
for runtime settings and output details.

Notebook outputs are cleared in the repository so that local paths, hardware
details, and stale numerical results are not published as source artifacts.
