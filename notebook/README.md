# Notebooks

The notebook environment keeps tutorial-only dependencies out of the package
environment. From the repository root, instantiate it with:

```bash
julia --project=notebook -e 'using Pkg; Pkg.instantiate()'
```

The notebook project resolves EBCoDeCMB from the repository root and Falcons from
its public Git repository through the `[sources]` entries in
`notebook/Project.toml`.

## Full-sphere pipeline tutorial

`full_sphere_pipeline_tutorial.ipynb` runs the complete GPU pipeline from a
Falcons pointing stream to `(I, Q, U)` maps. It covers:

- pixel-ordered scan buffering;
- harmonic sky and beam preparation;
- the Wigner--Chebyshev plan and memory model;
- single- and multi-GPU sphere convolution;
- streaming output modes; and
- checkpoint and resume.

The sky and Gaussian-beam coefficients are the small validation arrays in
`test/data`, so no sibling repository or machine-specific path is required.
The one-year scan is intentionally a research-scale example and requires a
functional CUDA GPU.

The final legacy-comparison section is optional. To run it, set
`EBCODE_LEGACY_RESULT` to a compatible JLD2 result file before starting the
notebook. Without that variable the section reports that it was skipped.

For real multi-GPU parallelism, start Julia with at least as many threads as
devices, for example:

```bash
JULIA_NUM_THREADS=8 julia --project=notebook
```

## Validation notebooks

- `prepare_alm_validation.ipynb` demonstrates packed `(T,E,B)` sky input,
  spin conversion, and negative-mode reconstruction. It runs on the CPU.
- `prepare_blm_validation.ipynb` demonstrates beam layout, conjugation, and
  spin symmetry. It runs on the CPU.

Notebook outputs are cleared in the repository so that local paths, hardware
details, and stale numerical results are not published as source artifacts.
