# Notebooks

The notebook environment keeps notebook-only dependencies out of the package
environment. From the repository root, instantiate it with:

```bash
julia --project=notebook -e 'using Pkg; Pkg.instantiate()'
```

The notebook project resolves Scanvolver from the repository root through the
`[sources]` entry in `notebook/Project.toml`.

## Validation notebooks

- `prepare_alm_validation.ipynb` demonstrates packed `(T,E,B)` sky input,
  spin conversion, and negative-mode reconstruction. It runs on the CPU.
- `prepare_blm_validation.ipynb` demonstrates beam layout, conjugation, and
  spin symmetry. It runs on the CPU.

Notebook outputs are cleared in the repository so that local paths, hardware
details, and stale numerical results are not published as source artifacts.
