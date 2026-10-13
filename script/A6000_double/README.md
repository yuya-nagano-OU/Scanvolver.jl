# A6000 double-precision full-sphere convolution

`run_convolution.jl` is a standalone driver for the EBCoDeCMB full-sphere
pipeline. It generates Falcons pointing, bins it once into a
`RingScanBuffer`, constructs the scan plan and CPU Wigner-Chebyshev table,
runs `convolve_sphere_multigpu`, and writes the selected output.

`A6000_double` means two NVIDIA RTX A6000 devices in double precision. The
editable defaults in the script set `DEVICES = [0, 1]` and
`PRECISION = Float64`.

## Run

Run from any directory. The script resolves the repository from its own
location and activates this directory as its Julia environment.

```bash
JULIA_NUM_THREADS=8 julia --project=script/A6000_double \
  script/A6000_double/run_convolution.jl
```

The first execution may instantiate this small script environment. For actual
two-GPU concurrency, start Julia with at least two threads. Fewer threads are
allowed and only serialize the GPU workers; the result is unchanged.

The default is a light smoke configuration: `nside=16`, `lmax=47`, one day of
Falcons pointing at 1 Hz, and the bundled CMB and axisymmetric Gaussian beam.

## Configuration

All defaults are in the single configuration block at the start of
`run_convolution.jl`. Any item can be overridden by an `EBCODE_` environment
variable with the same name. No separate configuration file is used.

| Setting | Default | Meaning |
| --- | --- | --- |
| `ALM_PATH` | bundled CMB `.npy` | Packed `(T,E,B)` sky coefficients |
| `BLM_PATH` | bundled Gaussian `.npy` | Packed `(T,E,B)` beam coefficients |
| `OUTPUT_DIR` | `script/A6000_double/output` | Destination for run products |
| `RUN_NAME` | `nside16_lmax47_1day` | Prefix for output files |
| `NSIDE` | `16` | HEALPix output resolution |
| `LSTART` | `0` | First multipole |
| `LSTOP` | `47` | Last convolved multipole |
| `LMAX` | `47` | Packed input band limit |
| `BNMAX`, `KAPPA` | `5`, `10` | Beam azimuthal and scan-moment limits |
| `SAMPLING_RATE` | `1.0` | Falcons samples per second |
| `DURATION_SECONDS` | `86400` | Falcons scan duration |
| `CHUNK_SECONDS` | `21600` | Pointing generation chunk duration |
| `PRECISION` | `Float64` | Harmonic and scan-moment precision |
| `MAP_PRECISION` | `Float64` | Saved-map precision |
| `BETA_ORDER` | `16` | Chebyshev order |
| `BETA_MARGIN` | `1.02` | `beta_max / max(beta)` |
| `RCOND` | `1e-2` | Mapmaking conditioning cutoff |
| `MOMENT_METHOD` | `gemm` | `gemm` or `direct` scan moments |
| `L_TILE` | `auto` | `auto` or a positive number of multipoles |
| `MEMORY_FRACTION` | `0.8` | Per-device memory budget fraction |
| `DEVICES` | `0,1` | Comma-separated CUDA device identifiers |
| `RECLAIM_INTERVAL` | `16` | Rings between CUDA reclamation calls |
| `CHECKPOINT_INTERVAL` | `16` | Completed rings between checkpoints |
| `RESUME` | `true` | Resume a compatible checkpoint when present |
| `OUTPUT_MODE` | `full_map` | `full_map`, `ring_callback`, or `discard` |

For example, this uses the second device only, writes elsewhere, and runs a
one-hour scan:

```bash
EBCODE_DEVICES=1 EBCODE_DURATION_SECONDS=3600 \
EBCODE_OUTPUT_DIR=/scratch/ebcode EBCODE_RUN_NAME=one_hour \
julia --project=script/A6000_double script/A6000_double/run_convolution.jl
```

Input files must be `.npy` arrays with shape
`(3, (lmax + 1) * (lmax + 2) / 2)`. The script infers the packed input lmax
from the coefficient count and stops before any GPU work if it differs from
`LMAX`; it never truncates a packed array by prefix. It also rejects requested
`EBCODE_HWP`, `EBCODE_HWP_RPM`, `EBCODE_NOISE`, and `EBCODE_NOISE_SIGMA`,
because the current pipeline does not support those modes.

## Outputs

For `OUTPUT_MODE=full_map`, the driver writes:

```text
<RUN_NAME>_maps.npy        maps[sky, beam, stokes, pixel]
<RUN_NAME>_meta.txt        settings, provenance, timing, and CUDA information
<RUN_NAME>_checkpoint.ckpt resumable full-map checkpoint
```

The checkpoint ID is derived from SHA-256 hashes of the alm file, blm file,
and pixel-ordered pointing buffer plus the calculation settings. A changed
input, pointing stream, or relevant configuration therefore refuses to resume
an incompatible checkpoint. Checkpoints are meaningful only for `full_map`.

`OUTPUT_MODE=ring_callback` writes `<RUN_NAME>_ring_i_rms.txt`, containing a
minimal per-ring I RMS summary. `OUTPUT_MODE=discard` only writes metadata.
The default `output/` directory is ignored by Git.

Before convolution, the driver logs the maximum angular pointing offset
`beta`, the corresponding `beta_max`, the pixel scale, and a representative
ring memory estimate from `ring_memory_bytes` and `suggest_l_tile`. The
Wigner table and scan moments are both built from the same `ScanMomentPlan`.
