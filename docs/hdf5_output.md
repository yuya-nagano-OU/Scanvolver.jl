# HDF5 output

`convolve_to_hdf5` convolves every sky with every beam on the rings of a scan
and writes the `(I, Q, U)` maps to one HDF5 file, ring by ring.
`convolve_detector_groups_to_hdf5` does the same with detector beams combined
into groups before the solve. Neither holds a full-sky map in memory.

## Calling

Pass a `ConvolutionParams`, a `RingScanBuffer`, and prepared skies and beams of
the same precision:

```julia
using Scanvolver

# cp, buffer, raw_alms and raw_blms come from your inputs.
alm = prepare_alm(cp, raw_alms; precision=Float32)
blm = prepare_blm(cp, raw_blms; precision=Float32)
path = convolve_to_hdf5(cp, buffer, alm, blm;
    output_path="/absolute/existing/output-directory/map.h5",
    beam_batch_size=2,
    devices=[0, 1],
    beta_order=8,
    l_tile=:auto,
    max_l_tile=32,
    sky_ids=["sky-$i" for i in 1:cp.numofsky],
    beam_ids=["beam-$i" for i in 1:cp.numofbeam],
    metadata=Dict("scan_id" => "identifier-of-input-scan"),
    progress_callback=r -> println("saved ring ", r.ring),
)
ring_map = read_ring_hdf5(path, 1)                      # (sky, beam, I/Q/U, ring_pixel)
subset = read_ring_hdf5(path, 1; skies=1:1, beams=1:2)
```

- The output directory must exist, and an existing file is never overwritten.
- `rings` defaults to every ring in the buffer; pass a vector or range of ring
  numbers to run a subset.
- `beam_metadata` takes vectors with one value per beam, for example the beam
  orientation or the input file. `metadata` takes any values HDF5 can store.
  IDs follow the order of the prepared inputs.
- The compute precision is that of the prepared inputs, and `map_precision`
  defaults to it. Other defaults: `rcond=1e-2`, `unseen=NaN`,
  `moment_method=:gemm`.
- Pixels without hits, rings that were not requested, and pixels whose scan
  does not cover enough polarization angles (see `ring_stokes`) are `unseen`.
- HWP and noise are rejected.
- With more than one device, start Julia with at least one thread per GPU.

### Runtime options

The keyword arguments to `convolve_to_hdf5` control GPU scheduling, memory
use, numerical output and bookkeeping. The defaults are intended to be a
reasonable starting point; in particular, `l_tile=:auto` normally removes the
need to tune the tile width by hand.

| Keyword | Default | Meaning |
|---|---:|---|
| `output_path` | required | New HDF5 file to create. Its parent directory must exist, and an existing path is never overwritten. |
| `beam_batch_size` | `9` | Number of beams applied together. Larger batches can improve throughput but require more temporary GPU memory. |
| `devices` | `[0]` | Zero-based CUDA device IDs. Each device needs one Julia thread. |
| `rings` | every buffered ring | Unique ring numbers to process, all inside the `RingScanBuffer`. |
| `beta_order` | `8` | Chebyshev order used for the automatic small-angle Wigner representation. |
| `l_tile` | `:auto` | Multipole tile width. `:auto` chooses the widest tile allowed by the memory budget. |
| `max_l_tile` | `32` | Upper limit on the automatically selected tile width. |
| `memory_fraction` | `0.7` | Fraction of currently available GPU memory that the tile selector may use. Must be in `(0, 1]`. |
| `moment_method` | `:gemm` | GPU scan-moment implementation. The production default is `:gemm`; `:direct` is also available. |
| `map_precision` | input precision | Element type of the output map, either `Float32` or `Float64`. Compute precision comes from the prepared inputs. |
| `rcond` | `1e-2` | Relative conditioning threshold for the per-pixel `(I,Q,U)` solve. Insufficiently constrained pixels become `unseen`. |
| `unseen` | `NaN` | Value written for unobserved or insufficiently constrained pixels. It must be finite or `NaN`, not infinity. |
| `reclaim_interval` | `16` | Number of rings processed by each GPU worker between `CUDA.synchronize()` and `CUDA.reclaim()` calls. Set to `0` to disable periodic reclamation. |
| `sky_ids`, `beam_ids` | `"1"`, `"2"`, ... | Labels stored along the corresponding HDF5 axes. Their lengths must match `numofsky` and `numofbeam`. |
| `metadata` | empty | Run-level values stored in the HDF5 `metadata/` group. |
| `beam_metadata` | empty | Per-beam vectors stored in `beam_metadata/`; every vector must have one element per beam. |
| `batch_callback` | `nothing` | Called after one beam batch has been written. The supplied CPU array is borrowed and ring-local. |
| `progress_callback` | `nothing` | Called after a ring has been completely written and marked complete. |

`reclaim_interval` controls a memory-management tradeoff, not the numerical
algorithm. CUDA.jl normally caches released allocations for reuse. Periodic
reclamation returns unused cached allocations and can keep long runs within a
tighter memory envelope, but synchronization and later reallocation add some
overhead. Lower the interval if cached GPU memory becomes a problem; increase
it, or use `0`, when allocation reuse and throughput are more important. The
counter is maintained independently by each GPU worker, so the default means
reclamation after every 16 rings handled by that device, not every 16 rings
globally.

## The β range

`beta_max` follows from the output resolution: HEALPix's centre-to-corner
radius, rounded up with a margin for the Float64 rotation arithmetic and the
conversion to the compute precision.

```text
radius   = max_pixrad(Resolution(nside_output))
margin   = 64*eps(Float64) + 8*eps(compute_precision)*radius
beta_max = nextfloat(compute_precision(radius + margin))
```

In Float32 this is about 0.004172565 rad at nside 256 and 0.002087058 rad at
nside 512. One `ScanMomentPlan` builds both the Wigner table and the scan
moments. Before the moments of a ring are built, every observed β is checked to
be finite and inside `[0, beta_max]`; a sample outside stops the run rather than
being extrapolated. The automatic range does not choose the Chebyshev order:
`beta_order` is a setting, and it is stored in the file.

## Order of work

1. Each GPU holds all skies, the Wigner table and a ringD context.
2. For each ring, the angles, the scan moments and ringD are built once, and
   the accumulators for all beams are allocated batch by batch.
3. For each `l` tile, the beam-independent kernel K and the sky ring U are built
   once; every beam batch is applied to the same K and U and accumulated.
4. After the last tile, each batch is solved for `(I, Q, U)` and written. The
   ring is marked complete once all its batches are written.

Beam batches are slices of the already prepared beams, so no beam is
conjugated twice. One cuFFT plan per shape is reused across the tiles of a ring
and destroyed when the ring ends.

The only all-beam array on the GPU is the current ring's accumulator,
`sizeof(Complex{T}) * num_sky * num_beam * active_pixels * 2` bytes; the beam
intermediates are bounded by the batch size. The host holds the inputs, the
scan buffer, the beam batches and the current ring's batch maps. Skies are not
split.

`l_tile=:auto` picks the widest tile whose K, full U and gathered U fit together
in `memory_fraction` (default 0.7) of the GPU memory still available once the
moments, ringD and accumulators are allocated. The cuFFT workspace depends on
the plan and is left as headroom. If not even one multipole fits, the run fails.

Changing `beam_batch_size` changes the output width of the contraction GEMM and
with it cuBLAS's summation order, so runs with different batch sizes agree to
rounding rather than bit for bit. The effective beams themselves are identical.

## File layout

Schema `EBCoDeCMB.beam_batched_hdf5.v1`:

| Item | Meaning |
|---|---|
| `map` | On disk `(sky, beam, 3, pixel)`: Stokes I/Q/U, HEALPix RING order |
| `sky_ids`, `beam_ids` | Input IDs along each axis |
| `beam_metadata/` | Per-beam vectors |
| `metadata/` | User-supplied run conditions |
| root attributes | Schema, precisions, nside, `l` range, bnmax, κ, β range and order, batch and tile settings, rcond, unseen |
| `rings/pixel_first`, `rings/pixel_last` | Global pixel range of each ring, **1-based and inclusive** |
| `rings/requested`, `rings/completed` | 0/1 flags for requested rings and fully written rings |
| `rings/device`, `rings/observed_beta_max` | GPU used and the largest observed β |
| `rings/elapsed_seconds` | Wall time of a ring, from its angles to its written and flushed map |
| `rings/*_builds`, `rings/tile_count`, `rings/beam_applications` | Reuse counters |
| `rings/estimated_tile_bytes`, `rings/accumulator_bytes` | Explicit GPU buffer estimates, not measured peaks |
| `run_state` | 0 running, 1 every requested ring complete, 2 failed |

HDF5.jl reverses dimensions, so reading `file["map"]` directly in Julia gives
`(pixel, stokes, beam, sky)`; `read_ring_hdf5` returns Scanvolver's
`(sky, beam, stokes, ring_pixel)`. C-order readers see
`(sky, beam, stokes, pixel)`, with pixels contiguous on disk.

`map` is chunked and written one ring at a time. All HDF5 access and the
callbacks run under one lock, and the file is closed only after every GPU worker
has joined. A ring's map is written and flushed before its `completed` flag is
written and flushed. On an error the GPU resources are released, the error
propagates, and the rings completed so far stay recorded. There is no guarantee
against a killed process or a power loss, and no automatic resume: check
`rings/completed` before reading a partial file.

`batch_callback` receives a borrowed, ring-local CPU array after it is written
and before the ring is marked complete; `progress_callback` is called after the
completion record. Exceptions from callbacks propagate.

## Detector groups

`convolve_detector_groups_to_hdf5` convolves every input detector beam,
combines consecutive detectors into groups at the sufficient-statistic level,
and solves once per group. The default order is `Qp, Qm, Up, Um`, with
orientation offsets `0, pi/2, pi/4, 3pi/4`, and each group forms

```text
d0 = (d0_Qp + d0_Qm + d0_Up + d0_Um) / 4
d2 = (d2_Qp - d2_Qm + i d2_Up - i d2_Um) / 4
```

before the solve, which follows Scanvolver's `x2 = Q + iU` convention.

The file (schema `EBCoDeCMB.detector_grouped_hdf5.v1`) has `map` axes
`(sky, detector_group, stokes, pixel)` on disk; the input-beam and group IDs and
metadata; the detector labels, offsets and weights; and the ring records above,
with `group_reductions`, `input_accumulator_bytes` and
`grouped_accumulator_bytes` in place of `accumulator_bytes`.
`read_group_ring_hdf5` returns `(sky, detector_group, stokes, ring_pixel)`.
