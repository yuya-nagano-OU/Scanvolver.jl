# Detector-grouped HDF5 pipeline

This pipeline is separate from `beam_batched_hdf5`. It convolves every input
detector beam, combines detector sufficient statistics, and performs one
mapmaking solve per physical detector group.

The default consecutive input order is `Qp, Qm, Up, Um`, with detector
orientation offsets `0, pi/2, pi/4, 3pi/4`. For each group it forms

```text
d0 = (d0_Qp + d0_Qm + d0_Up + d0_Um) / 4
d2 = (d2_Qp - d2_Qm + i d2_Up - i d2_Um) / 4
```

before mapmaking. The output follows Scanvolver's `x2 = Q + iU` convention.

Use `convolve_detector_groups_to_hdf5`. The HDF5 map has Julia axes
`(pixel, stokes, detector_group, sky)` and h5py axes
`(sky, detector_group, stokes, pixel)`.

Run the focused tests with:

```bash
julia --project=pipelines/detector_grouped_hdf5 \
  pipelines/detector_grouped_hdf5/test/test_detector_grouped_pipeline.jl
```
