"""Verify native HDF5 axes, ring indices, and numbers against Julia's .npy oracle."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

directory = Path(sys.argv[1]).resolve()
results = []
for path in sorted(directory.glob("Float*_b*_g*.h5")):
    precision = path.name.split("_", 1)[0]
    reference = np.load(directory / f"baseline_{precision}.npy")
    with h5py.File(path, "r") as f:
        actual = f["map"][:]
        assert actual.shape == reference.shape
        assert actual.dtype == reference.dtype
        # Fixed-length HDF5 strings are exposed as bytes by h5py.
        assert f.attrs["disk_axes"].decode("utf-8") == "sky,beam,stokes,pixel"
        assert f.attrs["pixel_ordering"].decode("utf-8") == "RING"
        assert f.attrs["pixel_index_base"] == 1
        assert f["run_state"][:].tolist() == [1]
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(reference))
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(reference))
        first = f["rings/pixel_first"][:]
        last = f["rings/pixel_last"][:]
        assert first[0] == 1 and last[-1] == actual.shape[-1]
        np.testing.assert_array_equal(first[1:], last[:-1] + 1)
        for ring in np.flatnonzero(f["rings/completed"][:]):
            # Stored pixel IDs are one-based, while h5py slicing is zero-based.
            np.testing.assert_array_equal(
                np.isnan(f["map"][..., first[ring]-1:last[ring]]),
                np.isnan(reference[..., first[ring]-1:last[ring]]),
            )
        errors = []
        for s, name in enumerate(("I", "Q", "U")):
            valid = np.isfinite(reference[:, :, s, :])
            ref = reference[:, :, s, :][valid].astype(np.float64)
            got = actual[:, :, s, :][valid].astype(np.float64)
            assert ref.size > 0
            atol, rtol = (2e-6, 2e-5) if precision == "Float32" else (1e-12, 1e-11)
            if f.attrs["nside_output"] == 256:
                atol *= np.max(np.abs(ref))
            delta = np.abs(got - ref)
            ratio = np.max(delta / (atol + rtol * np.abs(ref)))
            assert ratio <= 1
            errors.append({"stokes": name, "max_absolute_error": float(delta.max()),
                           "relative_l2_error": float(np.linalg.norm(delta)/np.linalg.norm(ref)),
                           "max_tolerance_ratio": float(ratio)})
        results.append({"file": path.name, "shape": actual.shape, "dtype": str(actual.dtype),
                        "status": "passed", "errors": errors})
        print(f"PASS {path.name}: axes=(sky,beam,stokes,pixel), shape={actual.shape}")
assert results, "No comparison HDF5 outputs found"
(directory / "h5py_verification.json").write_text(json.dumps(results, indent=2) + "\n")
