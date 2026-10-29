"""Audit generated observations without reading any historical dataset."""

from unified_geometry.paths import ROOT
import json
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from unified_geometry.paths import DATA, BENCHMARK, GRAPH, POSES, SPLITS, write
from unified_geometry.geometry.projection import backproject


def main():
    cfg = json.loads((DATA / "data_config.json").read_text())
    K = np.array(cfg["intrinsics"])
    T = np.array(cfg["camera_to_base"])
    camera = json.loads((BENCHMARK / "camera/calibration.json").read_text())
    groups = [
        ("canonical", DATA / "data", None),
        ("shapes", BENCHMARK / "shapes/data", None),
        ("camera", BENCHMARK / "camera/data", camera),
    ]
    counts = {}
    maximum = 0.0
    for name, root, calibration in groups:
        paths = sorted(root.rglob("*.npz"))
        assert len(paths) == (
            sum(SPLITS.values()) if name == "canonical" else 512 if name == "shapes" else 768
        )
        for path in tqdm(paths, desc="Audit " + name, mininterval=5):
            transform = T if calibration is None else np.array(calibration[path.parent.name])
            with np.load(path) as d:
                uv = d["pixel_uv"]
                P = d["scene_points"]
                G = d["target_points"]
                depth = d["depth_m"]
                labels = d["instance_labels"]
                assert (
                    P.shape == (4096, 3)
                    and G.shape == (4, 128, 3)
                    and (len(np.unique(uv, axis=0)) == 4096)
                )
                assert np.isfinite(P).all() and np.isfinite(G).all()
                z = depth[uv[:, 1], uv[:, 0]].astype(float)
                xyz = (
                    np.c_[
                        (uv[:, 0] - K[0, 2]) * z / K[0, 0],
                        (uv[:, 1] - K[1, 2]) * z / K[1, 1],
                        z,
                    ]
                    @ transform[:3, :3].T
                    + transform[:3, 3]
                )
                error = float(np.abs(P - xyz).max())
                assert error < 1e-06
                full, pixels = backproject(depth, K, transform)
                ids = labels[pixels[:, 1], pixels[:, 0]]
                for task in range(4):
                    surface = full[ids == task + 1]
                    assert len(surface) > 0
                    target_error = float(cKDTree(surface).query(G[task])[0].max())
                    assert target_error < 1e-06
                    maximum = max(maximum, error, target_error)
        counts[name] = len(paths)
    for split, count in SPLITS.items():
        assert np.load(GRAPH / f"{split}.npy").shape == (count, 4096, 16)
    for split in ["val", "iid"]:
        with np.load(POSES / f"{split}.npz") as d:
            R = d["R"]
            assert R.shape == (5, 128, 3, 3) and d["t"].shape == (5, 128, 3)
            np.testing.assert_allclose(
                R @ R.swapaxes(-1, -2), np.broadcast_to(np.eye(3), R.shape), atol=1e-06
            )
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-06)
    result = dict(
        status="passed",
        scenes=counts,
        max_depth_or_GT_surface_error_m=maximum,
        checks=[
            "finite shapes",
            "unique observation UV",
            "XYZ agrees with optical depth and calibration",
            "GT belongs to visible object surface",
            "UV graph shapes",
            "SE3 proper rotations",
        ],
    )
    write(ROOT / "data_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
