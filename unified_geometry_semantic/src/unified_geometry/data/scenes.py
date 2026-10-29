"""Selected frozen DINOv3 point features/XYZ and original pixel graphs."""

import json
import numpy as np
import torch
from tqdm import tqdm
from unified_geometry.paths import ROOT, DATA, GRAPH, SPLITS, N, DIM
from unified_geometry.geometry.projection import backproject
from unified_geometry.geometry.transforms import apply


class Scenes:
    def __init__(self, split, analysis=False, limit=None):
        self.split = split
        self.count = SPLITS[split] if limit is None else min(limit, SPLITS[split])
        self.points = np.empty((self.count, N, 3), np.float32)
        self.targets = np.empty((self.count, 4, 128, 3), np.float32)
        self.surfaces = [] if analysis else None
        if analysis:
            config = json.loads((DATA / "data_config.json").read_text())
            K = np.array(config["intrinsics"])
            T = np.array(config["camera_to_base"])
        for scene in tqdm(range(self.count), desc=f"E178 load {split}", mininterval=10):
            with np.load(DATA / "data" / split / f"{scene:05d}.npz") as data:
                self.points[scene] = data["scene_points"]
                self.targets[scene] = data["target_points"]
                if analysis:
                    full, uv = backproject(data["depth_m"], K, T)
                    labels = data["instance_labels"][uv[:, 1], uv[:, 0]]
                    self.surfaces.append(tuple((full[labels == task + 1] for task in range(4))))
        from unified_geometry.data.feature_cache import ensure

        ensure(split)
        features = np.load(ROOT / "features" / f"{split}.npy", mmap_mode="r")
        self.features = torch.tensor(np.array(features[: self.count]), device="cuda")
        self.graph = torch.tensor(
            np.load(GRAPH / f"{split}.npy")[: self.count].astype(np.int32),
            device="cuda",
        )
        assert (
            not self.features.requires_grad
            and self.features.dtype == torch.float16
            and (self.features.shape == (self.count, N, DIM))
        )

    def batch(self, ids, transform=None):
        ids = np.asarray(ids, dtype=np.int64)
        device_ids = torch.as_tensor(ids, device="cuda")
        P = torch.tensor(self.points[ids], device="cuda")
        G = torch.tensor(self.targets[ids], device="cuda")
        if transform is not None:
            R, t = [
                torch.as_tensor(value, device="cuda", dtype=torch.float32) for value in transform
            ]
            P = apply(P, R, t)
            G = apply(G, R, t)
        return (P, self.features[device_ids], G, self.graph[device_ids].long())

    def surface(self, scene, task):
        assert self.surfaces is not None
        return self.surfaces[scene][task]
