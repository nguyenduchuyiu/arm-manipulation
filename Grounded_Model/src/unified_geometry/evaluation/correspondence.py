"""Cross-scene feature swap and within-scene permutation with best frozen DINO model."""

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm
from unified_geometry.paths import (
    ROOT,
    DATA,
    BATCH,
    setup,
    provenance,
    write,
    digest,
    attach_readme,
)
from unified_geometry.data.scenes import Scenes
from unified_geometry.models.grounder import EquivariantGrounder
from unified_geometry.metrics.reconstruction import score, stack, summarize, cd


@torch.inference_mode()
def main():
    setup()
    meta = provenance()
    out = ROOT / "test_swap"
    out.mkdir(exist_ok=False)
    attach_readme(out, "featureswaprunning")
    checkpoint = ROOT / "train/best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = EquivariantGrounder().cuda().eval().requires_grad_(False)
    model.load_state_dict(saved["model"], strict=True)
    data = Scenes("iid", analysis=True)
    uv = []
    for scene in range(128):
        with np.load(DATA / "data/iid" / f"{scene:05d}.npz") as d:
            uv.append(d["pixel_uv"])
    uv = np.array(uv)
    donor = np.arange(128) ^ 1
    indices = np.stack(
        [cKDTree(uv[donor[i]].astype(float)).query(uv[i].astype(float))[1] for i in range(128)]
    )
    rng = np.random.default_rng(17562)
    permutation = np.stack([rng.permutation(4096) for _ in range(128)])
    features = data.features.cpu().numpy()
    methods = ["baseline", "other_index", "other_uv", "own_shuffled"]
    results = {}
    for method in methods:
        pred = []
        for start in tqdm(
            range(0, 128, BATCH), desc="DINOfeaturecontrol " + method, mininterval=10
        ):
            ids = np.arange(start, start + BATCH)
            P, F, G, index = data.batch(ids)
            if method == "other_index":
                F = torch.tensor(features[donor[ids]], device="cuda")
            elif method == "other_uv":
                F = torch.tensor(features[donor[ids, None], indices[ids]], device="cuda")
            elif method == "own_shuffled":
                F = torch.tensor(features[ids[:, None], permutation[ids]], device="cuda")
            pred.extend(model.all_tasks(P, F, index).cpu().numpy())
        pred = np.array(pred)
        assert np.isfinite(pred).all()
        rows = []
        for scene in tqdm(range(128), desc="DINOfeaturecontrolmetrics " + method, mininterval=10):
            centers = data.targets[scene].mean(1)
            for task in range(4):
                p = pred[scene, task]
                g = data.targets[scene, task]
                row = score(p, g, data.surface(scene, task))
                row.update(
                    centroid_error_mm=float(np.linalg.norm(p.mean(0) - g.mean(0)) * 1000),
                    centroid_aligned_CD_mm=cd(p - p.mean(0), g - g.mean(0)),
                    target_selection_percent=float(
                        np.linalg.norm(centers - p.mean(0), axis=-1).argmin() == task
                    )
                    * 100,
                )
                rows.append(row)
        results[method] = summarize(rows)
        np.savez_compressed(
            out / f"{method}.npz",
            prediction=pred,
            target=data.targets,
            donor_scene=donor,
            **stack(rows),
        )
        if method == "baseline":
            with np.load(ROOT / "evaluation/iid_E178_best.npz") as previous:
                baseline_drift_mm = float(
                    np.linalg.norm(pred - previous["prediction"], axis=-1).max() * 1000
                )
                assert baseline_drift_mm < 0.05, baseline_drift_mm
    write(
        out / "summary.json",
        dict(
            status="complete",
            source_commit=meta["git_commit"],
            checkpoint=dict(step=saved["step"], sha256=digest(checkpoint)),
            results=results,
            baseline_max_drift_mm=baseline_drift_mm,
            configuration=dict(
                scenes=128,
                tasks=4,
                donor="Bxor1",
                permutation_seed=17562,
                GT_inputs=False,
                geometry_graph_task="unchangedrecipientB",
            ),
        ),
    )
    attach_readme(out, "featureswapcomplete")
    del model, data
    torch.cuda.empty_cache()
    from unified_geometry.data.feature_cache import discard_test

    discard_test("iid")
    print("Feature swap controls complete", flush=True)


if __name__ == "__main__":
    main()
