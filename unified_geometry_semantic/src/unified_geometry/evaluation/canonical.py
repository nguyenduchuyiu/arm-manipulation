"""Canonical fullsplits and deployedobservation-fixedUVgraph globalSE3 evaluation."""

import json
import numpy as np
import torch
from tqdm import tqdm
from unified_geometry.paths import (
    ROOT,
    POSES,
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
def predict(models, data, pose=None):
    result = {name: [] for name in models}
    for start in tqdm(
        range(0, data.count, BATCH),
        desc="DINO " + data.split + " inference",
        mininterval=10,
        leave=False,
    ):
        ids = np.arange(start, min(start + BATCH, data.count))
        transform = None if pose is None else (pose[0][ids], pose[1][ids])
        P, F, G, index = data.batch(ids, transform)
        for name, model in models.items():
            pred = model.all_tasks(P, F, index)
            assert torch.isfinite(pred).all()
            result[name].extend(pred.cpu().numpy())
    return {name: np.array(rows) for name, rows in result.items()}


def measure(data, prediction, R, t):
    rows = []
    truth = np.einsum("btkj,bij->btki", data.targets, R) + t[:, None, None]
    for scene in tqdm(
        range(data.count),
        desc="DINO " + data.split + " geometry",
        mininterval=10,
        leave=False,
    ):
        for task in range(4):
            p = prediction[scene, task]
            g = truth[scene, task]
            surface = data.surface(scene, task) @ R[scene].T + t[scene]
            row = score(p, g, surface)
            row["centroid_error_mm"] = float(np.linalg.norm(p.mean(0) - g.mean(0)) * 1000)
            row["centroid_aligned_CD_mm"] = cd(p - p.mean(0), g - g.mean(0))
            centers = truth[scene].mean(1)
            row["target_selection_percent"] = (
                float(np.linalg.norm(centers - p.mean(0), axis=-1).argmin() == task) * 100
            )
            rows.append(row)
    return (rows, truth)


def main():
    setup()
    meta = provenance()
    config = json.loads((ROOT / "train/config.json").read_text())
    assert config["status"] == "trained"
    out = ROOT / "evaluation"
    out.mkdir(exist_ok=False)
    attach_readme(out, "canonical/SE3evaluationrunning")
    models = {}
    checkpoints = {}
    for name, file in [
        ("E178_18k", "step18000.pt"),
        ("E178_best", "best.pt"),
        ("E178_last", "last.pt"),
    ]:
        path = ROOT / "train" / file
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = EquivariantGrounder().cuda().eval().requires_grad_(False)
        model.load_state_dict(saved["model"], strict=True)
        models[name] = model
        checkpoints[name] = dict(path=str(path), step=saved["step"], sha256=digest(path))
    results = {name: {} for name in models}
    equivariance = {}
    for split in [
        "train",
        "val",
        "iid",
        "positions",
        "rotations",
        "sizes",
        "combinations",
    ]:
        data = Scenes(split, analysis=True)
        output = predict(models, data)
        R = np.tile(np.eye(3, dtype=np.float32), (data.count, 1, 1))
        t = np.zeros((data.count, 3), np.float32)
        for name in models:
            rows, truth = measure(data, output[name], R, t)
            results[name][split] = summarize(rows)
            np.savez_compressed(
                out / f"{split}_{name}.npz",
                prediction=output[name],
                target=truth,
                **stack(rows),
            )
        if split == "iid":
            with np.load(POSES / "iid.npz") as labels:
                rotations = labels["R"][1:]
                translations = labels["t"][1:]
            records = {name: [] for name in models}
            clouds = {name: [] for name in models}
            drifts = {name: [] for name in models}
            for view in tqdm(range(4), desc="DINO4globalSE3/fixedUVgraph", mininterval=10):
                moved = predict(models, data, (rotations[view], translations[view]))
                for name in models:
                    rows, _ = measure(data, moved[name], rotations[view], translations[view])
                    records[name].extend(rows)
                    clouds[name].append(moved[name])
                    expected = np.einsum(
                        "btkj,bij->btki",
                        output[name].astype(float),
                        rotations[view].astype(float),
                    ) + translations[view, :, None, None].astype(float)
                    drifts[name].append(
                        np.linalg.norm(moved[name].astype(float) - expected, axis=-1) * 1000
                    )
            for name in models:
                results[name]["SE3"] = summarize(records[name])
                error = np.array(drifts[name])
                equivariance[name] = dict(
                    RMS_mm=float(np.sqrt(np.square(error).mean())),
                    mean_mm=float(error.mean()),
                    max_mm=float(error.max()),
                    p99_mm=float(np.quantile(error, 0.99)),
                    graph="OriginalUVgraphfixedwithpoint-imageassociation",
                )
                np.savez_compressed(
                    out / f"SE3_{name}.npz",
                    prediction=np.array(clouds[name]),
                    R=rotations,
                    t=translations,
                    pointwise_equivariance_error_mm=error,
                    **stack(records[name]),
                )
        write(
            out / "progress.json",
            dict(completed_split=split, results=results, equivariance=equivariance),
        )
        del data, output
        torch.cuda.empty_cache()
        if split not in ["train", "val", "iid"]:
            from unified_geometry.data.feature_cache import discard_test

            discard_test(split)
    result = dict(
        status="complete",
        checkpoints=checkpoints,
        results=results,
        equivariance=equivariance,
        training=config,
        provenance=meta,
        caveats=[
            "Noaugmentation/GTinputs;fullvisiblepointcloudsupervisiononly.",
            "SE3testsFP32inferencewithsameRGB/DINOfeatures/UVgraph;physicalcamera changesaredifferenttest.",
            "E175architecturebutbackbone/featurewidth/resolution/BF16training/LRhorizon differ;notpurecontrolledbackboneablation.",
            "One trainingseed,clutterfixed,idealdepth.",
        ],
    )
    write(out / "summary.json", result)
    attach_readme(out, "canonical/SE3complete")
    print("Canonical/SE3 evaluation complete", flush=True)


if __name__ == "__main__":
    main()
