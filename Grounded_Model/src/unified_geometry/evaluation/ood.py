"""Frozen best checkpoint on identical physical shape/camera observations used by E175."""

import json
import numpy as np
import torch
from tqdm import tqdm
from unified_geometry.paths import (
    ROOT,
    DATA,
    BENCHMARK,
    setup,
    provenance,
    write,
    digest,
    attach_readme,
)
from unified_geometry.data.vision import Features, WEIGHTS
from unified_geometry.models.grounder import EquivariantGrounder
from unified_geometry.data.graph import graph
from unified_geometry.geometry.projection import backproject
from unified_geometry.metrics.reconstruction import score, stack, summarize, cd


def groups():
    config = json.loads((DATA / "data_config.json").read_text())
    T = np.array(config["camera_to_base"])
    result = []
    for name in ["sphere", "ellipsoid", "cylinder", "capsule"]:
        result.append(
            dict(domain="shapes", name=name, folder=BENCHMARK / "shapes/data" / name, T=T)
        )
    camera = dict(calibration=json.loads((BENCHMARK / "camera/calibration.json").read_text()))
    for name, T in camera["calibration"].items():
        result.append(
            dict(domain="camera", name=name, folder=BENCHMARK / "camera/data" / name, T=np.array(T))
        )
    return result


def load(path):
    with np.load(path) as sample:
        return {key: sample[key] for key in sample.files}


def measure(sample, prediction, K, T, scene):
    truth = sample["target_points"]
    assert np.isfinite(truth).all(), "ExistingOODprotocol expects allfourtargetsobservable"
    full, uv = backproject(sample["depth_m"], K, T)
    labels = sample["instance_labels"][uv[:, 1], uv[:, 0]]
    rows = []
    for task in range(4):
        p = prediction[task]
        g = truth[task]
        r = score(p, g, full[labels == task + 1])
        centers = truth.mean(1)
        r.update(
            centroid_error_mm=float(np.linalg.norm(p.mean(0) - g.mean(0)) * 1000),
            centroid_aligned_CD_mm=cd(p - p.mean(0), g - g.mean(0)),
            target_selection_percent=float(
                np.linalg.norm(centers - p.mean(0), axis=-1).argmin() == task
            )
            * 100,
            input_points=int(
                (
                    sample["instance_labels"][sample["pixel_uv"][:, 1], sample["pixel_uv"][:, 0]]
                    == task + 1
                ).sum()
            ),
            scene=scene,
            task=task,
        )
        rows.append(r)
    return rows


@torch.inference_mode()
def main():
    setup()
    meta = provenance()
    out = ROOT / "ood"
    out.mkdir(exist_ok=False)
    attach_readme(out, "OODrunning")
    checkpoint = ROOT / "train/best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = EquivariantGrounder().cuda().eval().requires_grad_(False)
    model.load_state_dict(saved["model"], strict=True)
    cfg = json.loads((DATA / "data_config.json").read_text())
    K = np.array(cfg["intrinsics"])
    T0 = np.array(cfg["camera_to_base"])
    fcfg = json.loads((ROOT / "features/config.json").read_text())
    weight_hash = digest(WEIGHTS / "model.safetensors")
    assert weight_hash == fcfg["weights_sha256"]
    extractor = Features(
        K, T0, json.loads((ROOT / "feature_selection.json").read_text())["selected"]
    )
    results = {"shapes": {}, "camera": {}}
    subsets = {"shapes": {}, "camera": {}}
    metadata = []
    for domain in results:
        path = out / domain
        path.mkdir()
        attach_readme(path, domain + "running")
    for group in groups():
        domain, name = (group["domain"], group["name"])
        extractor.T = group["T"]
        predictions = []
        truths = []
        for start in tqdm(
            range(0, 128, 4), desc="E178OOD DINOv3 " + domain + "/" + name, mininterval=10
        ):
            samples = [load(group["folder"] / f"{i:05d}.npz") for i in range(start, start + 4)]
            F = extractor(samples)
            P = torch.tensor(np.stack([d["scene_points"] for d in samples]), device="cuda")
            index = torch.tensor(
                np.stack([graph(d["pixel_uv"]) for d in samples]), device="cuda", dtype=torch.long
            )
            prediction = model.all_tasks(P, F, index).cpu().numpy()
            assert np.isfinite(prediction).all()
            predictions.extend(prediction)
            truths.extend([d["target_points"] for d in samples])
            del F, P, index, prediction
        predictions = np.array(predictions)
        truths = np.array(truths)
        rows = []
        for scene in tqdm(
            range(128), desc="E178OOD metrics " + domain + "/" + name, mininterval=10
        ):
            sample = load(group["folder"] / f"{scene:05d}.npz")
            assert np.array_equal(sample["target_points"], truths[scene])
            rows.extend(measure(sample, predictions[scene], K, group["T"], scene))
        summary = summarize(rows)
        summary["zero_input_target_queries"] = sum((r["input_points"] == 0 for r in rows))
        summary["median_input_points"] = float(np.median([r["input_points"] for r in rows]))
        results[domain][name] = summary
        subsets[domain][name] = {
            str(n): summarize([r for r in rows if r["input_points"] >= n])
            if any((r["input_points"] >= n for r in rows))
            else None
            for n in [1, 8]
        }
        path = out / domain
        np.savez_compressed(
            path / f"{name}.npz", prediction=predictions, target=truths, **stack(rows)
        )
        write(path / f"{name}_records.json", rows)
        metadata.append(
            dict(
                domain=domain,
                name=name,
                data_folder=str(group["folder"]),
                camera_to_base=group["T"].tolist(),
            )
        )
        write(out / "progress.json", dict(last_group=domain + "/" + name, results=results))
    assert (
        extractor.max_projection_error < 0.001
        and digest(WEIGHTS / "model.safetensors") == weight_hash
    )
    result = dict(
        status="complete",
        source_commit=meta["git_commit"],
        checkpoint=dict(path=str(checkpoint), step=saved["step"], sha256=digest(checkpoint)),
        results=results,
        density_subsets=subsets,
        inputs=metadata,
        DINOv3weights_sha256=weight_hash,
        projection_roundtrip_max_error_pixels=extractor.max_projection_error,
        source_data_manifests=dict(
            shapes=digest(BENCHMARK / "shapes/render_manifest.json"),
            camera=digest(BENCHMARK / "camera/render_manifest.json"),
        ),
        caveats=[
            "Samephysicalshape/cameraobservationsasE175,all512queriespergroup;bestcheckpointselectedonlycanonicalvalidation.",
            "NewUVgraphcomputedforeachobservation;DINOv3fresh,notstale.",
            "Densitysubsetsarediagnostic,allqueriesprimary;unequalshapevolumes/visibility.",
            "No GT/modeladaptation;singletrainseed/fixedclutter/idealdepth.",
        ],
    )
    write(out / "summary.json", result)
    attach_readme(out, "OODcomplete")
    for domain in results:
        attach_readme(out / domain, domain + "complete")
    del model, extractor
    torch.cuda.empty_cache()
    print("Shape/camera OOD evaluation complete", flush=True)


if __name__ == "__main__":
    main()
