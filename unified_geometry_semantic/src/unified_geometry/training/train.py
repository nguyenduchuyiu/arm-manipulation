"""Scratch DINO+E175decoder30k, EMDonly, originalcanonical observations."""

import json
import math
import time
import numpy as np
import torch
from tqdm import tqdm
from unified_geometry.paths import (
    ROOT,
    DATA,
    GRAPH,
    POSES,
    BATCH,
    STEPS,
    setup,
    provenance,
    write,
    save,
    digest,
    attach_readme,
)
from unified_geometry.data.scenes import Scenes
from unified_geometry.models.grounder import EquivariantGrounder
from unified_geometry.losses.emd import emd
from unified_geometry.geometry.shape import geometry
from unified_geometry.geometry.transforms import sample_se3, apply


@torch.inference_mode()
def validation(model, data, transform=None):
    model.eval()
    errors = []
    centers = []
    radii = []
    for start in tqdm(
        range(0, data.count, BATCH),
        desc="DINO validation " + data.split,
        mininterval=10,
        leave=False,
    ):
        ids = np.arange(start, min(start + BATCH, data.count))
        pose = None if transform is None else (transform[0][ids], transform[1][ids])
        P, F, G, index = data.batch(ids, pose)
        prediction = model.all_tasks(P, F, index).flatten(0, 1)
        truth = G.flatten(0, 1)
        errors.extend((emd(prediction, truth) * 1000).cpu().tolist())
        pc, pr = geometry(prediction)
        gc, gr = geometry(truth)
        centers.extend(((pc - gc).norm(dim=-1) * 1000).cpu().tolist())
        radii.extend(((pr - gr).abs() * 1000).cpu().tolist())
    return dict(
        EMD_mm=float(np.mean(errors)),
        centroid_error_mm=float(np.mean(centers)),
        Rg_absolute_error_mm=float(np.mean(radii)),
    )


@torch.inference_mode()
def equivariance_monitor(model, data):
    model.eval()
    P, F, G, index = data.batch(np.arange(2))
    generator = torch.Generator().manual_seed(17599)
    R, t = sample_se3(2, generator, device="cpu")
    R = R.cuda()
    t = t.cuda()
    base = model.all_tasks(P, F, index)
    moved = model.all_tasks(apply(P, R, t), F, index)
    error = (moved - apply(base, R, t)).norm(dim=-1) * 1000
    return dict(RMS_mm=float(error.square().mean().sqrt()), max_mm=float(error.max()))


def main(resume=False):
    setup()
    meta = provenance()
    assert json.loads((DATA / "data_config.json").read_text())["status"] == "data_complete"
    out = ROOT / "train"
    if not resume:
        out.mkdir(exist_ok=False)
    train, val = (Scenes("train"), Scenes("val"))
    model = EquivariantGrounder().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    rng = np.random.default_rng(170)
    with np.load(POSES / "val.npz") as pose:
        val_transform = (pose["R"][1], pose["t"][1])
    if resume:
        cfg = json.loads((out / "config.json").read_text())
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        assert (
            cfg["status"] == "training"
            and cfg["provenance"]["git_commit"] == meta["git_commit"]
            and (cfg["steps"] == STEPS)
        )
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["data_rng"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first = saved["step"] + 1
        best = saved["best"]
        best_step = saved["best_step"]
        previous_elapsed = saved["elapsed_s"]
    else:
        cfg = dict(
            status="training",
            steps=STEPS,
            parameters=sum((p.numel() for p in model.parameters())),
            encoder_parameters=sum(
                (
                    p.numel()
                    for name, p in model.named_parameters()
                    if name.startswith(("feature_norm", "point_input", "vector_input", "encoder"))
                )
            ),
            initialization="Alltrainablemodulesfromscratch,noE175trainedweights;onlypretrainedDINOv3ViTLfrozen",
            architecture="E1752localscalar-vectorblocks128/8vectors/pixelUV16;2layer128query4headGroundedDecoder+equivariantcoordinatehead;onlyvisualinput256→1024",
            input="Full4096XYZ+matchingDINOv3features1024+E175UVgraph+taskID;noGTmask/center/size/pose/objecttokens",
            equivariance_contract="Globalcoordinateframechange,RGB/DINOpointassociationandUVgraphfixed;hscalarinvariant,Vrotates;primaryevaluationFP32",
            graph="OriginalE17516nearestUVneighbors,nocrop/GT",
            feature_selection=json.loads((ROOT / "feature_selection.json").read_text()),
            frozen_features_root=str(ROOT / "features"),
            vision_features_config_sha256=digest(ROOT / "features/config.json"),
            graph_manifest_sha256=digest(GRAPH / "manifest.json"),
            train_scenes=1024,
            train_task_examples=4096,
            batch_scenes=BATCH,
            batch_tasks=BATCH * 4,
            original_points=4096,
            output_points=128,
            augmentation="None;originalcanonicalE157observationsonly,noSE3orXYZscaleaugmentation;physicalcuboidresizealreadyrenderedinE157",
            data="E157originalfixedcamera,robot/chair/rug/table/floor;128GTvisiblepoints",
            loss="OnlyexactunsquaredHungarianEMD128points,inmetres",
            precision="BF16autocastscalarMLPs/transformers,FP32vectorgeometry/XYZreadout/HungarianEMD/masterweights/optimizer;FP32eval,TF32off;frozenDINOcacheFP16",
            lr=0.0003,
            min_lr=3e-05,
            warmup=100,
            weight_decay=0.01,
            gradient_clip=1.0,
            model_seed=170,
            data_seed=170,
            selection="MinimumcanonicalvalEMDevery1000;step0eligible;SE3diagnosticonlynotselection;save18kmilestone,best,last,resume",
            comparison="Controlled E177 block6 vs E178 block24: same backbone weights/normalization/featurewidth/resize/architecture/data/graph/seeds/BF16/loss/batch/LR/schedule/30kbudget; only selected hidden layer changes. E175 SAM3 comparison has other differences.",
            provenance=meta,
        )
        write(out / "config.json", cfg)
        initial = validation(model, val)
        best = initial["EMD_mm"]
        best_step = 0
        first = 1
        previous_elapsed = 0.0
        save(
            out / "best.pt",
            dict(model=model.state_dict(), step=0, val_EMD_mm=best, config=cfg),
        )
        write(out / "initial_validation.json", initial)
    attach_readme(out, "training30kcanonical")
    attach_readme(ROOT, "training30kcanonical")
    history = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in tqdm(range(first, STEPS + 1), desc="E178 DINO+groundeddecoder30k", mininterval=10):
        model.train()
        fraction = max(0.0, (step - 1 - 100) / (STEPS - 100))
        lr = min(1.0, step / 100) * (
            3e-05 + (0.0003 - 3e-05) * 0.5 * (1 + math.cos(math.pi * fraction))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        ids = rng.integers(0, train.count, size=BATCH)
        rng.integers(0, 5, size=BATCH)
        P, F, G, index = train.batch(ids)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.all_tasks(P, F, index).flatten(0, 1)
        assert prediction.dtype == torch.float32
        loss = emd(prediction, G.flatten(0, 1)).mean()
        assert torch.isfinite(loss)
        loss.backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        )
        optimizer.step()
        history.append(float(loss.detach()) * 1000)
        elapsed = previous_elapsed + time.perf_counter() - started
        if step % 50 == 0:
            row = dict(
                step=step,
                train_EMD_mm=float(np.mean(history[-50:])),
                gradient_norm=gradient,
                lr=lr,
                best_step=best_step,
                best_val_EMD_mm=best,
                elapsed_s=elapsed,
                peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            )
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            write(out / "progress.json", row)
            print(json.dumps(row), flush=True)
        if step == 18000:
            save(
                out / "step18000.pt",
                dict(model=model.state_dict(), step=step, config=cfg),
            )
        if step % 1000 == 0:
            canonical = validation(model, val)
            transformed = validation(model, val, val_transform)
            equivariance = equivariance_monitor(model, val)
            if canonical["EMD_mm"] < best:
                best = canonical["EMD_mm"]
                best_step = step
                save(
                    out / "best.pt",
                    dict(model=model.state_dict(), step=step, val_EMD_mm=best, config=cfg),
                )
            row = dict(
                step=step,
                canonical=canonical,
                SE3=transformed,
                equivariance=equivariance,
                best_step=best_step,
                best_val_EMD_mm=best,
            )
            with (out / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            write(out / "validation_progress.json", row)
            print(json.dumps(row), flush=True)
            save(
                out / "resume.pt",
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step,
                    best=best,
                    best_step=best_step,
                    data_rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                    elapsed_s=previous_elapsed + time.perf_counter() - started,
                    config=cfg,
                ),
            )
    save(out / "last.pt", dict(model=model.state_dict(), step=STEPS, config=cfg))
    cfg.update(
        status="trained",
        best_step=best_step,
        best_val_EMD_mm=best,
        elapsed_s=previous_elapsed + time.perf_counter() - started,
    )
    write(out / "config.json", cfg)
    attach_readme(out, "trained30k,evaluationpending")
    attach_readme(ROOT, "trained30k,evaluationpending")
    print("DINO30k training complete; next canonical/SE3/OOD tests", flush=True)


if __name__ == "__main__":
    main()
