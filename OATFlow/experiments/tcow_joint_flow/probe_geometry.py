"""Read cover grasp geometry from frozen TCOW tokens on scene-disjoint data."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
import zipfile

import av
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from tqdm.auto import tqdm

from OATFlow.experiments.tcow_joint_flow.model import expand_depth_channel


class GeometryProbe(nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 16), nn.SiLU(),
                                     nn.Flatten(1), nn.Linear(300 * 16, 3))

    def forward(self, tokens):
        return self.readout(tokens.float())


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in tqdm(iter(lambda: stream.read(4 * 1024**2), b""),
                          total=(path.stat().st_size + 4 * 1024**2 - 1) // (4 * 1024**2),
                          desc=f"hash {path.name}", unit="block", mininterval=5, file=sys.stdout):
            h.update(block)
    return h.hexdigest()


def selected_npz_frames(path, key, frames):
    """Stream only the needed NPY rows, without materializing a whole episode."""
    frames = np.asarray(frames, dtype=np.int64)
    if np.any(np.diff(frames) <= 0):
        raise ValueError("NPZ frame indices must be strictly increasing")
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version: {version}")
        if fortran or dtype.hasobject or frames[0] < 0 or frames[-1] >= shape[0]:
            raise ValueError(f"invalid frame array: {path}, {key}")
        offset = stream.tell()
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        result = np.empty((len(frames), *shape[1:]), dtype=dtype)
        for index, frame in enumerate(frames):
            stream.seek(offset + int(frame) * row_bytes)
            raw = stream.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError(f"truncated array: {path}, {key}")
            result[index] = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
    return result


def prepare_episode(root, row):
    path = root / row["path"]
    meta = json.loads((path / "episode.json").read_text())
    close = next(item for item in meta["phase_intervals"] if item["phase"] == "cover_close")
    ends = [int(meta["decision_frames"]["t_occ"]) - 1,
            int(close["start_frame"]), int(close["end_frame_exclusive"]) - 1]
    indices = [np.rint(np.linspace(0, end, 30)).astype(np.int64) for end in ends]
    frames = np.unique(np.concatenate(indices))
    wanted = set(frames.tolist())
    rgb = {}
    with av.open(str(path / "rgb.mp4")) as video:
        for index, frame in enumerate(video.decode(video=0)):
            if index in wanted:
                rgb[index] = frame.to_ndarray(format="rgb24")
            if index >= frames[-1]:
                break
    if set(rgb) != wanted:
        raise ValueError(f"missing video frames: {path}")
    depth = selected_npz_frames(path / "observation.npz", "depth_m", frames)
    lookup = {int(frame): index for index, frame in enumerate(frames)}
    rgbs = np.stack([np.stack([rgb[int(frame)] for frame in clip]) for clip in indices])
    depths = np.stack([depth[[lookup[int(frame)] for frame in clip]] for clip in indices])
    query = np.asarray(Image.open(path / "query_mask.png")) > 0
    with np.load(path / "debug_poses.npz") as z:
        body = z["body_names"].tolist().index(meta["correct_cover_body"])
        poses = z["poses"][ends, body * 7:(body + 1) * 7]
    # The oracle grasps the cover at body + 0.130 m; the handle spans z=.078:.162.
    rotation = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
    point = poses[:, :3] + rotation[:, :, 2] * .130
    state = selected_npz_frames(path / "observation.npz", "joint_position", np.array([ends[0]]))[0]
    records = [{"episode": row["path"], "seed": int(row["seed"]),
                "group": row["group"], "split": row["split"],
                "target": meta["target_object_id"], "side": int(meta["correct_cover_id"]),
                "phase": phase, "frame": end, "truth_xyz_m": point[index].tolist()}
               for index, (phase, end) in enumerate(zip(("context", "pre_close", "late_close"), ends))]
    return rgbs, depths, query, point.astype(np.float32), state, records


def extract(args, rows, provenance):
    from OATFlow.experiments.tcow_joint_flow.tcow import Seeker
    config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    seeker_args = dict(config["seeker_args"], tracker_pretrained=False)
    tcow = expand_depth_channel(Seeker(logging.getLogger("tcow"), **seeker_args))
    weights = {key.removeprefix("tcow."): value for key, value in saved["model"].items()
               if key.startswith("tcow.")}
    tcow.load_state_dict(weights, strict=True)
    tcow = tcow.to(args.device).eval().requires_grad_(False)
    captured = []

    def capture(_module, _input, output):
        captured.append(output[0][:, :, -1].flatten(2).transpose(1, 2))

    handle = tcow.seeker.tracker_backbone.register_forward_hook(capture)
    features = torch.empty((len(rows) * 3, 300, 768), dtype=torch.bfloat16)
    labels = torch.empty((len(rows) * 3, 3))
    metadata, states = [], []
    with ThreadPoolExecutor(max_workers=4) as loader, torch.inference_mode():
        iterator = iter(rows)
        pending = deque(loader.submit(prepare_episode, args.data, row) for row in
                        [next(iterator) for _ in range(min(4, len(rows)))])
        for episode_index in tqdm(range(len(rows)), desc="extract frozen TCOW geometry", unit="episode",
                                  mininterval=5, file=sys.stdout):
            rgbs, depths, query_mask, points, state, records = pending.popleft().result()
            next_row = next(iterator, None)
            if next_row is not None:
                pending.append(loader.submit(prepare_episode, args.data, next_row))
            rgb = torch.from_numpy(rgbs).permute(0, 4, 1, 2, 3).to(args.device).float()
            rgb = (rgb / 255 - .45) / .225
            depth = torch.from_numpy(depths).to(args.device).float()
            depth = ((depth.clamp(.4, 1.6) - 1) / .6)[:, None]
            rgbd = torch.cat((rgb, depth), dim=1)
            query = torch.zeros((3, 1, 30, 240, 320), device=args.device)
            query[:, 0, 0] = torch.from_numpy(query_mask.copy()).to(args.device)
            captured.clear()
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
                tcow(rgbd, query)
            tokens = captured.pop()
            if tokens.shape != (3, 300, 768) or not torch.isfinite(tokens).all():
                raise RuntimeError("invalid frozen TCOW features")
            features[episode_index * 3:(episode_index + 1) * 3] = tokens.cpu().to(torch.bfloat16)
            labels[episode_index * 3:(episode_index + 1) * 3] = torch.from_numpy(points)
            metadata.extend(records)
            states.append(state)
    handle.remove()
    state_range = np.ptp(np.stack(states), axis=0)
    if np.max(state_range) > 1e-6:
        raise ValueError(f"context robot states differ: {state_range.tolist()}")
    result = {"features": features, "labels": labels, "records": metadata,
              "provenance": provenance, "source_step": int(saved["step"]),
              "context_proprio_range": state_range.tolist()}
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    temp = args.cache.with_suffix(".tmp")
    torch.save(result, temp)
    temp.replace(args.cache)
    print(json.dumps({"event": "cache_ready", "samples": len(features), "source_step": saved["step"],
                      "cache": str(args.cache), "context_proprio_range": state_range.tolist()}), flush=True)
    return result


def error_summary(prediction, truth):
    delta = (prediction - truth) * 1000
    distance = np.linalg.norm(delta, axis=1)
    return {"samples": len(distance), "mean_mm": float(distance.mean()),
            "median_mm": float(np.median(distance)), "p95_mm": float(np.percentile(distance, 95)),
            "axis_mae_mm": np.abs(delta).mean(axis=0).tolist(),
            "axis_bias_mm": delta.mean(axis=0).tolist(),
            "truth_axis_std_mm": (truth.std(axis=0) * 1000).tolist(),
            "within_5mm": float((distance <= 5).mean()), "within_10mm": float((distance <= 10).mean())}


def project_features(features, checkpoint, stage, device):
    """Apply only the frozen checkpoint adapter to existing raw TCOW tokens."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if saved["flow_type"] != "dense_action_expert":
        raise ValueError("adapter probe requires the dense action expert checkpoint")
    # Preserve the seed used by the readout and its shuffled minibatches.
    with torch.random.fork_rng(devices=[]):
        adapter = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256),
                                nn.SiLU(), nn.Linear(256, 960))
    weights = {key.removeprefix("flow.visual."): value for key, value in saved["model"].items()
               if key.startswith("flow.visual.")}
    adapter.load_state_dict(weights, strict=True)
    adapter = adapter.to(device).eval().requires_grad_(False)
    if stage == "bottleneck":
        adapter = adapter[:3]  # 256 channels after SiLU, before expansion.
    width = 256 if stage == "bottleneck" else 960
    projected = torch.empty((len(features), 300, width), dtype=torch.bfloat16)
    with torch.no_grad():
        for start in tqdm(range(0, len(features), 64), desc=f"project {stage}", unit="batch",
                          mininterval=5, file=sys.stdout):
            part = features[start:start + 64].to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                value = adapter(part if device == "cuda" else part.float())
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite adapter features")
            projected[start:start + len(part)] = value.cpu().to(torch.bfloat16)
    return projected


def fit(args, cache):
    records = cache["records"]
    train_context = [i for i, r in enumerate(records) if r["split"] == "train" and r["phase"] == "context"]
    train = [i for i, r in enumerate(records) if r["split"] == "train" and
             (args.train_phases == "all" or r["phase"] == "context")]
    val = [i for i, r in enumerate(records) if r["split"] == "val" and
           (args.train_phases == "all" or r["phase"] == "context")]
    features = cache["features"].to(args.device)
    labels = cache["labels"].to(args.device)
    center = labels[train].mean(dim=0)
    target = (labels - center) / .1
    model = GeometryProbe(features.shape[-1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best, best_state, history = float("inf"), None, []
    for epoch in tqdm(range(args.epochs), desc="train geometry probe", unit="epoch",
                      mininterval=5, file=sys.stdout):
        model.train()
        order = torch.tensor(train, device=args.device)[torch.randperm(len(train), device=args.device)]
        total = 0.0
        for indices in order.split(64):
            loss = (model(features[indices]) - target[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 0 and total == 0:
                grad = model.readout[-1].weight.grad
                if not torch.isfinite(grad).all() or grad.abs().sum() == 0:
                    raise RuntimeError("invalid probe gradients")
                print(json.dumps({"event": "probe_gradient_smoke", "loss": float(loss.detach()),
                                  "trainable_parameters": sum(p.numel() for p in model.parameters())}), flush=True)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.inference_mode():
            estimate = torch.cat([model(features[indices]) for indices in torch.tensor(val, device=args.device).split(64)])
            val_error = float(torch.linalg.vector_norm((estimate - target[val]) * 100, dim=1).mean())
        history.append({"epoch": epoch + 1, "train_mse": total / len(train), "val_mean_mm": val_error})
        if val_error < best:
            best = val_error
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            tqdm.write(json.dumps({"event": "probe_validation", **history[-1], "best_val_mean_mm": best}), file=sys.stdout)
    model.load_state_dict(best_state)
    with torch.inference_mode():
        prediction = torch.cat([model(part) for part in features.split(64)]) * .1 + center
    prediction, truth = prediction.cpu().numpy(), labels.cpu().numpy()
    train_sides = np.array([records[i]["side"] for i in train_context])
    means = np.stack([truth[np.array(train_context)[train_sides == side]].mean(axis=0) for side in (0, 1)])
    groups = {}
    for group, split in (("standard", "train"), ("standard", "val"),
                         ("standard", "test"), ("composition", "test")):
        for phase in ("context", "pre_close", "late_close"):
            indices = [i for i, r in enumerate(records) if (r["group"], r["split"], r["phase"]) == (group, split, phase)]
            baseline = means[[records[i]["side"] for i in indices]]
            groups[f"{group}_{split}_{phase}"] = {
                "probe": error_summary(prediction[indices], truth[indices]),
                "oracle_side_mean": error_summary(baseline, truth[indices])}
    for index, record in enumerate(records):
        record["prediction_xyz_m"] = prediction[index].tolist()
        record["error_mm"] = float(np.linalg.norm(prediction[index] - truth[index]) * 1000)
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "target_center": center.cpu(), "target_scale_m": .1, "source_step": cache["source_step"],
                "feature_stage": args.feature_stage, "feature_width": features.shape[-1],
                "provenance": cache["provenance"]}, args.output / "probe.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = {"source_step": cache["source_step"], "train_samples": len(train), "val_samples": len(val),
               "probe_parameters": sum(p.numel() for p in model.parameters()), "best_val_mean_mm": best,
               "train_phases": args.train_phases,
               "feature_stage": args.feature_stage, "feature_width": features.shape[-1],
               "context_proprio_range": cache["context_proprio_range"], "groups": groups,
               "label": "world cover pose applied to local grasp marker [0,0,0.130] m",
               "protocol": ("Train only context frames; choose by scene-disjoint validation. Close-frame results test transfer to expert arm/contact observations."
                            if args.train_phases == "context" else
                            "Train context and close frames; choose by scene-disjoint validation. Test scenes remain unseen in all phases."),
               "baseline": "Train-context mean point conditioned on oracle GT left/right side; no test labels used to fit means."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", **summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--train-phases", choices=("context", "all"), default="context")
    parser.add_argument("--feature-stage", choices=("tcow", "bottleneck", "adapter"), default="tcow",
                        help="raw 768 tokens, post-SiLU 256 tokens, or final 960 adapter tokens")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.feature_stage != "tcow" and not args.cache.is_file():
        raise FileNotFoundError("adapter probes require an existing raw TCOW cache")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    provenance = {"checkpoint_sha256": sha256(args.checkpoint),
                  "config_sha256": sha256(args.config_checkpoint),
                  "manifest_sha256": sha256(args.data / "manifest.jsonl"), "smoke": args.smoke}
    rows = [json.loads(line) for line in (args.data / "manifest.jsonl").read_text().splitlines()]
    splits = {name: {r["seed"] for r in rows if r["split"] == name} for name in ("train", "val", "test")}
    if any(splits[a] & splits[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("scene split leakage")
    if args.smoke:
        first = next(r for r in rows if r["split"] == "train")
        train_queries = [r for r in rows if r["split"] == "train" and r["seed"] == first["seed"]]
        rows = train_queries + [next(r for r in rows if (r["group"], r["split"]) == pair)
                                for pair in (("standard", "val"), ("standard", "test"),
                                             ("composition", "test"))]
        args.epochs = 2
    (args.output / "config.json").write_text(json.dumps({**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                                        **provenance}, indent=2) + "\n")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False) if args.cache.exists() else extract(args, rows, provenance)
    if cache["provenance"] != provenance:
        raise ValueError("cached features have different checkpoint/data provenance")
    if args.feature_stage != "tcow":
        cache["features"] = project_features(cache["features"], args.checkpoint, args.feature_stage, args.device)
    fit(args, cache)


if __name__ == "__main__":
    main()
