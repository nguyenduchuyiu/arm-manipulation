"""Evaluate query-conditioned TCOW masks on scene-disjoint validation episodes."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from OATFlow.policy.data import split_rows
from OATFlow.policy.loader import ActionChunkDataset, make_dataloader
from OATFlow.policy.model import load_training_tracker, set_tracker_input
from OATFlow.policy.tracking import Seeker


CHANNELS = ("target_amodal", "frontmost_occluder", "outermost_container")


def foreground_iou(prediction, truth):
    """Empty GT masks are absent-class cases, never perfect foreground IoU."""
    intersection = (prediction & truth).sum((-1, -2))
    union = (prediction | truth).sum((-1, -2))
    positive = truth.any(dim=(-1, -2))
    iou = intersection.float() / union.clamp_min(1)
    return iou.masked_fill(~positive, float("nan")), positive


def validation_plan(root, rows):
    samples, groups = [], []
    for index, row in enumerate(rows):
        meta = json.loads((root / row["path"] / "episode.json").read_text())
        plan = meta["context_plan"]
        points = [(plan["reveal_frames"] - 1, "visible")]
        cursor = plan["reveal_frames"] + plan["occlude_frames"]
        for swap in plan["swaps"]:
            points.append((cursor + swap["frames"] // 2, "shuffle"))
            cursor += swap["frames"] + swap["pause_frames"]
        points += [(meta["decision_frames"]["t_occ"] - 1, "post_shuffle"),
                   (meta["decision_frames"]["t_obj"], "exposed"),
                   (meta["grasp_frames"]["object"], "grasp"),
                   (meta["frames"] - 2, "lift")]
        for frame, group in points:
            samples.append((index, frame, False))
            groups.append(group)
    return samples, groups


def high_iou(result, threshold):
    values = list(result["foreground_iou"].values()) + [result["occluded_target_iou"]]
    values += list(result["post_shuffle_cover_iou"].values())
    absent = [value for value in result["absent_false_positive_pixel_rate"].values() if value is not None]
    return (all(value is not None and value >= threshold for value in values)
            and all(value <= .005 for value in absent))


@torch.inference_mode()
def evaluate(model, loader, samples, groups, rows):
    model.eval()
    scores, positives, false_rates, records = [], [], [], []
    offset = 0
    for tensors in tqdm(loader, desc="TCOW val IoU", unit="batch", mininterval=5):
        rgb, query, truth = [value.cuda(non_blocking=True) for value in tensors[:3]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(rgb, query)
        if not torch.isfinite(logits[:, :, -1]).all():
            raise ValueError("nonfinite TCOW validation logits")
        predicted = logits[:, :, -1] > 0
        ground_truth = truth[:, :, -1] > .5
        iou, positive = foreground_iou(predicted, ground_truth)
        rate = predicted.float().mean((-1, -2)).masked_fill(positive, float("nan"))
        iou, positive, rate = (value.cpu().numpy() for value in (iou, positive, rate))
        scores.append(iou)
        positives.append(positive)
        false_rates.append(rate)
        for i in range(len(iou)):
            episode, frame, _ = samples[offset + i]
            records.append(dict(reference=rows[episode]["path"], frame=frame, group=groups[offset + i],
                                iou={name: float(iou[i, channel]) if positive[i, channel] else None
                                     for channel, name in enumerate(CHANNELS)}))
        offset += len(iou)
        del rgb, query, truth, logits, predicted, ground_truth
    scores, positives, rates = map(np.concatenate, (scores, positives, false_rates))
    means = lambda values: {name: float(np.nanmean(values[:, i])) if np.isfinite(values[:, i]).any() else None
                            for i, name in enumerate(CHANNELS)}
    hidden = positives[:, 1] & np.isfinite(scores[:, 0])
    after_shuffle = np.array(groups) == "post_shuffle"
    foreground = means(scores)
    result = dict(scenes=len(rows), clips=len(samples), split="standard/val", metric="mean per-current-frame foreground IoU",
                  foreground_iou=foreground,
                  mean_foreground_iou=float(np.mean(list(foreground.values()))) if all(v is not None for v in foreground.values()) else None,
                  occluded_target_iou=float(scores[hidden, 0].mean()) if hidden.any() else None,
                  post_shuffle_cover_iou={name: means(scores[after_shuffle])[name] for name in CHANNELS[1:]},
                  absent_false_positive_pixel_rate=means(rates),
                  positive_frames={name: int(positives[:, i].sum()) for i, name in enumerate(CHANNELS)},
                  threshold_logit=0., query_frame_scored=False, records=records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "config-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--episode", type=Path,
                        help="evaluate one validation episode instead of the full validation split")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    _, rows = split_rows(args.data)
    if args.episode is not None:
        episode = args.episode.resolve()
        rows = [row for row in rows if (args.data / row["path"]).resolve() == episode]
        if len(rows) != 1:
            raise ValueError(f"episode is not a unique standard/val scene: {args.episode}")
    samples, groups = validation_plan(args.data, rows)
    videos = {(args.data / row["path"] / "rgb.mp4").resolve(): str(args.data / row["path"] / "rgb.mp4") for row in rows}
    dataset = ActionChunkDataset(args.data, rows, samples, videos, False, False)
    loader = make_dataloader(dataset, [list(range(start, min(start + args.batch_size, len(samples))))
                                      for start in range(0, len(samples), args.batch_size)], 0, True)
    config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    seeker_args = dict(config["seeker_args"])
    seeker_args["tracker_pretrained"] = False
    model = set_tracker_input(Seeker(logging.getLogger("tcow"), **seeker_args))
    load_training_tracker(model, source)
    result = evaluate(model.cuda(), loader, samples, groups, rows)
    args.output.mkdir(parents=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "README.md").write_text(
        f"# TCOW val IoU\n\nCheckpoint {args.checkpoint}; config {args.config_checkpoint}; data {args.data}. "
        f"{len(rows)} val scenes, {len(samples)} current-frame endpoints;30-frame causal RGB/query clips, batch{args.batch_size}. "
        "Foreground IoU excludes empty GT; report absent-mask false positive area separately. "
        "Tracking evaluation only; no FM actions/physical rollout. Status: complete. Results: results.json.\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
