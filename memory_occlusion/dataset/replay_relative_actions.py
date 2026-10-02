"""Replay ten relative-action episodes with live chunk anchors before training."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from memory_occlusion.dataset.episode import context
from memory_occlusion.environment.env import MemoryOcclusionEnv
from memory_occlusion.experiments.tcow_joint_flow.rollout import physical_action
from memory_occlusion.task import normalized


def select_rows(rows):
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r: (r["seed"], r["target"]))
    targets = sorted({r["target"] for r in train})
    selected, seeds = [], set()
    for index in range(10):
        candidates = [r for r in train if r["target"] == targets[index % len(targets)]
                      and r["swaps"] == index % 3 + 1 and r["seed"] not in seeds]
        if not candidates:
            raise ValueError("need ten distinct train scenes spanning targets and swaps")
        selected.append(candidates[0])
        seeds.add(candidates[0]["seed"])
    return selected


def replay(data, row, output, statistics):
    path = data / row["path"]
    meta = json.loads((path / "episode.json").read_text())
    with np.load(path / "relative_actions.npz") as z:
        starts, relative, weights, recorded_anchor = z["starts"], z["action"], z["valid"], z["anchor"]
    with np.load(path / "supervision.npz") as z:
        original = z["expert_action"]
    mean, std = np.array(statistics["action_mean"]), np.array(statistics["action_std"])
    standardized = ((relative - mean) / std).astype(np.float32)
    decoded = standardized * std + mean
    roundtrip = decoded.copy()
    roundtrip[..., :5] += recorded_anchor[:, None, :5]
    indices = np.minimum(starts[:, None] + np.arange(25), len(original) - 1)
    error = float(np.abs(roundtrip - original[indices])[weights].max())
    if error > 2e-6 or not np.allclose(decoded[..., 5], relative[..., 5], rtol=0, atol=2e-7):
        raise ValueError(f"normalized roundtrip failed: {path}")
    output.mkdir()
    env = MemoryOcclusionEnv(context_fps=25, resolution=320,
                             max_episode_steps=int(weights[:, :10].sum()) + 25)
    writer = iio.imopen(output / "replay_25hz.mp4", "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=25, pixel_format="yuv420p")
    frames, executed, max_anchor_error, max_command_error = 0, 0, 0., 0.
    trace = []

    def write(rgb, phase):
        nonlocal frames
        frame = Image.fromarray(rgb[40:280].copy())
        ImageDraw.Draw(frame).text((5, 5), f"Relative replay K10 | {phase} | frame {frames}", fill="white", stroke_width=1, stroke_fill="black")
        writer.write(np.asarray(frame), is_batch=False)
        frames += 1

    try:
        env.reset(seed=meta["seed"], options={"upright_targets": True, "layout": meta["layout"]})
        env.data.time = 0.
        context(env, meta["context_plan"], lambda phase, valid: write(env._overview()[0], phase))
        if frames != starts[0]:
            raise ValueError("context/action frame boundary differs")
        env.query(meta["target_object_id"])
        env.choose_cover(meta["correct_cover_body"])
        limits = env.model.actuator_ctrlrange.copy()
        np.testing.assert_allclose(limits, meta["normalization_limits"], rtol=0, atol=1e-8)
        done = False
        for index in tqdm(range(len(starts)), desc=f"replay {path.name}", unit="chunk",
                          mininterval=5, file=sys.stdout):
            anchor = normalized(env.data.qpos[env.robot_qpos_addresses], limits)
            max_anchor_error = max(max_anchor_error, float(np.abs(anchor - recorded_anchor[index]).max()))
            absolute = decoded[index].copy()
            absolute[:, :5] += anchor[None, :5]
            # All ten executed actions use the same observed chunk-start anchor.
            for offset in range(min(10, int(weights[index].sum()))):
                action = absolute[offset].astype(np.float32)
                action[:5] = np.clip(action[:5], -1, 1)
                action[5] = float(action[5] >= .5)
                frame = int(starts[index] + offset)
                max_command_error = max(max_command_error, float(np.abs(action - original[frame]).max()))
                observation, _, terminated, truncated, _ = env.step(physical_action(action, limits))
                write(observation["overview_rgb"], "action")
                trace.append({"frame": frame, "chunk_start": int(starts[index]),
                              "anchor": anchor.tolist(), "absolute_action": action.tolist(),
                              "success": bool(env.success), "target_grasped": bool(env.target_grasped)})
                executed += 1
                if terminated or truncated:
                    done = True
                    break
            if done:
                break
        result = {"episode": row["path"], "seed": row["seed"], "target": row["target"], "swaps": row["swaps"],
                  "execute_chunk": 10, "anchor": "live observed chunk-start proprio", "fps": 25,
                  "video_frames": frames, "action_steps": executed, "roundtrip_max_error": error,
                  "max_live_anchor_error": max_anchor_error, "max_live_command_error": max_command_error,
                  "cover_removed": bool(env._cover_in_drop_zone()), "target_grasped": bool(env.target_grasped),
                  "success": bool(env.success), "failure_reason": env.failure_reason,
                  "video": str(output / "replay_25hz.mp4")}
    finally:
        writer.close()
        env.close()
    (output / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace))
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "replay_episode_complete", **result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = select_rows([json.loads(line) for line in (args.data / "manifest.jsonl").read_text().splitlines()])
    statistics = json.loads((args.data / "normalization.json").read_text())
    (args.output / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    results = []
    for row in tqdm(rows, desc="replay gate", unit="episode", file=sys.stdout):
        results.append(replay(args.data, row, args.output / Path(row["path"]).name, statistics))
    summary = {"episodes": len(results), "successes": sum(r["success"] for r in results),
               "passed": all(r["success"] and r["cover_removed"] and r["target_grasped"] for r in results),
               "roundtrip_max_error": max(r["roundtrip_max_error"] for r in results), "results": results}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "replay_gate_complete", **summary}), flush=True)
    if not summary["passed"]:
        raise RuntimeError("relative replay gate failed; do not start training")


if __name__ == "__main__":
    main()
