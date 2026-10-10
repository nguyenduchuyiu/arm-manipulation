"""Batched learned-policy evaluation with paired color queries and unseen layouts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import imageio.v3 as iio
import numpy as np
import torch
from tqdm.auto import tqdm

from OATFlow.environment.control import step_joint_waypoint
from OATFlow.environment.env import MemoryOcclusionEnv
from OATFlow.environment.success import jaw_contacts
from OATFlow.pick.scene import setup
from OATFlow.prior.model import PriorPolicy
from OATFlow.task import TARGETS, normalized, physical_joint_action


def benchmark_rows(data, split, all_permutations=False):
    rows = [json.loads(line) for line in (data / "manifest.jsonl").read_text().splitlines()]
    groups = sorted({r["seed"] for r in rows if r["split"] == split})
    if len(groups) < 4:
        raise ValueError("evaluation needs at least four layout groups per split")
    # Across four layouts exhaust all 24 permutations, querying all four colors
    # at each physical state. No validation/test trajectory enters training.
    selected = []
    for index, seed in enumerate(groups[:4]):
        selected.extend(r for r in rows if r["seed"] == seed and (all_permutations or r["permutation"] // 6 == index))
    if len(selected) != (384 if all_permutations else 96):
        raise ValueError("incomplete paired-query benchmark")
    return selected


def rollout_batch(model, saved, data, rows, output, offset, frames, k):
    envs, metas, initial, results, writers, traces = [], [], [], [], [], []
    for index, row in enumerate(rows):
        meta = json.loads((data / row["path"] / "episode.json").read_text())
        env = MemoryOcclusionEnv()
        setup(env, meta["scene"], meta["permutation"])
        envs.append(env); metas.append(meta); initial.append(env.target_positions())
        results.append(dict(reference=row["path"], seed=row["seed"], permutation=row["permutation"],
                            target=meta["target_object_id"], target_slot=row["target_slot"],
                            success=False, reason="timeout", frames=0, held=0,
                            max_lift_m={name: 0. for name in TARGETS},
                            max_jaw_contacts={name: 0 for name in TARGETS}))
        writer = None
        # One complete four-query scene per split, including failed rollouts.
        if offset + index < 4:
            path = output / f"case{offset + index:03d}_{meta['target_object_id']}.mp4"
            writer = iio.imopen(path, "w", plugin="pyav")
            writer.init_video_stream("libx264", fps=25, pixel_format="yuv420p")
            results[-1]["video"] = path.name
        writers.append(writer)
        traces.append(dict(qpos=[], command=[], objects=[], jaws=[],
                           initial_qpos=env.data.qpos.copy(), initial_qvel=env.data.qvel.copy(),
                           prediction_frame=[], prediction=[], plan_qpos=[]))
    active = np.ones(len(rows), bool)
    generator = torch.Generator(device="cuda").manual_seed(0)
    chunk = None
    try:
        for frame in range(frames):
            if not active.any():
                break
            if frame % k == 0:
                rgb = np.stack([env._overview() for env in envs])
                wrist = np.stack([env.wrist_image() for env in envs])
                proprio = np.stack([normalized(env.data.qpos[env.robot_qpos_addresses], env.model.actuator_ctrlrange) for env in envs])
                views = [torch.from_numpy(v).permute(0, 3, 1, 2).cuda() for v in (rgb, wrist)]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    inputs = (*views, torch.from_numpy(proprio).cuda(),
                              torch.zeros(len(rows), dtype=torch.long, device="cuda"),
                              torch.tensor([m["target_id"] for m in metas], device="cuda"))
                    prediction = (model.sample(*inputs) if model.action_head_type == 'act' else
                                  model.sample(*inputs, model.flow.sample_noise(len(rows) // 4, "cuda", generator).repeat_interleave(4, 0)))
                chunk = prediction.float().cpu().numpy()
                if not np.isfinite(chunk).all():
                    raise ValueError("nonfinite action prediction")
                for index, env in enumerate(envs):
                    if offset + index < 4 and active[index]:
                        traces[index]["prediction_frame"].append(frame)
                        traces[index]["prediction"].append(chunk[index].copy())
                        traces[index]["plan_qpos"].append(env.data.qpos.copy())
            for index, env in enumerate(envs):
                if not active[index]:
                    continue
                result, target = results[index], metas[index]["target_object_id"]
                if writers[index]:
                    writers[index].write(np.concatenate((env._overview(), env.wrist_image()), axis=1), is_batch=False)
                command = physical_joint_action(chunk[index, frame % k], env.model.actuator_ctrlrange)
                command = np.clip(command, *env.model.actuator_ctrlrange.T)
                step_joint_waypoint(env.model, env.data, command)
                positions = env.target_positions()
                contacts = {name: len(jaw_contacts(env.model, env.data, name)) for name in TARGETS}
                for name in TARGETS:
                    result["max_lift_m"][name] = max(result["max_lift_m"][name], float(positions[name][2] - initial[index][name][2]))
                    result["max_jaw_contacts"][name] = max(result["max_jaw_contacts"][name], contacts[name])
                result["frames"] = frame + 1
                lifted = positions[target][2] - initial[index][target][2] > .04
                result["held"] = result["held"] + 1 if lifted and contacts[target] == 2 else 0
                wrong = [name for name in TARGETS if name != target and positions[name][2] - initial[index][name][2] > .04]
                trace = traces[index]
                trace["qpos"].append(env.data.qpos[env.robot_qpos_addresses].copy())
                trace["command"].append(command.copy())
                trace["objects"].append(np.array([positions[name] for name in TARGETS]))
                trace["jaws"].append([contacts[name] for name in TARGETS])
                # Failure takes precedence if both targets are lifted.
                if wrong:
                    result.update(reason="wrong_object_lifted", wrong_object=wrong[0]); active[index] = False
                elif any(position[2] < -.02 for position in positions.values()):
                    result["reason"] = "object_off_table"; active[index] = False
                elif result["held"] >= 10:
                    result.update(success=True, reason="target_lifted"); active[index] = False
        for index, result in enumerate(results):
            if offset + index < 4:
                np.savez_compressed(output / f"case{offset + index:03d}_trace.npz", **traces[index])
        return results
    finally:
        for writer in writers:
            if writer:
                writer.close()
        for env in envs:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--execute-chunk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--all-permutations", action="store_true", help="evaluate every query/permutation in four held-out groups")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 4 or args.batch_size % 4 or args.execute_chunk < 1 or args.max_frames < 1:
        raise ValueError("batch must contain complete four-query scenes; positive frames and K required")
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if saved["config"]["objective"] != "target_lift":
        raise ValueError("pick-only checkpoint required")
    horizon = saved["config"]["horizon"]
    if args.execute_chunk > horizon:
        raise ValueError("execute chunk exceeds the checkpoint action horizon")
    action_head = saved['config'].get('action_head', 'fm')
    if saved['architecture'] != f'oracle_task_target_{action_head}':
        raise ValueError('checkpoint architecture differs from action head')
    model = PriorPolicy(horizon=horizon, action_head=action_head).cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    rows = benchmark_rows(args.data, args.split, args.all_permutations)
    args.output.mkdir(parents=True)
    config = dict(**{k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  checkpoint_epoch=saved["epoch"], checkpoint_step=saved["step"],
                  horizon=horizon, action_head=action_head, flow_steps=10 if action_head == 'fm' else 0,
                  noise_seed=0 if action_head == 'fm' else None,
                  paired_query_noise='ACT zero latent' if action_head == 'act' else "identical per scene/timestep", policy_hz=25, servo_hz=50,
                  temporal_ensemble=False, control=f"learned {action_head.upper()} policy; no expert actions",
                  selection=f"four layouts x{24 if args.all_permutations else 6} permutations x4 paired target queries",
                  success="correct target >4cm, both jaws, ten consecutive frames; wrong lift fails",
                  training=saved["config"])
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    readme = (f"# {args.output.name}\n\nLearned pick-only policy closed-loop evaluation. "
              f"Checkpoint {args.checkpoint}, epoch {saved['epoch']}, step {saved['step']}. "
              f"Split {args.split}; {len(rows)} cases, paired queries; H{horizon}/K{args.execute_chunk}, head {action_head}, "
              f"{'Euler10,seed0' if action_head == 'fm' else 'single pass,zero latent'}; "
              f"{args.max_frames} frame budget; 25Hz waypoints/50Hz interpolation, no TE. "
              "Only color query/current cameras/proprio enter policy. No oracle action or slot ID. "
              "Training configuration/provenance and test settings: config.json. Status: running.\n")
    (args.output / "README.md").write_text(readme)
    results = []; started = time.perf_counter()
    with (args.output / "results.jsonl").open("w") as file:
        for offset in tqdm(range(0, len(rows), args.batch_size), desc=f"pick {args.split}", file=sys.stdout):
            batch = rollout_batch(model, saved, args.data, rows[offset:offset + args.batch_size],
                                  args.output, offset, args.max_frames, args.execute_chunk)
            results.extend(batch)
            for result in batch:
                file.write(json.dumps(result) + "\n")
            file.flush()
            print(json.dumps(dict(event="eval_progress", output=str(args.output), cases=len(results), successes=sum(r["success"] for r in results))), flush=True)
    confusion = {target: Counter(r.get("wrong_object", r["target"] if r["success"] else "no_lift")
                                for r in results if r["target"] == target) for target in TARGETS}
    summary = dict(cases=len(results), successes=sum(r["success"] for r in results),
                   success_rate=sum(r["success"] for r in results) / len(results),
                   reasons=Counter(r["reason"] for r in results), confusion=confusion,
                   by_target={target: dict(cases=sum(r["target"] == target for r in results),
                                           successes=sum(r["target"] == target and r["success"] for r in results)) for target in TARGETS},
                   by_slot={str(slot): dict(cases=sum(r["target_slot"] == slot for r in results),
                                            successes=sum(r["target_slot"] == slot and r["success"] for r in results)) for slot in range(4)},
                   wall_s=time.perf_counter() - started)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "README.md").write_text(readme.replace("Status: running", "Status: complete") + f"\n{json.dumps(summary)}\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
