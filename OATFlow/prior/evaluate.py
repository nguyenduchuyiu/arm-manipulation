"""Balanced closed-loop evaluation: both targets of every task, train and held-out."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import imageio.v3 as iio
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from OATFlow.task import physical_joint_action
from OATFlow.environment.env import TARGETS, MemoryOcclusionEnv
from OATFlow.dataset.collect_expert import collect_one
from OATFlow.task import TASKS, task_id_from_state
from OATFlow.prior.model import PriorPolicy
from OATFlow.environment.success import cover_deposited
from OATFlow.task import normalized


def description(config, status):
    train = config["training"]
    return (f"# {config['name']}\n\nLearned vision/FM policy evaluation: {config['scenes']} scenes.\n"
            f"Checkpoint: {config['checkpoint']}; step {config['step']}; data {train['data']}.\n"
            f"Training: {train['epochs']} epochs, batch {train['batch_size']}, LR {train['lr']}; "
            "SmolVLA FM initialization, frozen ImageNet vision, trainable adapters/embeddings/proprio/decoder/FM.\n"
            f"Actions: {train['action_representation']}, H25/K{config['execute_chunk']}, Euler10, 25Hz, noise0.\n"
            f"Frame budget {config['max_frames']}; batch {config['inference_batch']} simulations with separate physics processes, current RGB only.\n"
            "Held-out seeds are disjoint from training; expert trajectories verify scene feasibility.\n"
            "Success: drop selected cover in broad green zone; lift correct target >4cm with both jaws for ten frames.\n"
            "Config: config.json; policy videos and results in per-scene folders; aggregate: summary.json/results.jsonl.\n"
            f"Status: {status}.\n")


def contacts(env, body):
    body_id = env.model.body(body).id
    jaws = {env.model.geom(f"link_6_{side}_jaw_collision_0").id for side in ("left", "right")}
    touching = {jaw for contact in env.data.contact for jaw in jaws
                if ((contact.geom1 == jaw and env.model.geom_bodyid[contact.geom2] == body_id)
                    or (contact.geom2 == jaw and env.model.geom_bodyid[contact.geom1] == body_id))}
    return touching == jaws


def make_scene(reference, split, output, config):
    meta = json.loads((reference / "episode.json").read_text())
    env = MemoryOcclusionEnv(resolution=320)
    with np.load(reference / "sim_state.npz") as z:
        env.data.qpos[:] = z["initial_qpos"]
        env.data.qvel[:] = z["initial_qvel"]
        env.data.ctrl[:] = z["initial_ctrl"]
    env.assignment = meta["object_to_occluder"]
    mujoco.mj_forward(env.model, env.data)
    if task_id_from_state(env, meta["task"]["objects"]) != meta["task_id"]:
        raise ValueError("oracle task ID differs from simulator state")
    target, cover = meta["target_object_id"], meta["selected_cover"]
    path = output / f"{split}_t{meta['task_id']:02d}_{target}"
    path.mkdir()
    scene_config = {**config, "reference": str(reference), "split": split,
                    "seed": meta["seed"], "task_id": meta["task_id"], "target": target}
    (path / "config.json").write_text(json.dumps(scene_config, indent=2) + "\n")
    (path / "README.md").write_text(description(scene_config, "running") +
                                     f"\nThis scene: {split}, task {meta['task_id']}, target {target}, seed {meta['seed']}.\n")
    writer = iio.imopen(path / "rollout.mp4", "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=25, pixel_format="yuv420p")
    result = dict(split=split, task_id=meta["task_id"], target=target, seed=meta["seed"],
                  reference=str(reference), path=str(path.relative_to(output)), success=False,
                  reason="frame_budget", frames=0, cover_lifted=False, cover_removed=False,
                  max_cover_lift_m=0., max_target_lift_m=0., target_contact_frames=0,
                  target_grasped=False, target_lifted=False,
                  cover_removed_frame=None)
    return dict(env=env, meta=meta, target=target, cover=cover, path=path, writer=writer,
                initial=env.target_positions(), cover_z=float(env.data.xpos[env.model.body(cover).id, 2]),
                held=0, cover_held=0,
                result=result,
                qpos=[], action=[], target_xyz=[], cover_xyz=[])


def advance_scene(scene, frame, config):
    env, result = scene["env"], scene["result"]
    command = physical_joint_action(scene["chunk"][frame % config["execute_chunk"]], env.model.actuator_ctrlrange)
    command = np.clip(command, *env.model.actuator_ctrlrange.T).astype(np.float32)
    env.data.ctrl[:] = command
    scene["qpos"].append(env.data.qpos.copy())
    scene["action"].append(command.copy())
    mujoco.mj_step(env.model, env.data, 20)
    positions = env.target_positions()
    target_pos = positions[scene["target"]]
    cover_pos = env.data.xpos[env.model.body(scene["cover"]).id].copy()
    scene["target_xyz"].append(target_pos.copy())
    scene["cover_xyz"].append(cover_pos)
    gain = float(target_pos[2] - scene["initial"][scene["target"]][2])
    cover_gain = float(cover_pos[2] - scene["cover_z"])
    target_contacts = contacts(env, scene["target"])
    cover_contacts = contacts(env, scene["cover"])
    scene["cover_held"] = scene["cover_held"] + 1 if cover_gain > .04 and cover_contacts else 0
    cover_removed = cover_deposited(env.model, env.data, scene["cover"])
    if cover_removed and not result["cover_removed"]:
        result["cover_removed_frame"] = frame + 1
    result.update(frames=frame + 1, cover_lifted=result["cover_lifted"] or scene["cover_held"] >= 5,
                  cover_removed=result["cover_removed"] or bool(cover_removed),
                  max_cover_lift_m=max(result["max_cover_lift_m"], cover_gain),
                  max_target_lift_m=max(result["max_target_lift_m"], gain),
                  target_contact_frames=result["target_contact_frames"] + int(target_contacts))
    scene["held"] = scene["held"] + 1 if gain > .04 and target_contacts and cover_removed else 0
    result["target_grasped"] |= scene["held"] >= 10
    if any(positions[name][2] > scene["initial"][name][2] + .04 for name in TARGETS if name != scene["target"]):
        result["reason"] = "wrong_object_lifted"
        return True
    elif scene["held"] >= 10:
        result.update(success=True, target_lifted=True, reason="target_lifted")
        return True
    return False


def finish_scene(scene, elapsed):
    result = scene["result"]
    if result["reason"] == "frame_budget":
        result["failure_stage"] = ("target_grasp" if result["cover_removed"] else
                                   "cover_transport" if result["cover_lifted"] else "cover_grasp")
    result["batch_wall_s"] = elapsed
    (scene["path"] / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(scene["path"] / "trace.npz", qpos=np.stack(scene["qpos"]),
                        command=np.stack(scene["action"]), target_xyz=np.stack(scene["target_xyz"]),
                        cover_xyz=np.stack(scene["cover_xyz"]))
    p = scene["path"] / "README.md"
    p.write_text(p.read_text().replace("Status: running", "Status: complete") + f"\nResult: {json.dumps(result)}\n")
    return result


def worker_observation(scene):
    env = scene["env"]
    return dict(views=(env._overview(), env.wrist_image()),
                proprio=normalized(env.data.qpos[env.robot_qpos_addresses], env.model.actuator_ctrlrange),
                task_id=scene["meta"]["task_id"], target_id=scene["meta"]["target_id"],
                result=scene["result"].copy())


def rollout_worker(connection, reference, split, output, config):
    scene = make_scene(reference, split, output, config)
    started = time.perf_counter()
    try:
        observation = worker_observation(scene)
        connection.send(dict(done=False, observation=observation))
        frame = 0
        while frame < config["max_frames"]:
            scene["chunk"] = connection.recv()
            env = scene["env"]
            for index in range(config["execute_chunk"]):
                views = observation["views"] if index == 0 else (env._overview(), env.wrist_image())
                scene["writer"].write(np.concatenate(views, axis=1), is_batch=False)
                done = advance_scene(scene, frame, config)
                frame += 1
                if done or frame >= config["max_frames"]:
                    result = finish_scene(scene, time.perf_counter() - started)
                    connection.send(dict(done=True, result=result))
                    return
            observation = worker_observation(scene)
            connection.send(dict(done=False, observation=observation))
    finally:
        scene["writer"].close()
        scene["env"].close()
        connection.close()


@torch.inference_mode()
def run_batch(model, references, output, config):
    context = get_context("spawn")
    actors = []
    for reference, split in references:
        parent, child = context.Pipe()
        process = context.Process(target=rollout_worker, args=(child, reference, split, output, config))
        process.start()
        child.close()
        actors.append(dict(connection=parent, process=process,
                           generator=torch.Generator(device="cuda").manual_seed(0)))
    active = list(actors)
    results = []
    started = time.perf_counter()
    try:
        for actor in active:
            actor["observation"] = actor["connection"].recv()["observation"]
        progress = tqdm(total=config["max_frames"], desc="policy frames", unit="frame", mininterval=10, file=sys.stdout)
        for frame in range(0, config["max_frames"], config["execute_chunk"]):
            if not active:
                break
            observations = [actor["observation"] for actor in active]
            overview, wrist = [torch.from_numpy(np.stack([obs["views"][i] for obs in observations])).permute(0, 3, 1, 2).cuda()
                               for i in (0, 1)]
            proprio = torch.from_numpy(np.stack([obs["proprio"] for obs in observations])).cuda()
            task_ids = torch.tensor([obs["task_id"] for obs in observations], device="cuda")
            target_ids = torch.tensor([obs["target_id"] for obs in observations], device="cuda")
            noise = torch.cat([model.flow.sample_noise(1, "cuda", actor["generator"]) for actor in active])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunks = model.sample(overview, wrist, proprio, task_ids, target_ids, noise).float().cpu().numpy()
            if not np.isfinite(chunks).all():
                raise ValueError("nonfinite policy actions")
            for actor, chunk in zip(active, chunks):
                actor["connection"].send(chunk)
            finished = []
            for actor in active:
                message = actor["connection"].recv()
                if message["done"]:
                    actor["process"].join()
                    result = message["result"]
                    results.append(result)
                    print(json.dumps(dict(event="scene_complete", **result)), flush=True)
                    finished.append(actor)
                else:
                    actor["observation"] = message["observation"]
            for actor in finished:
                active.remove(actor)
            progress.update(min(config["execute_chunk"], config["max_frames"] - frame))
            if progress.n % 100 == 0:
                live_results = results + [actor["observation"]["result"] for actor in active]
                live = dict(status="running", frame=progress.n, scenes=live_results)
                (output / "live.json").write_text(json.dumps(live, indent=2) + "\n")
                print(json.dumps(dict(event="live_scene_metrics", frame=progress.n,
                                      cover_lifted=sum(row["cover_lifted"] for row in live_results),
                                      cover_removed=sum(row["cover_removed"] for row in live_results),
                                      successes=sum(row["success"] for row in live_results))), flush=True)
        progress.close()
        print(json.dumps(dict(event="batch_complete", scenes=len(results), wall_s=time.perf_counter()-started)), flush=True)
        return results
    finally:
        for actor in actors:
            actor["connection"].close()
            if actor["process"].is_alive():
                actor["process"].terminate()
            actor["process"].join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--references", type=Path, help="reuse prepared held-out references")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-chunk", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=1800)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.execute_chunk <= 25 or args.max_frames < 1:
        raise ValueError("K must be 1–25; positive frame budget")
    rows = [json.loads(line) for line in (args.data / "manifest.jsonl").read_text().splitlines()]
    train_seeds = {row["seed"] for row in rows}
    args.output.mkdir(parents=True)
    train_config = json.loads((args.checkpoint.parent / "config.json").read_text())
    config = dict(name=args.output.name, checkpoint=str(args.checkpoint), step=train_config["total_steps"],
                  training=train_config, horizon=25, execute_chunk=args.execute_chunk, flow_steps=10,
                  noise_seed=0, max_frames=args.max_frames, scenes=48, inference_batch=8, physics_workers=8)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "README.md").write_text(description(config, "preparing held-out scenes"))
    if args.references:
        heldout_root = args.references
        heldout_rows = [json.loads(line) for line in (heldout_root / "manifest.jsonl").read_text().splitlines()]
        expected = {(task_id, target_index) for task_id in range(12) for target_index in (0, 1)}
        if len(heldout_rows) != 24 or {(row["task_id"], row["demo_id"]) for row in heldout_rows} != expected:
            raise ValueError("prepared held-out coverage differs")
        if train_seeds & {row["seed"] for row in heldout_rows}:
            raise ValueError("prepared held-out seed leaked from training")
        for row in heldout_rows:
            meta = json.loads((heldout_root / row["path"] / "episode.json").read_text())
            if not meta["success"] or meta["seed"] != row["seed"]:
                raise ValueError("prepared expert reference is incomplete")
    else:
        heldout_root = args.output / "references"
        heldout_root.mkdir()
        (heldout_root / "README.md").write_text("# Held-out references\n\n24 fresh seeds, two targets per task, verified by physical expert. Used only for evaluation. Training: none here. Evaluation checkpoint, H/K, noise seed and frame budget in ../config.json. Config: ../config.json; reference episode.json files; manifest.jsonl. Status: collecting.\n")
        jobs = [(heldout_root, task["task_id"], target_index, 2000000 + task["task_id"] * 1000 + target_index)
                for task in TASKS for target_index in (0, 1)]
        heldout_rows = []
        with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(collect_one, job) for job in jobs]
            for future in tqdm(as_completed(futures), total=24, desc="verify fresh scenes", unit="scene", file=sys.stdout):
                row = future.result()
                if row["seed"] in train_seeds:
                    raise ValueError("held-out seed leaked from training")
                heldout_rows.append(row)
        (heldout_root / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in heldout_rows))
        p = heldout_root / "README.md"
        p.write_text(p.read_text().replace("Status: collecting", "Status: complete"))
    config["heldout_references"] = str(heldout_root)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    heldout_rows.sort(key=lambda row: (row["task_id"], row["demo_id"]))
    train_references = {}
    for row in rows:
        if row["split"] != "train" or row["group"] != "standard":
            continue
        reference = args.data / row["path"]
        meta = json.loads((reference / "episode.json").read_text())
        train_references.setdefault((meta["task_id"], meta["target_id"]), reference)
    expected = [(task["task_id"], TARGETS.index(target)) for task in TASKS for target in task["objects"]]
    if any(key not in train_references for key in expected):
        raise ValueError("balanced evaluation requires both targets of every task in train data")
    references = [(train_references[key], "train") for key in expected]
    references += [(heldout_root / row["path"], "heldout") for row in heldout_rows]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if saved["architecture"] != "oracle_task_target_fm" or saved["step"] != config["step"]:
        raise ValueError("checkpoint architecture/step differs from configuration")
    if train_config["action_representation"] != "absolute_joint":
        raise ValueError("absolute joint checkpoint required")
    model = PriorPolicy().cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    del saved
    (args.output / "README.md").write_text(description(config, "evaluating"))
    results = []
    started = time.perf_counter()
    for start in tqdm(range(0, len(references), 8), desc="scene batches", unit="batch", file=sys.stdout):
        batch_results = run_batch(model, references[start:start + 8], args.output, config)
        results.extend(batch_results)
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in batch_results))
        partial = {split: dict(scenes=sum(row["split"] == split for row in results),
                               successes=sum(row["split"] == split and row["success"] for row in results),
                               cover_removed=sum(row["split"] == split and row["cover_removed"] for row in results))
                   for split in ("train", "heldout")}
        (args.output / "progress.json").write_text(json.dumps(partial, indent=2) + "\n")
        print(json.dumps(dict(event="evaluation_progress", **partial)), flush=True)
    summary = dict(status="complete", scenes=48, wall_s=time.perf_counter() - started,
                   checkpoint=str(args.checkpoint), step=config["step"], execute_chunk=args.execute_chunk,
                   heldout_seeds_disjoint=True, expert_rejected_attempts=sum(len(row["rejected_attempts"]) for row in heldout_rows))
    for split in ("train", "heldout"):
        subset = [row for row in results if row["split"] == split]
        summary[split] = dict(scenes=len(subset), successes=sum(row["success"] for row in subset),
                             success_rate=sum(row["success"] for row in subset) / len(subset),
                             cover_lifted=sum(row["cover_lifted"] for row in subset),
                             cover_removed=sum(row["cover_removed"] for row in subset),
                             failure_stages={stage: sum(row.get("failure_stage") == stage for row in subset)
                                             for stage in ("cover_grasp", "cover_transport", "target_grasp")},
                             wrong_object_lifted=sum(row["reason"] == "wrong_object_lifted" for row in subset))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "README.md").write_text(description(config, "complete") + f"\nResults: {json.dumps(summary)}\n")
    print(json.dumps(dict(event="evaluation_complete", **summary)), flush=True)


if __name__ == "__main__":
    main()
