"""Solve a uniform NexArm workspace grid and collect verified 50 Hz joint trajectories."""

from sam3_pick.paths import ASSETS
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from tqdm import tqdm
from sam3_pick.simulation.robot_scene import (
    assert_robot_table_only,
    assert_user_arm_configuration,
    robot_table_xml,
)

ROBOT = ASSETS / "robot/physics_reference.xml"
SAMPLES = ASSETS / "robot/ik_seeds.npz"
ARM_JOINTS = [
    "joint_1_base_to_link_1",
    "joint_2_link_1_to_link_2",
    "joint_3_link_2_to_link_3",
    "joint_4_link_3_to_link_4",
    "joint_5_link_4_to_link_5",
]
CHANNELS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
FPS = 50
DT = 1 / FPS
IK_TOL = 0.0005
ARRIVAL_TOL = 0.001
CLOSING_TOL = 0.002
PENETRATION_TOL = 0.0002
SPEED = 0.8
CLOSE_SECONDS = 0.6
HOLD_SECONDS = 0.5
WORK = {}


class TrajectoryFailure(RuntimeError):
    pass


def physics_xml():
    return robot_table_xml()


def set_configuration(data, q, gripper=0.0):
    data.qpos[WORK["arm_qpos"]] = q
    data.qpos[WORK["right_qpos"]] = gripper
    data.qpos[WORK["left_qpos"]] = -gripper
    data.qpos[WORK["pinion_qpos"]] = WORK["gear_ratio"] * gripper


def point_and_jacobian(q):
    model, data = (WORK["model"], WORK["static"])
    set_configuration(data, q)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac, None, WORK["site"])
    return (data.site_xpos[WORK["site"]].copy(), jac[:, WORK["arm_dof"]])


def collision_free(q, gripper=0.0):
    model, data = (WORK["model"], WORK["static"])
    set_configuration(data, q, gripper)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_collision(model, data)
    return not any((c.dist < -1e-06 for c in data.contact))


def edge_free(a, b):
    steps = max(1, int(np.ceil(np.max(np.abs(b - a)) / 0.04)))
    return all((collision_free(a + (b - a) * t) for t in np.linspace(0, 1, steps + 1)[1:]))


def plan_path(start, goal, rng):
    if edge_free(start, goal):
        return np.stack((start, goal))
    trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]

    def extend(tree, target):
        nodes, parents = tree
        nearest = int(np.argmin(np.linalg.norm(np.array(nodes) - target, axis=1)))
        delta = target - nodes[nearest]
        length = np.linalg.norm(delta)
        candidate = nodes[nearest] + delta * min(1.0, 0.3 / max(length, 1e-12))
        if not edge_free(nodes[nearest], candidate):
            return (None, False)
        nodes.append(candidate)
        parents.append(nearest)
        return (len(nodes) - 1, length <= 0.3)

    def trace(tree, index):
        nodes, parents = tree
        path = []
        while index >= 0:
            path.append(nodes[index])
            index = parents[index]
        return path[::-1]

    for iteration in range(160):
        a, b = (iteration % 2, 1 - iteration % 2)
        target = trees[b][0][0] if iteration % 5 == 0 else rng.uniform(WORK["low"], WORK["high"])
        index_a, _ = extend(trees[a], target)
        if index_a is None:
            continue
        target = trees[a][0][index_a]
        for _ in range(40):
            index_b, reached = extend(trees[b], target)
            if index_b is None:
                break
            if reached:
                path_a, path_b = (trace(trees[a], index_a), trace(trees[b], index_b))
                path = path_a + path_b[-2::-1]
                if a == 1:
                    path = path[::-1]
                short, i = ([path[0]], 0)
                while i < len(path) - 1:
                    j = len(path) - 1
                    while j > i + 1 and (not edge_free(path[i], path[j])):
                        j -= 1
                    short.append(path[j])
                    i = j
                return np.array(short)
    return None


def command(desired, gripper, data):
    compensation = data.qfrc_bias[WORK["arm_dof"]] / WORK["model"].actuator_gainprm[:5, 0]
    return np.r_[np.clip(desired + compensation, WORK["low"], WORK["high"]), gripper]


def initialize_worker(scene):
    model = mujoco.MjModel.from_xml_string(scene)
    assert_robot_table_only(model)
    assert_user_arm_configuration(model)
    joint_ids = [model.joint(name).id for name in ARM_JOINTS]
    arm_qpos, arm_dof = (model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids])
    if model.nu != 6 or not np.array_equal(model.actuator_trnid[:5, 0], joint_ids):
        raise ValueError("Expected five arm actuators followed by the gripper actuator")
    if not np.allclose(model.actuator_ctrlrange[:5], model.jnt_range[joint_ids]):
        raise ValueError("Arm command limits differ from joint limits")
    right_id = model.joint("right_jaw_slide_joint").id
    eq_id = model.equality("gripper_pinion_to_right_jaw").id
    with np.load(SAMPLES) as samples:
        valid = samples["collision_checked_prefix_valid"]
        seeds = samples["q_rad"][: len(valid)][valid]
        points = samples["points_base_m"][: len(valid)][valid] + samples["base_origin_world_m"]
    WORK.update(
        model=model,
        static=mujoco.MjData(model),
        data=mujoco.MjData(model),
        arm_qpos=arm_qpos,
        arm_dof=arm_dof,
        site=model.site("grasp_site").id,
        low=model.actuator_ctrlrange[:5, 0],
        high=model.actuator_ctrlrange[:5, 1],
        right_qpos=int(model.jnt_qposadr[right_id]),
        left_qpos=int(model.jnt_qposadr[model.joint("left_jaw_slide_joint").id]),
        pinion_qpos=int(model.jnt_qposadr[model.joint("gripper_pinion_joint").id]),
        gear_ratio=float(model.eq_data[eq_id, 1]),
        tree=cKDTree(points),
        seeds=seeds,
        state_addresses=np.r_[arm_qpos, model.jnt_qposadr[right_id]],
    )
    data = WORK["data"]
    substeps = round(DT / model.opt.timestep)
    if not np.isclose(substeps * model.opt.timestep, DT):
        raise ValueError("Physics timestep must divide the 50 Hz control period")
    WORK["substeps"] = substeps
    mujoco.mj_forward(model, data)
    for _ in range(75):
        data.ctrl[:] = command(np.zeros(5), 0.0, data)
        mujoco.mj_step(model, data, substeps)
        mujoco.mj_forward(model, data)
    if any((c.dist < -PENETRATION_TOL for c in data.contact)):
        raise ValueError("Initial home pose penetrates collision geometry")
    WORK["home"] = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())


def simulate(path, target):
    model, data = (WORK["model"], WORK["data"])
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:], data.ctrl[:] = WORK["home"]
    mujoco.mj_forward(model, data)
    actions, references, phases = ([], [], [])
    positions, velocities, tcp = (
        [data.qpos.copy()],
        [data.qvel.copy()],
        [data.site_xpos[WORK["site"]].copy()],
    )
    minimum_contact_distance = 0.0

    def step(desired, gripper, phase):
        nonlocal minimum_contact_distance
        action = command(desired, gripper, data)
        data.ctrl[:] = action
        actions.append(action)
        references.append(np.r_[desired, gripper])
        phases.append(phase)
        for _ in range(WORK["substeps"]):
            mujoco.mj_step(model, data)
            if len(data.contact):
                distance = min((c.dist for c in data.contact))
                minimum_contact_distance = min(minimum_contact_distance, distance)
                if distance < -PENETRATION_TOL:
                    raise TrajectoryFailure("dynamic_collision")
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise TrajectoryFailure("nonfinite_state")
        if np.max(np.abs(data.qvel[WORK["arm_dof"]])) > 3.0:
            raise TrajectoryFailure("excessive_joint_speed")
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        tcp.append(data.site_xpos[WORK["site"]].copy())

    for start, goal in zip(path[:-1], path[1:]):
        frames = max(25, int(np.ceil(1.875 * np.max(np.abs(goal - start)) / SPEED * FPS)))
        for i in range(frames):
            t = (i + 1) / frames
            fraction = t**3 * (10 - 15 * t + 6 * t**2)
            step(start + fraction * (goal - start), 0.0, 0)
    desired = path[-1].copy()

    def correct():
        error = target - data.site_xpos[WORK["site"]]
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, WORK["site"])
        jac = jac[:, WORK["arm_dof"]]
        delta = 0.5 * jac.T @ np.linalg.solve(jac @ jac.T + 1e-05 * np.eye(3), error)
        return np.clip(desired + np.clip(delta, -0.008, 0.008), WORK["low"], WORK["high"])

    stable = 0
    for _ in range(125):
        desired = correct()
        step(desired, 0.0, 1)
        error = np.linalg.norm(data.site_xpos[WORK["site"]] - target)
        ready = error <= ARRIVAL_TOL and np.max(np.abs(data.qvel[WORK["arm_dof"]])) < 0.05
        stable = stable + 1 if ready else 0
        if stable >= 5:
            break
    else:
        raise TrajectoryFailure("arrival_tracking_failure")
    close_start = len(actions)
    for i in range(round(CLOSE_SECONDS * FPS)):
        t = (i + 1) / round(CLOSE_SECONDS * FPS)
        fraction = t**3 * (10 - 15 * t + 6 * t**2)
        desired = correct()
        step(desired, -0.0255 * fraction, 2)
        if np.linalg.norm(data.site_xpos[WORK["site"]] - target) > CLOSING_TOL:
            raise TrajectoryFailure("closing_ee_drift")
    for _ in range(round(HOLD_SECONDS * FPS)):
        desired = correct()
        step(desired, -0.0255, 3)
        if np.linalg.norm(data.site_xpos[WORK["site"]] - target) > CLOSING_TOL:
            raise TrajectoryFailure("closed_hold_ee_drift")
    if abs(data.qpos[WORK["right_qpos"]] + 0.0255) > 0.0005:
        raise TrajectoryFailure("gripper_not_closed")
    final_error = float(np.linalg.norm(data.site_xpos[WORK["site"]] - target))
    if final_error > ARRIVAL_TOL:
        raise TrajectoryFailure("final_position_error")
    actions, positions, velocities, tcp = map(np.array, (actions, positions, velocities, tcp))
    phases = np.asarray(phases, np.uint8)
    frames = len(actions)
    trajectory = {
        "action_absolute": actions.astype(np.float32),
        "joint_reference": np.asarray(references, np.float32),
        "state_joint_position": positions[:, WORK["state_addresses"]].astype(np.float32),
        "state_full_qpos": positions.astype(np.float64),
        "state_full_qvel": velocities.astype(np.float64),
        "ee_position_world_m": tcp.astype(np.float32),
        "timestamp_action_s": np.arange(frames, dtype=np.float64) / FPS,
        "timestamp_state_s": np.arange(frames + 1, dtype=np.float64) / FPS,
        "phase": phases,
        "target_world_m": target.astype(np.float64),
        "initial_ctrl": WORK["home"][2].copy(),
    }
    quality = {
        "frames": frames,
        "duration_s": frames / FPS,
        "arrival_error_m": float(np.linalg.norm(tcp[close_start] - target)),
        "final_error_m": final_error,
        "max_closing_hold_error_m": float(np.linalg.norm(tcp[close_start:] - target, axis=1).max()),
        "min_contact_distance_m": minimum_contact_distance,
        "final_gripper_position_m": float(data.qpos[WORK["right_qpos"]]),
        "max_joint_speed_rad_s": float(np.abs(velocities[:, WORK["arm_dof"]]).max()),
    }
    return (trajectory, quality)


def solve_voxel(job):
    voxel_id, target = job
    result = {
        "voxel_id": int(voxel_id),
        "target_world_m": target.tolist(),
        "ik_reachable": False,
        "trajectory_success": False,
        "reason": "ik_not_found",
    }
    _, indices = WORK["tree"].query(target, k=3)
    seeds = WORK["seeds"][indices].copy()
    seeds[0, 4] = 0.0
    seeds = np.vstack((np.zeros(5), seeds))
    candidates = []
    closest = np.inf
    for seed in seeds:

        def residual(q):
            return point_and_jacobian(q)[0] - target

        def jacobian(q):
            return point_and_jacobian(q)[1]

        optimum = least_squares(
            residual,
            np.clip(seed, WORK["low"], WORK["high"]),
            jac=jacobian,
            bounds=(WORK["low"], WORK["high"]),
            max_nfev=80,
            ftol=1e-09,
            xtol=1e-09,
            gtol=1e-09,
        )
        error = float(np.linalg.norm(residual(optimum.x)))
        closest = min(closest, error)
        if error > IK_TOL:
            continue
        if not all((collision_free(optimum.x, grip) for grip in np.linspace(0, -0.0255, 9))):
            result["reason"] = "endpoint_collision"
            continue
        if not any((np.linalg.norm(optimum.x - q) < 0.03 for q in candidates)):
            candidates.append(optimum.x.copy())
    result["best_ik_error_m"] = closest
    if not candidates:
        return (result, None)
    home = WORK["home"][0][WORK["arm_qpos"]]
    candidates.sort(key=lambda q: np.linalg.norm(q - home))
    result["ik_reachable"] = True
    result["ik_q_rad"] = candidates[0].tolist()
    result["reason"] = "path_not_found"
    rng = np.random.default_rng(np.random.SeedSequence([42, int(voxel_id)]))
    for candidate in candidates:
        path = plan_path(home, candidate, rng)
        if path is None:
            continue
        try:
            trajectory, quality = simulate(path, target)
        except TrajectoryFailure as error:
            result["reason"] = str(error)
            continue
        result.update(
            trajectory_success=True,
            reason="accepted",
            ik_q_rad=candidate.tolist(),
            path_waypoints=len(path),
            **quality,
        )
        trajectory["planned_joint_path"] = path
        return (result, trajectory)
    return (result, None)


def uniform_grid(edge, origin):
    xy = np.arange(-int(np.ceil(0.475 / edge)), int(np.ceil(0.475 / edge)) + 1)
    z = np.arange(int(np.ceil(0.582 / edge)))
    ix, iy, iz = np.meshgrid(xy, xy, z, indexing="ij")
    indices = np.column_stack((ix.ravel(), iy.ravel(), iz.ravel())).astype(np.int32)
    base_centers = (indices + np.array([0, 0, 0.5])) * edge
    return (indices, base_centers, base_centers + origin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Near-home voxel count for smoke tests; zero tests the full grid",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep tested centers and continue the unfinished grid",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --resume, retry centers without an accepted trajectory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.edge <= 0
        or not 1 <= args.workers <= 32
        or args.limit < 0
        or (args.retry_failed and (not args.resume))
    ):
        raise ValueError("Require positive edge, 1..32 workers and nonnegative limit")
    selected_cores = sorted(os.sched_getaffinity(0))[: args.workers]
    os.sched_setaffinity(0, selected_cores)
    scene = physics_xml()
    preserved = []
    if args.resume:
        previous = json.loads((args.output / "config.json").read_text())
        if (
            previous["grid_edge_m"] != args.edge
            or previous["fps"] != FPS
            or previous["source_robot_sha256"] != hashlib.sha256(ROBOT.read_bytes()).hexdigest()
            or ((args.output / "scene.xml").read_text().strip() != scene.strip())
        ):
            raise ValueError("Resume requires the same grid, robot, physics scene and frequency")
        backup = args.output / "index_before_resume.jsonl"
        backup_number = 1
        while backup.exists():
            backup_number += 1
            backup = args.output / f"index_before_resume_{backup_number:02d}.jsonl"
        shutil.copy2(args.output / "index.jsonl", backup)
        previous_rows = [json.loads(line) for line in backup.read_text().splitlines()]
        preserved = (
            [row for row in previous_rows if row["trajectory_success"]]
            if args.retry_failed
            else previous_rows
        )
        if any(
            (
                not (args.output / row["trajectory_file"]).is_file()
                for row in preserved
                if row["trajectory_success"]
            )
        ):
            raise FileNotFoundError("An accepted trajectory is missing")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "trajectories").mkdir()
        (args.output / "scene.xml").write_text(scene + "\n")
    with np.load(SAMPLES) as samples:
        origin = samples["base_origin_world_m"].copy()
    indices, base_centers, targets = uniform_grid(args.edge, origin)
    home_target = np.array([0.539369, -0.039412, 0.230434])
    order = np.argsort(np.linalg.norm(targets - home_target, axis=1), kind="stable")
    if args.limit:
        order = order[: args.limit]
    preserved_ids = {row["voxel_id"] for row in preserved}
    order = np.array([i for i in order if int(i) not in preserved_ids], dtype=int)
    config = {
        "status": "running",
        "grid_edge_m": args.edge,
        "total_grid_voxels": len(targets),
        "tested_grid_voxels": len(order) + len(preserved),
        "preserved_centers": len(preserved),
        "preserved_trajectories": sum((row["trajectory_success"] for row in preserved)),
        "resumed": args.resume,
        "retry_failed": args.retry_failed,
        "grid_center_rule": "base x=i*edge, y=j*edge, z=(k+0.5)*edge",
        "base_origin_world_m": origin.tolist(),
        "workers": args.workers,
        "cpu_core_limit": len(selected_cores),
        "fps": FPS,
        "physics_timestep_s": 0.002,
        "ik_tolerance_m": IK_TOL,
        "arrival_final_tolerance_m": ARRIVAL_TOL,
        "closing_hold_tolerance_m": CLOSING_TOL,
        "physics_penetration_tolerance_m": PENETRATION_TOL,
        "max_reference_joint_speed_rad_s": SPEED,
        "close_seconds": CLOSE_SECONDS,
        "closed_hold_seconds": HOLD_SECONDS,
        "orientation": "unrestricted",
        "gripper_open_m": 0.0,
        "gripper_closed_m": -0.0255,
        "action_representation": "absolute_joint",
        "action_channels": CHANNELS,
        "action_units": ["rad"] * 5 + ["m"],
        "state_convention": "T commands; T+1 states. Command t acts during [t/50,(t+1)/50).",
        "phases": {
            "0": "move_open",
            "1": "align_open",
            "2": "close",
            "3": "hold_closed",
        },
        "source_robot": str(ROBOT),
        "source_robot_sha256": hashlib.sha256(ROBOT.read_bytes()).hexdigest(),
        "source_samples": str(SAMPLES),
        "model": "scene.xml",
        "ik": "bounded analytic-Jacobian least_squares, exact-zero home plus three KD-tree seeds, 80 evaluations each",
        "planning": "straight joint path; bidirectional RRT-Connect up to 160 iterations",
        "seed": 42,
        "physics": "fixed robot/table configuration from assets; no other objects",
        "user_arm_configuration": assert_user_arm_configuration(
            mujoco.MjModel.from_xml_string(scene)
        ),
        "software": {"mujoco": mujoco.__version__, "numpy": np.__version__},
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    readme = args.output / "README.md"
    readme.write_text(
        f"# Uniform voxel IK and 50 Hz absolute-joint reaching data\n\nStatus: running.\n\nConfiguration: config.json. Scene: scene.xml. Source: sam3_pick/data/voxel/generate.py. No training, weights, frozen modules or policy evaluation are involved. Grid: {args.edge:g} m edge, {len(order) + len(preserved):,}/{len(targets):,} centers tested; {len(preserved):,} tested centers and {sum((row['trajectory_success'] for row in preserved)):,} trajectories preserved from the prior pass. seed 42. CPU-only, {len(selected_cores)} selected cores.\n\nEach accepted episode starts at a common dynamically settled home pose with open gripper, follows a collision-checked absolute-joint path, aligns the measured TCP, then closes and holds the gripper using gravity compensation and Cartesian feedback. Frames are recorded at exactly 50 Hz. No image recording. Commands have six channels: five revolute joint positions in radians and right-jaw slide position in metres. Gripper command 0=open, -0.0255=closed.\n\ntrajectory_success labels only dynamically verified episodes. ik_reachable labels collision-free IK configurations throughout a sampled closure sweep; unsuccessful IK is unconfirmed, not a proof of unreachability. Only voxel centers are tested. The model uses the user's configure_mvp_model, table top and legs; no other objects. Closing is in empty space. EE goals remain the voxel centers, without object-grasp offsets.\n"
    )
    status = np.full(len(targets), -1, np.int8)
    ik_valid = np.zeros(len(targets), bool)
    q_solutions = np.full((len(targets), 5), np.nan)
    reasons, accepted_errors, counts = (Counter(), [], Counter())
    for record in preserved:
        voxel_id = record["voxel_id"]
        ik_valid[voxel_id] = record["ik_reachable"]
        status[voxel_id] = 2 if record["trajectory_success"] else int(record["ik_reachable"])
        if record["ik_reachable"]:
            q_solutions[voxel_id] = record["ik_q_rad"]
        counts.update(tested=1, ik_reachable=int(record["ik_reachable"]))
        reasons[record["reason"]] += 1
        if record["trajectory_success"]:
            accepted_errors.append(record["final_error_m"])
            counts.update(accepted=1, frames=record["frames"])
    started = time.monotonic()
    jobs = ((int(i), targets[i]) for i in order)
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_worker, initargs=(scene,)
    ) as pool:
        with (args.output / "index.jsonl").open("w", buffering=1) as index_file:
            for record in preserved:
                index_file.write(json.dumps(record) + "\n")
            progress = tqdm(
                pool.map(solve_voxel, jobs, chunksize=4),
                total=len(order),
                desc="IK + 50 Hz trajectory collection",
                mininterval=2,
            )
            for result, trajectory in progress:
                voxel_id = result["voxel_id"]
                ik_valid[voxel_id] = result["ik_reachable"]
                if result["ik_reachable"]:
                    q_solutions[voxel_id] = result["ik_q_rad"]
                status[voxel_id] = 2 if trajectory is not None else int(result["ik_reachable"])
                reasons[result["reason"]] += 1
                counts["tested"] += 1
                counts["ik_reachable"] += int(result["ik_reachable"])
                if trajectory is not None:
                    path = Path("trajectories") / f"v{voxel_id:06d}.npz"
                    np.savez_compressed(args.output / path, **trajectory)
                    result["trajectory_file"] = str(path)
                    accepted_errors.append(result["final_error_m"])
                    counts["accepted"] += 1
                    counts["frames"] += result["frames"]
                index_file.write(json.dumps(result) + "\n")
                if counts["tested"] % 128 == 0:
                    progress.set_postfix(ik=counts["ik_reachable"], saved=counts["accepted"])
                    (args.output / "progress.json").write_text(json.dumps(dict(counts)) + "\n")
    np.savez_compressed(
        args.output / "grid.npz",
        grid_indices=indices,
        centers_base_m=base_centers,
        centers_world_m=targets,
        status=status,
        ik_reachable=ik_valid,
        ik_q_rad=q_solutions,
        edge_m=args.edge,
        status_labels=np.array(["unconfirmed", "ik_only", "trajectory_verified"]),
        base_origin_world_m=origin,
    )
    summary = {
        "status": "complete",
        "grid_voxels": len(targets),
        **dict(counts),
        "reasons": dict(reasons),
        "elapsed_seconds": time.monotonic() - started,
        "final_ee_error_m": {
            "max": max(accepted_errors) if accepted_errors else None,
            "mean": float(np.mean(accepted_errors)) if accepted_errors else None,
        },
        "grid_status": {
            "-1": "not tested",
            "0": "unconfirmed IK",
            "1": "IK valid, no accepted trajectory",
            "2": "trajectory verified",
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    config["status"] = "complete"
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    readme.write_text(
        readme.read_text().replace("Status: running.", "Status: complete.")
        + f"\nResults: {counts['tested']:,} centers tested, {counts['ik_reachable']:,} IK-valid, {counts['accepted']:,} accepted trajectories, {counts['frames']:,} commands. See summary.json, index.jsonl, grid.npz and trajectories/*.npz.\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
