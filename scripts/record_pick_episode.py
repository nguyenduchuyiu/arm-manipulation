"""Run NexArmEnv episode to pick up the red cube and record MP4 videos.

Executes a 5-phase pick-and-lift trajectory, saves MP4 videos (front and wrist cameras)
and saves PNG snapshots for key phases of the trajectory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.nexarm_env import NexArmEnv


# Analytical waypoints for NexArm to pick cube far at [0.78, 0.0, 0.025]
Q_HOVER = np.array([1.8344, 1.4293, 1.6111, 1.7453, 0.0007], dtype=np.float32)
Q_GRASP = np.array([1.8150, 1.9798, 1.3934, 1.7453, -0.0008], dtype=np.float32)
Q_LIFT  = np.array([1.8383, 1.3065, 1.6560, 1.7453, 0.0016], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a NexArm pick and lift episode")
    parser.add_argument("--scene", default="assets/robot/scene.xml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True)

    env = NexArmEnv(
        scene_path=args.scene,
        image_height=384,
        image_width=384,
        frame_skip=10,
        max_episode_steps=500,
        randomize_object=False,
        render_mode="rgb_array",
    )

    obs, info = env.reset(seed=args.seed)
    obj_pos = info["object_position"]
    print(f"Initial object position: {obj_pos.tolist()}")

    plan = [
        # (target_q, gripper_val, steps, snapshot_name)
        (Q_HOVER, 0.02, 60, "1_hover"),      # Phase 1: Hover above cube
        (Q_GRASP, 0.02, 40, "2_approach"),   # Phase 2: Lower jaws around cube
        (Q_GRASP, 0.00, 40, "3_grasp"),      # Phase 3: Squeeze gripper
        (Q_LIFT,  0.00, 80, "4_lifting"),    # Phase 4: Lift cube into air
        (Q_LIFT,  0.00, 80, "5_lifted"),     # Phase 5: Hold lifted cube
    ]

    front_frames = [obs["observation.images.front"]]
    wrist_frames = [obs["observation.images.wrist"]]
    snapshots = {}

    curr_q = obs["observation.state"][:5].copy()
    curr_g = obs["observation.state"][5]

    for p_idx, (t_q, t_g, steps, snap_key) in enumerate(plan):
        s_q = curr_q.copy()
        s_g = curr_g
        for s in range(steps):
            alpha = (s + 1) / float(steps)
            t = 0.5 * (1.0 - np.cos(alpha * np.pi))

            cmd_q = (1.0 - t) * s_q + t * t_q
            cmd_g = (1.0 - t) * s_g + t * t_g

            action = np.append(cmd_q, cmd_g).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            front_frames.append(obs["observation.images.front"])
            wrist_frames.append(obs["observation.images.wrist"])

        snapshots[f"pick_{snap_key}_front.png"] = obs["observation.images.front"]
        snapshots[f"pick_{snap_key}_wrist.png"] = obs["observation.images.wrist"]

        print(f"Phase {p_idx+1} ({snap_key}): obj_z={info['object_z']:.4f} m, dist_grasp={info['object_to_grasp_distance']:.4f} m, success={info['success']}")

    fps = round(1.0 / env.control_dt)
    front_video_path = output_dir / "pick_front.mp4"
    wrist_video_path = output_dir / "pick_wrist.mp4"

    iio.imwrite(front_video_path, np.array(front_frames), fps=fps)
    iio.imwrite(wrist_video_path, np.array(wrist_frames), fps=fps)

    for name, img in snapshots.items():
        iio.imwrite(output_dir / name, img)

    env.close()

    print(f"\nEpisode finished:")
    print(f"  Total steps      : {info['elapsed_steps']}")
    print(f"  Success          : {info['success']}")
    print(f"  Terminated reason: {info['terminated_reason']}")
    print(f"  Final object z   : {info['object_z']:.4f} m")
    print(f"  Front MP4 video  : {front_video_path}")
    print(f"  Wrist MP4 video  : {wrist_video_path}")


if __name__ == "__main__":
    main()
