"""Test 1: receive a delta action and produce motion through LeRobot's NexArm
backend.

LeRobot's control contract is ABSOLUTE raw servo positions 0..4095. A delta
action (per-step increment) is applied on top: next = current_raw + delta. This
is the minimal test that a delta-action source (e.g. a VLA) moves the arm via
the LeRobot control path. Records front+wrist MP4s.

Run: python scripts/test_delta_action.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.nexarm_mujoco_backend import JOINT_NAMES, NexArmMujocoBackend

FPS = 50
STEPS = 80
# Delta per step (raw 0..4095). Swing shoulder_lift + elbow + wrist a little,
# then reverse, so motion is visible in both directions.
DELTA = np.array([0.0, 25.0, -20.0, 15.0, 0.0, 0.0], dtype=np.float64)
OUT = Path("outputs")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    backend = NexArmMujocoBackend(
        model_path=Path("assets/robot/scene.xml"),
        fps=FPS, camera_width=384, camera_height=384,
        camera_names=("front", "wrist"),
    )
    backend.reset(settle_steps=200)

    names = list(JOINT_NAMES)
    cur = backend.joint_positions()
    cur_raw = np.array([cur[f"{n}.pos"] for n in names], dtype=np.float64)
    print(f"home raw: {np.round(cur_raw,1).tolist()}")

    front = [backend.render("front")]
    wrist = [backend.render("wrist")]

    half = STEPS // 2
    for s in range(STEPS):
        delta = DELTA if s < half else -DELTA
        target_raw = cur_raw + delta
        action = {f"{names[j]}.pos": float(np.clip(target_raw[j], 0, 4095)) for j in range(6)}
        backend.step(action)
        cur_raw = np.array([backend.joint_positions()[f"{n}.pos"] for n in names], dtype=np.float64)
        front.append(backend.render("front"))
        wrist.append(backend.render("wrist"))
        if s == 0 or s == half:
            print(f"step {s}: raw={np.round(cur_raw,1).tolist()}")

    iio.imwrite(OUT / "delta_front.mp4", np.array(front), fps=FPS)
    iio.imwrite(OUT / "delta_wrist.mp4", np.array(wrist), fps=FPS)
    backend.close()
    print(f"\nframes: {len(front)}")
    print(f"front: {OUT/'delta_front.mp4'}")
    print(f"wrist: {OUT/'delta_wrist.mp4'}")


if __name__ == "__main__":
    main()
