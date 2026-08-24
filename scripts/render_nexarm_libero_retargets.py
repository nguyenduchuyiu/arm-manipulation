"""Retarget all downloaded LIBERO drawer demos and render one NexArm MP4 each."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import imageio.v3 as iio
import mujoco
import numpy as np

from retarget_libero_drawer import (
    CLOSED_DRAWER_QPOS,
    CONTROLLER_HZ,
    CONTROL_STEPS_PER_RECORD,
    OPEN_DRAWER_QPOS,
    RECORD_HZ,
    SCENE,
    SOURCE,
    advance_to_target,
    arm_addresses,
    object_id,
    render,
    robot_touches_bodies,
    solve_initial_pose,
    spline_resample,
    terminal_push_targets,
)


OUTPUT_DIR = Path("outputs/nexarm_libero_50")
MANIFEST = OUTPUT_DIR / "manifest.json"
TARGET_FINAL = np.array([0.533, -0.21, 0.274], dtype=np.float64)
SCENE_ROTATION = np.diag([-1.0, -1.0, 1.0])


def add_label(
    frame: np.ndarray,
    demo_name: str,
    step: int,
    total_steps: int,
    drawer_qpos: float,
) -> np.ndarray:
    labeled = np.ascontiguousarray(frame.copy())
    cv2.rectangle(labeled, (0, 0), (256, 29), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        f"{demo_name} {step + 1}/{total_steps} drawer={drawer_qpos:+.3f}",
        (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(SCENE.resolve()))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=256, width=256)

    arm_actuators, arm_qpos, arm_dofs = arm_addresses(model)
    gripper_actuator = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_control")
    gripper_joint = int(model.actuator_trnid[gripper_actuator, 0])
    gripper_qpos = int(model.jnt_qposadr[gripper_joint])
    drawer_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "top_drawer_joint")
    drawer_body = int(model.jnt_bodyid[drawer_joint])
    drawer_qpos = int(model.jnt_qposadr[drawer_joint])
    drawer_dof = int(model.jnt_dofadr[drawer_joint])
    site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    robot_bodies = {
        object_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in (
            "link_1",
            "link_2",
            "link_3",
            "link_4",
            "link_5",
            "link_6_gripper_base",
            "link_6_left_jaw",
            "link_6_right_jaw",
        )
    }
    cabinet_bodies = {
        object_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("cabinet", "top_drawer", "middle_drawer", "bottom_drawer")
    }

    results = []
    with h5py.File(SOURCE, "r") as source:
        source_data = source["data"]
        demo_names = sorted(
            source_data.keys(), key=lambda name: int(name.removeprefix("demo_"))
        )

        for demo_name in demo_names:
            teacher_positions = source_data[f"{demo_name}/obs/ee_pos"][()]
            targets = (SCENE_ROTATION @ teacher_positions.T).T
            targets += TARGET_FINAL - targets[-1]
            control_targets = list(spline_resample(targets)[1:])
            control_targets += terminal_push_targets(targets[-1])

            mujoco.mj_resetData(model, data)
            data.qpos[drawer_qpos] = OPEN_DRAWER_QPOS
            data.qpos[gripper_qpos] = 0.0
            mujoco.mj_forward(model, data)

            try:
                solve_initial_pose(model, data, site_id, arm_qpos, arm_dofs, targets[0])
            except RuntimeError as error:
                result = {"demo": demo_name, "success": False, "error": str(error)}
                results.append(result)
                print(f"{demo_name}: FAIL {error}", flush=True)
                continue

            data.ctrl[arm_actuators] = data.qpos[arm_qpos]
            data.ctrl[gripper_actuator] = 0.0
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            for _ in range(100):
                data.qpos[drawer_qpos] = OPEN_DRAWER_QPOS
                data.qvel[drawer_dof] = 0.0
                mujoco.mj_step(model, data)

            if robot_touches_bodies(model, data, cabinet_bodies, robot_bodies):
                result = {
                    "demo": demo_name,
                    "success": False,
                    "error": "Robot overlaps cabinet in the initial pose",
                }
                results.append(result)
                print(f"{demo_name}: FAIL initial overlap", flush=True)
                continue

            frames = []
            contact_steps = 0
            record_steps = len(control_targets) // CONTROL_STEPS_PER_RECORD
            for step in range(record_steps):
                target_slice = control_targets[
                    step * CONTROL_STEPS_PER_RECORD :
                    (step + 1) * CONTROL_STEPS_PER_RECORD
                ]
                contact = False
                for target in target_slice:
                    advance_to_target(
                        model,
                        data,
                        target,
                        site_id,
                        arm_actuators,
                        arm_qpos,
                        arm_dofs,
                    )
                    contact = contact or robot_touches_bodies(
                        model, data, {drawer_body}, robot_bodies
                    )
                contact_steps += contact
                frame = render(renderer, data, "front")
                frames.append(
                    add_label(
                        frame,
                        demo_name,
                        step,
                        record_steps,
                        float(data.qpos[drawer_qpos]),
                    )
                )

            final_drawer_qpos = float(data.qpos[drawer_qpos])
            success = final_drawer_qpos >= CLOSED_DRAWER_QPOS
            video_path = OUTPUT_DIR / f"{demo_name}.mp4"
            iio.imwrite(video_path, np.asarray(frames), fps=RECORD_HZ)
            result = {
                "demo": demo_name,
                "source_steps": len(teacher_positions),
                "video_frames": len(frames),
                "final_drawer_qpos": final_drawer_qpos,
                "contact_steps": int(contact_steps),
                "success": bool(success),
                "video": str(video_path),
            }
            results.append(result)
            print(
                f"{demo_name}: success={success} drawer={final_drawer_qpos:+.4f} "
                f"contacts={contact_steps}",
                flush=True,
            )

    renderer.close()
    success_count = sum(result["success"] for result in results)
    manifest = {
        "source": str(SOURCE),
        "scene": str(SCENE),
        "num_demos": len(results),
        "successes": success_count,
        "controller_hz": CONTROLLER_HZ,
        "record_hz": RECORD_HZ,
        "trajectory_resampling": "savgol_then_cubic_spline",
        "results": results,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"success={success_count}/{len(results)}")
    print(f"manifest={MANIFEST}")
    if success_count != len(results):
        raise RuntimeError(f"Only {success_count}/{len(results)} NexArm retargets succeeded")


if __name__ == "__main__":
    main()
