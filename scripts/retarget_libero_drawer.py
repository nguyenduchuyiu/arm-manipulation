"""Retarget one LIBERO drawer demonstration to NexArm and record a new HDF5."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import imageio.v3 as iio
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


SOURCE = Path(
    "data/libero_sample/"
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5"
)
SCENE = Path("assets/robot/libero_cabinet_scene.xml")
OUTPUT = Path("data/nexarm_libero/close_top_drawer_demo_0.hdf5")
VIDEO = Path("outputs/nexarm_libero_close_drawer.mp4")

CONTROLLER_HZ = 100
RECORD_HZ = 20
PHYSICS_STEPS_PER_CONTROL = 5  # 500 Hz MuJoCo -> 100 Hz controller.
CONTROL_STEPS_PER_RECORD = CONTROLLER_HZ // RECORD_HZ
HOLD_RECORD_STEPS = 15
TERMINAL_PUSH_DISTANCE = 0.03
OPEN_DRAWER_QPOS = -0.159
CLOSED_DRAWER_QPOS = -0.01
MAX_POSITION_DELTA = 0.05
MAX_JOINT_DELTA = 0.08
IK_DAMPING = 0.04
TRACKING_GAIN = 0.6

ARM_ACTUATORS = (
    "joint_1_base_to_link_1_control",
    "joint_2_link_1_to_link_2_control",
    "joint_3_link_2_to_link_3_control",
    "joint_4_link_3_to_link_4_control",
    "joint_5_link_4_to_link_5_control",
)


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise RuntimeError(f"Missing MuJoCo object: {name}")
    return value


def load_teacher() -> np.ndarray:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    with h5py.File(SOURCE, "r") as source:
        demo = source["data/demo_0"]
        return demo["obs/ee_pos"][()]


def arm_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actuator_ids = np.array(
        [object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_ACTUATORS],
        dtype=np.int32,
    )
    joint_ids = model.actuator_trnid[actuator_ids, 0]
    qpos = np.array([model.jnt_qposadr[joint] for joint in joint_ids], dtype=np.int32)
    dofs = np.array([model.jnt_dofadr[joint] for joint in joint_ids], dtype=np.int32)
    return actuator_ids, qpos, dofs


def site_position(data: mujoco.MjData, site_id: int) -> np.ndarray:
    return np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()


def joint_delta(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    arm_dofs: np.ndarray,
    position_delta: np.ndarray,
) -> np.ndarray:
    jacobian = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacobian, None, site_id)
    jacobian = jacobian[:, arm_dofs]
    system = jacobian.T @ jacobian + IK_DAMPING**2 * np.eye(5)
    delta = np.linalg.solve(system, jacobian.T @ position_delta)
    return np.clip(delta, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)


def spline_resample(positions: np.ndarray) -> np.ndarray:
    """Smooth 20 Hz Cartesian waypoints and evaluate them at 100 Hz."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 7:
        raise ValueError("Expected at least seven XYZ waypoints")

    smoothed = savgol_filter(positions, window_length=7, polyorder=3, axis=0)
    smoothed[0] = positions[0]
    smoothed[-1] = positions[-1]
    source_time = np.arange(len(positions), dtype=np.float64) / RECORD_HZ
    control_time = np.arange((len(positions) - 1) * CONTROL_STEPS_PER_RECORD + 1)
    control_time = control_time / CONTROLLER_HZ
    zero_velocity = np.zeros(3, dtype=np.float64)
    spline = CubicSpline(
        source_time,
        smoothed,
        bc_type=((1, zero_velocity), (1, zero_velocity)),
        axis=0,
    )
    return spline(control_time)


def terminal_push_targets(position: np.ndarray) -> list[np.ndarray]:
    """Finish the retarget with a smooth 3 cm push into the cabinet."""
    steps = HOLD_RECORD_STEPS * CONTROL_STEPS_PER_RECORD
    phase = np.arange(1, steps + 1, dtype=np.float64) / steps
    blend = 0.5 - 0.5 * np.cos(np.pi * phase)
    offset = np.zeros((steps, 3), dtype=np.float64)
    offset[:, 1] = -TERMINAL_PUSH_DISTANCE * blend
    return list(position + offset)


def advance_to_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    site_id: int,
    arm_actuators: np.ndarray,
    arm_qpos: np.ndarray,
    arm_dofs: np.ndarray,
) -> None:
    error = target - site_position(data, site_id)
    delta = joint_delta(
        model,
        data,
        site_id,
        arm_dofs,
        TRACKING_GAIN * error,
    )
    data.ctrl[arm_actuators] = np.clip(
        data.qpos[arm_qpos] + delta,
        model.actuator_ctrlrange[arm_actuators, 0],
        model.actuator_ctrlrange[arm_actuators, 1],
    )
    for _ in range(PHYSICS_STEPS_PER_CONTROL):
        mujoco.mj_step(model, data)


def solve_initial_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    arm_qpos: np.ndarray,
    arm_dofs: np.ndarray,
    target: np.ndarray,
) -> None:
    joint_ids = model.actuator_trnid[:5, 0]
    lower = model.jnt_range[joint_ids, 0]
    upper = model.jnt_range[joint_ids, 1]
    for _ in range(300):
        error = target - site_position(data, site_id)
        if np.linalg.norm(error) < 1e-4:
            return
        data.qpos[arm_qpos] = np.clip(
            data.qpos[arm_qpos]
            + joint_delta(model, data, site_id, arm_dofs, np.clip(error, -0.02, 0.02)),
            lower,
            upper,
        )
        mujoco.mj_forward(model, data)
    raise RuntimeError(
        f"Initial teacher waypoint is unreachable; error={np.linalg.norm(error):.4f} m"
    )


def render(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def robot_touches_bodies(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_bodies: set[int],
    robot_bodies: set[int],
) -> bool:
    for contact in data.contact:
        body_a = int(model.geom_bodyid[contact.geom1])
        body_b = int(model.geom_bodyid[contact.geom2])
        if (body_a in target_bodies and body_b in robot_bodies) or (
            body_b in target_bodies and body_a in robot_bodies
        ):
            return True
    return False


def write_dataset(records: dict[str, list[np.ndarray | float]], success: bool) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT, "w") as output:
        data = output.create_group("data")
        data.attrs["num_demos"] = 1
        data.attrs["total"] = len(records["actions"])
        data.attrs["env_args"] = json.dumps(
            {
                "robot": "NexArm",
                "scene": str(SCENE),
                "control_freq": RECORD_HZ,
                "controller_freq": CONTROLLER_HZ,
                "trajectory_resampling": "savgol_then_cubic_spline",
                "action": "normalized_ee_delta",
                "source": str(SOURCE),
            }
        )
        data.attrs["language_instruction"] = "close the top drawer of the cabinet"

        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = len(records["actions"])
        demo.attrs["source_demo"] = "demo_0"
        demo.attrs["success"] = success
        demo.create_dataset("actions", data=np.asarray(records["actions"], dtype=np.float32))
        demo.create_dataset("rewards", data=np.asarray(records["rewards"], dtype=np.uint8))
        demo.create_dataset("dones", data=np.asarray(records["dones"], dtype=np.uint8))
        demo.create_dataset(
            "obs/front_rgb",
            data=np.asarray(records["front_rgb"], dtype=np.uint8),
            compression="gzip",
        )
        demo.create_dataset(
            "obs/eye_in_hand_rgb",
            data=np.asarray(records["wrist_rgb"], dtype=np.uint8),
            compression="gzip",
        )
        demo.create_dataset("obs/joint_states", data=np.asarray(records["joint_states"]))
        demo.create_dataset("obs/ee_pos", data=np.asarray(records["ee_pos"]))
        demo.create_dataset("obs/ee_ori", data=np.asarray(records["ee_ori"]))
        demo.create_dataset("obs/drawer_qpos", data=np.asarray(records["drawer_qpos"]))
        demo.create_dataset(
            "robot_drawer_contact",
            data=np.asarray(records["robot_drawer_contact"], dtype=np.uint8),
        )


def main() -> None:
    teacher_positions = load_teacher()
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

    target_final = np.array([0.533, -0.21, 0.274], dtype=np.float64)
    scene_rotation = np.diag([-1.0, -1.0, 1.0])
    targets = (scene_rotation @ teacher_positions.T).T
    targets += target_final - targets[-1]
    control_targets = list(spline_resample(targets)[1:])
    control_targets += terminal_push_targets(targets[-1])

    mujoco.mj_resetData(model, data)
    data.qpos[drawer_qpos] = OPEN_DRAWER_QPOS
    data.qpos[gripper_qpos] = 0.0
    mujoco.mj_forward(model, data)
    solve_initial_pose(model, data, site_id, arm_qpos, arm_dofs, targets[0])
    data.ctrl[arm_actuators] = data.qpos[arm_qpos]
    data.ctrl[gripper_actuator] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    for _ in range(100):
        data.qpos[drawer_qpos] = OPEN_DRAWER_QPOS
        data.qvel[model.jnt_dofadr[drawer_joint]] = 0.0
        mujoco.mj_step(model, data)

    if robot_touches_bodies(model, data, cabinet_bodies, robot_bodies):
        raise RuntimeError("Robot overlaps cabinet in the initial pose")

    records: dict[str, list] = {
        "actions": [],
        "rewards": [],
        "dones": [],
        "front_rgb": [],
        "wrist_rgb": [],
        "joint_states": [],
        "ee_pos": [],
        "ee_ori": [],
        "drawer_qpos": [],
        "robot_drawer_contact": [],
    }
    video_frames: list[np.ndarray] = []
    record_steps = len(control_targets) // CONTROL_STEPS_PER_RECORD

    for step in range(record_steps):
        target_slice = control_targets[
            step * CONTROL_STEPS_PER_RECORD : (step + 1) * CONTROL_STEPS_PER_RECORD
        ]
        current_position = site_position(data, site_id)
        normalized_delta = np.clip(
            (target_slice[-1] - current_position) / MAX_POSITION_DELTA,
            -1.0,
            1.0,
        )
        action = np.concatenate((normalized_delta, np.zeros(3), [-1.0])).astype(np.float32)

        front = render(renderer, data, "front")
        wrist = render(renderer, data, "wrist")
        site_matrix = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
        current_drawer_qpos = float(data.qpos[drawer_qpos])

        records["actions"].append(action)
        records["front_rgb"].append(front)
        records["wrist_rgb"].append(wrist)
        records["joint_states"].append(
            np.concatenate((data.qpos[arm_qpos], [data.qpos[gripper_qpos]])).copy()
        )
        records["ee_pos"].append(current_position)
        records["ee_ori"].append(Rotation.from_matrix(site_matrix).as_rotvec())
        records["drawer_qpos"].append(current_drawer_qpos)
        video_frames.append(front)

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

        closed = float(data.qpos[drawer_qpos]) >= CLOSED_DRAWER_QPOS
        records["robot_drawer_contact"].append(contact)
        records["rewards"].append(float(closed))
        records["dones"].append(step == record_steps - 1)

    final_drawer_qpos = float(data.qpos[drawer_qpos])
    success = final_drawer_qpos >= CLOSED_DRAWER_QPOS
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(VIDEO, np.asarray(video_frames), fps=RECORD_HZ)
    if not success:
        raise RuntimeError(
            f"Retarget failed: drawer qpos {final_drawer_qpos:.4f} < "
            f"{CLOSED_DRAWER_QPOS:.4f}; ee={site_position(data, site_id).round(4)}"
        )

    write_dataset(records, success)
    renderer.close()

    print(f"source_steps={len(teacher_positions)} recorded_steps={len(records['actions'])}")
    print(f"drawer_qpos={OPEN_DRAWER_QPOS:.4f}->{final_drawer_qpos:.4f} success={success}")
    print(f"robot_drawer_contact_steps={sum(records['robot_drawer_contact'])}")
    print(f"dataset={OUTPUT}")
    print(f"video={VIDEO}")


if __name__ == "__main__":
    main()
