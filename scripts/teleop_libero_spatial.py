"""Keyboard teleoperation for NexArm in an exact LIBERO-Spatial scene.

macOS:
    conda run -n mujoco-vla mjpython scripts/teleop_libero_spatial.py --task 1

Controls are printed at startup. The Cartesian controller runs at 100 Hz while
RGB observations and LIBERO-style EE-delta actions are recorded at 20 Hz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock

import h5py
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retarget_libero_drawer import arm_addresses, object_id, render, site_position
from retarget_libero_spatial import SOURCE, build_model, joint_path, remap_assets

CONTROL_HZ = 100
RECORD_HZ = 20
PHYSICS_STEPS = 5
RECORD_EVERY = CONTROL_HZ // RECORD_HZ
MAX_POSITION_DELTA = 0.05
MAX_ROTATION_DELTA = 0.5
IK_DAMPING = 0.04
ORIENTATION_WEIGHT = 0.2
MAX_JOINT_DELTA = 0.025
TRACKING_GAIN = 2.0
COMMAND_SMOOTHING = 0.25
VELOCITY_TIME_CONSTANT = 0.06


@dataclass
class TeleopState:
    target_pos: np.ndarray
    target_rot: np.ndarray
    arm_command: np.ndarray
    camera_basis: np.ndarray
    linear_speed: float
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pressed: set = field(default_factory=set)
    gripper_open: bool = True
    recording: bool = False
    success: bool = False
    discard: bool = False
    quit: bool = False
    lock: Lock = field(default_factory=Lock)


class Recorder:
    def __init__(self, model, data, renderer, arm_qpos, grip_qpos, site):
        self.model = model
        self.data = data
        self.renderer = renderer
        self.arm_qpos = arm_qpos
        self.grip_qpos = grip_qpos
        self.site = site
        self.frames = []

    def sample(self, gripper_open):
        rotation = np.asarray(self.data.site_xmat[self.site]).reshape(3, 3).copy()
        self.frames.append(
            {
                "front": render(self.renderer, self.data, "agentview"),
                "wrist": render(self.renderer, self.data, "wrist"),
                "joint": np.r_[
                    self.data.qpos[self.arm_qpos], self.data.qpos[self.grip_qpos]
                ].copy(),
                "ee_pos": site_position(self.data, self.site),
                "ee_rot": rotation,
                "state": np.r_[
                    self.data.time, self.data.qpos, self.data.qvel, self.data.act
                ].copy(),
                "gripper": 1.0 if gripper_open else -1.0,
            }
        )

    def write(self, output, source_path, demo_name, problem_info, success):
        if len(self.frames) < 2:
            return False
        output.parent.mkdir(parents=True, exist_ok=True)
        positions = np.asarray([frame["ee_pos"] for frame in self.frames])
        rotations = np.asarray([frame["ee_rot"] for frame in self.frames])
        actions = np.zeros((len(self.frames) - 1, 7), dtype=np.float32)
        actions[:, :3] = np.clip(
            np.diff(positions, axis=0) / MAX_POSITION_DELTA, -1.0, 1.0
        )
        for index in range(len(actions)):
            delta = rotations[index + 1] @ rotations[index].T
            actions[index, 3:6] = np.clip(
                Rotation.from_matrix(delta).as_rotvec() / MAX_ROTATION_DELTA,
                -1.0,
                1.0,
            )
        actions[:, 6] = [frame["gripper"] for frame in self.frames[:-1]]

        with h5py.File(output, "w") as file:
            data = file.create_group("data")
            data.attrs.update(
                num_demos=1,
                total=len(actions),
                successes=int(success),
                problem_info=problem_info,
                env_args=json.dumps(
                    {
                        "robot": "NexArm",
                        "action": "normalized_ee_delta",
                        "control_freq": RECORD_HZ,
                        "controller_freq": CONTROL_HZ,
                        "source": str(source_path),
                    }
                ),
            )
            demo = data.create_group("demo_0")
            demo.attrs.update(
                num_samples=len(actions),
                source_demo=demo_name,
                teleop=True,
                success=success,
            )
            demo.create_dataset("actions", data=actions)
            demo.create_dataset(
                "states", data=np.asarray([frame["state"] for frame in self.frames[:-1]])
            )
            observations = demo.create_group("obs")
            observations.create_dataset(
                "front_rgb",
                data=np.asarray([frame["front"] for frame in self.frames[:-1]]),
                compression="gzip",
            )
            observations.create_dataset(
                "eye_in_hand_rgb",
                data=np.asarray([frame["wrist"] for frame in self.frames[:-1]]),
                compression="gzip",
            )
            observations.create_dataset(
                "joint_states",
                data=np.asarray([frame["joint"] for frame in self.frames[:-1]]),
            )
            observations.create_dataset("ee_pos", data=positions[:-1])
            observations.create_dataset(
                "ee_ori",
                data=np.asarray(
                    [Rotation.from_matrix(matrix).as_rotvec() for matrix in rotations[:-1]]
                ),
            )
            done = np.zeros(len(actions), dtype=np.uint8)
            done[-1] = 1
            demo.create_dataset("dones", data=done)
            rewards = np.zeros(len(actions), dtype=np.uint8)
            rewards[-1] = int(success)
            demo.create_dataset("rewards", data=rewards)
        return True


def resolve_task(value):
    paths = sorted(SOURCE.glob("*.hdf5"))
    if value.isdigit():
        index = int(value)
        if not 0 <= index < len(paths):
            raise ValueError(f"task index must be in [0, {len(paths) - 1}]")
        return paths[index]
    matches = [path for path in paths if value.lower() in path.stem.lower()]
    if len(matches) != 1:
        raise ValueError(f"task query matched {len(matches)} files: {value!r}")
    return matches[0]


def joint_width(model, joint):
    joint_type = model.jnt_type[joint]
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def initialize_scene(source, model, poses, data, arm_qpos, arm_dofs, site):
    original = mujoco.MjModel.from_xml_string(remap_assets(source.attrs["model_file"]))
    original_qpos = source["states"][0, 1 : 1 + original.nq]
    mujoco.mj_resetData(model, data)

    for joint in range(original.njnt):
        name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_JOINT, joint)
        target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if target < 0 or name in poses:
            continue
        source_address = original.jnt_qposadr[joint]
        target_address = model.jnt_qposadr[target]
        width = joint_width(original, joint)
        data.qpos[target_address : target_address + width] = original_qpos[
            source_address : source_address + width
        ]
    for name, pose in poses.items():
        joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        address = model.jnt_qposadr[joint]
        data.qpos[address : address + 7] = pose

    start = source["obs/ee_pos"][0]
    data.qpos[arm_qpos] = joint_path(model, arm_qpos, arm_dofs, site, [start])[0]
    data.ctrl[:5] = data.qpos[arm_qpos]
    data.ctrl[5] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(100):
        mujoco.mj_step(model, data)


def update_target(state, dt):
    from pynput import keyboard

    with state.lock:
        horizontal = float(keyboard.Key.right in state.pressed) - float(
            keyboard.Key.left in state.pressed
        )
        vertical = float(keyboard.Key.up in state.pressed) - float(
            keyboard.Key.down in state.pressed
        )
        right = state.camera_basis[:, 0]
        up = state.camera_basis[:, 1]
        forward = -state.camera_basis[:, 2]
        if keyboard.Key.space in state.pressed:
            direction = vertical * forward
        else:
            direction = horizontal * right + vertical * up
        norm = np.linalg.norm(direction)
        if norm > 1.0:
            direction /= norm

        desired_linear = state.linear_speed * direction
        alpha = 1.0 - np.exp(-dt / VELOCITY_TIME_CONSTANT)
        state.linear_velocity += alpha * (desired_linear - state.linear_velocity)
        state.target_pos += dt * state.linear_velocity


def controller_step(model, data, state, arm_act, arm_qpos, arm_dofs, grip_act, site):
    with state.lock:
        target_pos = state.target_pos.copy()
        target_rot = state.target_rot.copy()
        gripper_open = state.gripper_open

    jacobian_pos = np.zeros((3, model.nv))
    jacobian_rot = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacobian_pos, jacobian_rot, site)
    current_rot = np.asarray(data.site_xmat[site]).reshape(3, 3)
    position_error = np.clip(target_pos - site_position(data, site), -0.02, 0.02)
    rotation_error = Rotation.from_matrix(target_rot @ current_rot.T).as_rotvec()
    jacobian = np.vstack(
        (
            jacobian_pos[:, arm_dofs],
            ORIENTATION_WEIGHT * jacobian_rot[:, arm_dofs],
        )
    )
    residual = np.r_[
        TRACKING_GAIN * position_error,
        ORIENTATION_WEIGHT * rotation_error,
    ]
    delta = np.linalg.solve(
        jacobian.T @ jacobian + IK_DAMPING**2 * np.eye(5),
        jacobian.T @ residual,
    )
    desired_command = data.qpos[arm_qpos] + np.clip(
        delta, -MAX_JOINT_DELTA, MAX_JOINT_DELTA
    )
    with state.lock:
        state.arm_command += COMMAND_SMOOTHING * (
            desired_command - state.arm_command
        )
        command = state.arm_command.copy()
    data.ctrl[arm_act] = np.clip(
        command,
        model.actuator_ctrlrange[arm_act, 0],
        model.actuator_ctrlrange[arm_act, 1],
    )
    data.ctrl[grip_act] = 0.0 if gripper_open else -0.0255
    for _ in range(PHYSICS_STEPS):
        mujoco.mj_step(model, data)


def make_keyboard_listener(state):
    from pynput import keyboard

    arrow_keys = {
        keyboard.Key.up,
        keyboard.Key.down,
        keyboard.Key.left,
        keyboard.Key.right,
    }
    def normalize(key):
        if isinstance(key, keyboard.KeyCode) and key.char:
            return key.char.lower()
        return key

    def on_press(key, _injected=False):
        key = normalize(key)
        with state.lock:
            if key in state.pressed:
                return
            state.pressed.add(key)
            if key == "g":
                state.gripper_open = not state.gripper_open
                print("gripper:", "open" if state.gripper_open else "closed")
            elif key == keyboard.Key.enter:
                if not state.recording:
                    state.recording = True
                    print("recording started")
                else:
                    state.success = True
                    state.quit = True
                    print("recording stopped: success")
            elif key == keyboard.Key.esc:
                state.discard = True
                state.quit = True

    def on_release(key, _injected=False):
        with state.lock:
            state.pressed.discard(normalize(key))

    options = {}
    if sys.platform == "darwin":
        from Quartz import CGEventGetIntegerValueField, kCGKeyboardEventKeycode

        blocked = {
            *(key.value.vk for key in arrow_keys),
            keyboard.Key.space.value.vk,
            keyboard.Key.enter.value.vk,
            keyboard.Key.esc.value.vk,
            0x05,  # G on the macOS virtual-key layout
        }

        def intercept(_event_type, event):
            virtual_key = CGEventGetIntegerValueField(
                event, kCGKeyboardEventKeycode
            )
            return None if virtual_key in blocked else event

        options["darwin_intercept"] = intercept
    return keyboard.Listener(on_press=on_press, on_release=on_release, **options)


def default_output(task_path, demo_index):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = task_path.stem.removesuffix("_demo")
    return Path("data/nexarm_teleop") / f"{name}_source{demo_index}_{stamp}.hdf5"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="0", help="task index 0..9 or filename substring")
    parser.add_argument("--demo", type=int, default=0, help="source scene initialization")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--linear-speed", type=float, default=0.06, help="m/s")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--check", action="store_true", help="build and step without a GUI")
    return parser.parse_args()


def main():
    args = parse_args()
    task_path = resolve_task(args.task)
    output = args.output or default_output(task_path, args.demo)
    with h5py.File(task_path) as file:
        demo_name = f"demo_{args.demo}"
        source = file[f"data/{demo_name}"]
        problem_info = file["data"].attrs["problem_info"]
        model, poses = build_model(source)
        data = mujoco.MjData(model)
        arm_act, arm_qpos, arm_dofs = arm_addresses(model)
        grip_act = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_control")
        grip_joint = int(model.actuator_trnid[grip_act, 0])
        grip_qpos = int(model.jnt_qposadr[grip_joint])
        site = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        camera = object_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
        initialize_scene(source, model, poses, data, arm_qpos, arm_dofs, site)
        camera_basis = np.asarray(data.cam_xmat[camera]).reshape(3, 3).copy()

        if args.check:
            target = site_position(data, site) + np.array([0.01, 0.0, 0.0])
            state = TeleopState(
                target_pos=target,
                target_rot=np.asarray(data.site_xmat[site]).reshape(3, 3).copy(),
                arm_command=data.qpos[arm_qpos].copy(),
                camera_basis=camera_basis,
                linear_speed=args.linear_speed,
            )
            for _ in range(50):
                controller_step(
                    model,
                    data,
                    state,
                    arm_act,
                    arm_qpos,
                    arm_dofs,
                    grip_act,
                    site,
                )
            print("scene:", task_path.name, demo_name)
            print("TCP:", np.round(site_position(data, site), 4))
            print(
                "controller error:",
                f"{1000 * np.linalg.norm(target - site_position(data, site)):.2f} mm",
            )
            print("contacts:", data.ncon)
            return

        renderer = mujoco.Renderer(model, args.image_size, args.image_size)
        state = TeleopState(
            target_pos=site_position(data, site),
            target_rot=np.asarray(data.site_xmat[site]).reshape(3, 3).copy(),
            arm_command=data.qpos[arm_qpos].copy(),
            camera_basis=camera_basis,
            linear_speed=args.linear_speed,
        )
        recorder = Recorder(model, data, renderer, arm_qpos, grip_qpos, site)
        print("scene:", task_path.name, demo_name)
        print("hold arrows: move in the camera image plane")
        print("hold Space + Up/Down: move away from/toward the camera")
        print("G gripper | Enter start/stop+save | Esc discard")
        print("output:", output)

        tick = 0
        listener = make_keyboard_listener(state)
        listener.start()
        listener.wait()
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera
                while viewer.is_running() and not state.quit:
                    start = time.perf_counter()
                    update_target(state, 1.0 / CONTROL_HZ)
                    controller_step(
                        model,
                        data,
                        state,
                        arm_act,
                        arm_qpos,
                        arm_dofs,
                        grip_act,
                        site,
                    )
                    if state.recording and tick % RECORD_EVERY == 0:
                        recorder.sample(state.gripper_open)
                    viewer.sync()
                    tick += 1
                    remaining = 1.0 / CONTROL_HZ - (time.perf_counter() - start)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            listener.stop()

        renderer.close()
        if state.discard:
            print("episode discarded")
        elif recorder.write(
            output, task_path, demo_name, problem_info, state.success
        ):
            print(f"saved {len(recorder.frames) - 1} samples: {output}")
        else:
            print("no episode saved (press Enter to start recording)")


if __name__ == "__main__":
    main()
