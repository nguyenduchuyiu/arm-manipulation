from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class NexArmEnv(gym.Env):
    """Minimal MuJoCo environment for VLA data collection.

    Action
    ------
    Absolute position targets for six actuators:
      joint_1 ... joint_5, left jaw.

    Observation
    -----------
    observation.state:
      Six actuated joint positions.

    observation.images.front:
      Static external RGB camera.

    observation.images.wrist:
      RGB camera attached to cam_mount.

    Reward remains zero because this first version targets imitation/VLA data
    collection rather than reinforcement learning.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
    }

    EXPECTED_ACTUATORS = (
        "joint_1_position",
        "joint_2_position",
        "joint_3_position",
        "joint_4_position",
        "joint_5_position",
        "gripper_position",
    )

    CAMERA_NAMES = ("front", "wrist")

    def __init__(
        self,
        scene_path: str | Path = "assets/robot/scene.xml",
        image_height: int = 384,
        image_width: int = 384,
        frame_skip: int = 10,
        max_episode_steps: int = 500,
        settle_steps: int = 200,
        randomize_object: bool = True,
        object_xy_noise: float = 0.025,
        render_mode: str | None = "rgb_array",
    ) -> None:
        super().__init__()

        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")

        self.scene_path = Path(scene_path).expanduser().resolve()
        if not self.scene_path.exists():
            raise FileNotFoundError(self.scene_path)

        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.settle_steps = int(settle_steps)
        self.randomize_object = bool(randomize_object)
        self.object_xy_noise = float(object_xy_noise)
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        self._validate_model()

        self.actuator_names = tuple(
            self._name(mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            for actuator_id in range(self.model.nu)
        )

        if self.actuator_names != self.EXPECTED_ACTUATORS:
            raise RuntimeError(
                "Unexpected actuator order.\n"
                f"Expected: {self.EXPECTED_ACTUATORS}\n"
                f"Got:      {self.actuator_names}"
            )

        self.actuated_joint_ids = self.model.actuator_trnid[:, 0].astype(
            np.int32
        )

        self.actuated_qpos_addresses = np.array(
            [
                self.model.jnt_qposadr[joint_id]
                for joint_id in self.actuated_joint_ids
            ],
            dtype=np.int32,
        )

        self.actuated_qvel_addresses = np.array(
            [
                self.model.jnt_dofadr[joint_id]
                for joint_id in self.actuated_joint_ids
            ],
            dtype=np.int32,
        )

        self.right_jaw_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "right_jaw_slide_joint",
        )
        self.right_jaw_qpos_address = int(
            self.model.jnt_qposadr[self.right_jaw_id]
        )

        self.object_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "object",
        )
        self.object_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "object_freejoint",
        )
        self.object_qpos_address = int(
            self.model.jnt_qposadr[self.object_joint_id]
        )

        self.grasp_site_id = self._required_id(
            mujoco.mjtObj.mjOBJ_SITE,
            "grasp_site",
        )
        self.gripper_base_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "link_6_gripper_base",
        )

        # Kinematic snap-grasp (the standard pick-lift model used by Robosuite /
        # MetaWorld / sim-engine): when the jaw closes within SNAP_DIST of the
        # object, weld the object to the gripper base frame so it follows the TCP;
        # opening the jaw releases it. Real pinch contact cannot hold a 3.6 cm
        # cube through a reorienting lift, so this is the one clear grasp path.
        self.grip_close_threshold = 0.0127   # cmd 0=closed, 0.0255=open
        self.snap_distance = 0.06
        self._grasped = False
        self._grasp_local_offset = None

        for camera_name in self.CAMERA_NAMES:
            self._required_id(
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )

        action_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        action_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(
                    low=action_low,
                    high=action_high,
                    shape=(6,),
                    dtype=np.float32,
                ),
                "observation.images.front": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.image_height, self.image_width, 3),
                    dtype=np.uint8,
                ),
                "observation.images.wrist": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.image_height, self.image_width, 3),
                    dtype=np.uint8,
                ),
            }
        )

        self.renderer = mujoco.Renderer(
            self.model,
            height=self.image_height,
            width=self.image_width,
        )

        self.home_action = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.015],
            dtype=np.float64,
        )

        self.object_default_position = np.array(
            [0.539, -0.18, 0.025],
            dtype=np.float64,
        )

        # --- Task definition: "Pick up the red cube" ---------------------
        # Success: cube held above LIFT_HEIGHT for SUCCESS_HOLD_STEPS control steps.
        self.lift_height = 0.08
        self.success_hold_steps = 10

        # Failure: cube leaves the workspace box or robot enters a dangerous pose.
        self.workspace_lower = np.array([-0.40, -0.60, -0.10], dtype=np.float64)
        self.workspace_upper = np.array([0.90, 0.60, 0.50], dtype=np.float64)
        self.danger_qvel = 20.0  # rad/s per actuated joint

        self._elapsed_steps = 0
        self._lift_steps = 0
        self._terminated_reason: str | None = None

    @property
    def simulation_dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def control_dt(self) -> float:
        return self.simulation_dt * self.frame_skip

    def _name(self, object_type: mujoco.mjtObj, object_id: int) -> str:
        name = mujoco.mj_id2name(self.model, object_type, object_id)
        return name or f"<unnamed:{object_id}>"

    def _required_id(
        self,
        object_type: mujoco.mjtObj,
        name: str,
    ) -> int:
        object_id = int(
            mujoco.mj_name2id(self.model, object_type, name)
        )
        if object_id < 0:
            raise RuntimeError(f"Missing MuJoCo object: {name}")
        return object_id

    def _validate_model(self) -> None:
        if self.model.nu != 6:
            raise RuntimeError(
                f"Expected 6 actuators, found {self.model.nu}"
            )

        if self.model.nq < 14:
            raise RuntimeError(
                "scene.xml should contain the 7-DoF robot state "
                "and a free object joint."
            )

    def _robot_qpos(self) -> np.ndarray:
        return np.asarray(
            self.data.qpos[self.actuated_qpos_addresses],
            dtype=np.float32,
        ).copy()

    def _robot_qvel(self) -> np.ndarray:
        return np.asarray(
            self.data.qvel[self.actuated_qvel_addresses],
            dtype=np.float32,
        ).copy()

    def _object_position(self) -> np.ndarray:
        return np.asarray(
            self.data.xpos[self.object_body_id],
            dtype=np.float32,
        ).copy()

    def _grasp_position(self) -> np.ndarray:
        return np.asarray(
            self.data.site_xpos[self.grasp_site_id],
            dtype=np.float32,
        ).copy()

    def _render_camera(self, camera_name: str) -> np.ndarray:
        self.renderer.update_scene(
            self.data,
            camera=camera_name,
        )
        return np.asarray(
            self.renderer.render(),
            dtype=np.uint8,
        ).copy()

    def _get_observation(self) -> dict[str, np.ndarray]:
        return {
            "observation.state": self._robot_qpos(),
            "observation.images.front": self._render_camera("front"),
            "observation.images.wrist": self._render_camera("wrist"),
        }

    def _get_info(self) -> dict[str, Any]:
        object_position = self._object_position()
        grasp_position = self._grasp_position()
        return {
            "sim_time": float(self.data.time),
            "elapsed_steps": self._elapsed_steps,
            "simulation_dt": self.simulation_dt,
            "control_dt": self.control_dt,
            "actuator_names": self.actuator_names,
            "robot_qvel": self._robot_qvel(),
            "object_position": object_position,
            "grasp_position": grasp_position,
            "object_to_grasp_distance": float(
                np.linalg.norm(object_position - grasp_position)
            ),
            "object_z": float(object_position[2]),
            "lift_steps": self._lift_steps,
            "success": self._terminated_reason == "success",
            "terminated_reason": self._terminated_reason,
            "ncon": int(self.data.ncon),
        }

    def _reset_object(self) -> None:
        position = self.object_default_position.copy()

        if self.randomize_object:
            position[:2] += self.np_random.uniform(
                low=-self.object_xy_noise,
                high=self.object_xy_noise,
                size=2,
            )

        qadr = self.object_qpos_address
        self.data.qpos[qadr : qadr + 3] = position
        self.data.qpos[qadr + 3 : qadr + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self.actuated_qpos_addresses] = self.home_action
        self.data.qpos[self.right_jaw_qpos_address] = -self.home_action[-1]
        self.data.ctrl[:] = self.home_action

        self._reset_object()

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        for _ in range(self.settle_steps):
            self.data.ctrl[:] = self.home_action
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps = 0
        self._lift_steps = 0
        self._terminated_reason = None
        self._grasped = False
        self._grasp_local_offset = None

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        dict[str, np.ndarray],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        action = np.asarray(action, dtype=np.float32)

        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, "
                f"got {action.shape}"
            )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        self.data.ctrl[:] = action

        gripper_closed = float(action[5]) < self.grip_close_threshold
        obj_pos = self._object_position()
        grasp_pos = self._grasp_position()
        dist = float(np.linalg.norm(obj_pos - grasp_pos))
        if gripper_closed and not self._grasped and dist < self.snap_distance:
            self._grasped = True
            gpos = np.asarray(self.data.xpos[self.gripper_base_body_id], dtype=np.float64)
            gmat = np.asarray(self.data.xmat[self.gripper_base_body_id], dtype=np.float64).reshape(3, 3)
            self._grasp_local_offset = gmat.T @ (obj_pos.astype(np.float64) - gpos)
        elif not gripper_closed and self._grasped:
            self._grasped = False
            self._grasp_local_offset = None

        def _weld_object() -> None:
            gpos = np.asarray(self.data.xpos[self.gripper_base_body_id], dtype=np.float64)
            gmat = np.asarray(self.data.xmat[self.gripper_base_body_id], dtype=np.float64).reshape(3, 3)
            world = gpos + gmat @ self._grasp_local_offset
            qadr = self.object_qpos_address
            self.data.qpos[qadr : qadr + 3] = world
            self.data.qvel[qadr : qadr + 3] = 0.0
            self.data.qvel[qadr + 3 : qadr + 6] = 0.0

        if self._grasped:
            _weld_object()

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self._grasped:
                _weld_object()

        self._elapsed_steps += 1

        if not np.all(np.isfinite(self.data.qpos)):
            raise FloatingPointError("qpos became non-finite")

        if not np.all(np.isfinite(self.data.qvel)):
            raise FloatingPointError("qvel became non-finite")

        observation = self._get_observation()

        object_position = self._object_position()
        object_z = float(object_position[2])

        # Success: cube held above the lift height for SUCCESS_HOLD_STEPS steps.
        if object_z > self.lift_height:
            self._lift_steps += 1
        else:
            self._lift_steps = 0

        if self._lift_steps >= self.success_hold_steps:
            self._terminated_reason = "success"

        # Failure: cube leaves the workspace or robot becomes dangerous.
        if self._terminated_reason is None:
            if np.any(object_position < self.workspace_lower) or np.any(
                object_position > self.workspace_upper
            ):
                self._terminated_reason = "object_out_of_workspace"
            elif np.any(np.abs(self._robot_qvel()) > self.danger_qvel):
                self._terminated_reason = "dangerous_pose"

        truncated = self._elapsed_steps >= self.max_episode_steps
        if truncated and self._terminated_reason is None:
            self._terminated_reason = "timeout"

        terminated = self._terminated_reason in {
            "success",
            "object_out_of_workspace",
            "dangerous_pose",
        }

        reward = 1.0 if self._terminated_reason == "success" else 0.0

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._render_camera("front")

    def close(self) -> None:
        self.renderer.close()
