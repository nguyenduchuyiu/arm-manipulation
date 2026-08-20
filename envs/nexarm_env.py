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
    Normalized absolute position targets in [-1, 1] for six actuators:
      joint_1 ... joint_5, right jaw. For the gripper, +1 is fully open
      and -1 is fully closed.

    Observation
    -----------
    observation.state:
      Six actuated joint positions.

    observation.images.front:
      Static external RGB camera.

    observation.images.wrist:
      RGB camera attached to cam_mount.

    Reward is one on task success and zero otherwise.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
    }

    EXPECTED_ACTUATORS = (
        "joint_1_base_to_link_1_control",
        "joint_2_link_1_to_link_2_control",
        "joint_3_link_2_to_link_3_control",
        "joint_4_link_3_to_link_4_control",
        "joint_5_link_4_to_link_5_control",
        "gripper_control",
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

        self.object_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "cube",
        )
        self.object_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "cube_joint",
        )
        self.object_qpos_address = int(
            self.model.jnt_qposadr[self.object_joint_id]
        )

        self.grasp_site_id = self._required_id(
            mujoco.mjtObj.mjOBJ_SITE,
            "grasp_site",
        )
        for camera_name in self.CAMERA_NAMES:
            self._required_id(
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )

        self.control_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.control_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # MuJoCo joint limits are soft constraints, so qpos can briefly move
        # beyond the commanded actuator range during contact or fast motion.
        state_margin = np.maximum(
            np.float32(0.1) * (self.control_high - self.control_low),
            np.float32(0.005),
        )
        state_margin[-1] = np.float32(0.05)
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(
                    low=self.control_low - state_margin,
                    high=self.control_high + state_margin,
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

        self.home_control = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self.home_action = self.control_to_action(self.home_control)

        self.object_default_position = np.array(
            [0.539, -0.18, 0.016],
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

    def action_to_control(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {action.shape}")
        normalized = np.clip(action, -1.0, 1.0)
        return self.control_low + np.float32(0.5) * (
            normalized + np.float32(1.0)
        ) * (self.control_high - self.control_low)

    def control_to_action(self, control: np.ndarray) -> np.ndarray:
        control = np.asarray(control, dtype=np.float32)
        if control.shape != (6,):
            raise ValueError(f"Expected control shape (6,), got {control.shape}")
        control = np.clip(control, self.control_low, self.control_high)
        return np.float32(2.0) * (
            control - self.control_low
        ) / (self.control_high - self.control_low) - np.float32(1.0)

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

        if self.model.nq < 15:
            raise RuntimeError(
                "scene.xml should contain the 8-DoF robot state "
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

        self.data.qpos[self.actuated_qpos_addresses] = self.home_control
        self.data.ctrl[:] = self.home_control

        self._reset_object()

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        for _ in range(self.settle_steps):
            self.data.ctrl[:] = self.home_control
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps = 0
        self._lift_steps = 0
        self._terminated_reason = None
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

        self.data.ctrl[:] = self.action_to_control(action)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

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
