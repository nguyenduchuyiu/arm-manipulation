"""Four-phase Gymnasium protocol for the memory/occlusion MVP."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from PIL import Image

from controllers.nexarm_mujoco_backend import JOINT_NAMES, MUJOCO_JOINTS


SCENE_XML = Path(__file__).with_name("scene.xml")
TARGETS = ("Butter", "Popcorn", "Milk", "Tuna")
SLOTS = ((0.45, -0.27), (0.45, -0.21), (0.65, -0.27), (0.65, -0.21))
TARGET_HALF_HEIGHT = {"Butter": 0.010565, "Popcorn": 0.014870,
                      "Milk": 0.028553, "Tuna": 0.006494}
COVER_XY = {"cover_a": (0.45, -0.24), "cover_b": (0.65, -0.24)}
PHASES = ("reveal", "occlude", "query", "execute")
SHOULDER_LIFT_FORCE_LIMIT_NM = 5.0
RESOLUTION = 480
REFERENCE_SIZE = 96
REVEAL_FRAMES = 75
COVER_FRAMES = 150
HOLD_FRAMES = 50
COVER_DROP_ZONE_XY = (0.30, -0.04)
TARGET_DROP_ZONE_XY = (0.79, -0.04)
WORKSPACE_LOWER = (0.34, -0.34, -0.02)
WORKSPACE_UPPER = (0.74, -0.04, 0.45)
TARGET_WORKSPACE_UPPER = (0.95, 0.08, 0.45)


def configure_mvp_model(model: mujoco.MjModel) -> None:
    """Set the arm and contact solver for the tabletop grasp task."""
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.impratio = 10
    actuator = model.actuator("joint_2_link_1_to_link_2_control").id
    model.actuator_forcerange[actuator] = (-SHOULDER_LIFT_FORCE_LIMIT_NM,
                                          SHOULDER_LIFT_FORCE_LIMIT_NM)
    camera = model.camera("wrist")
    camera.pos[:] = (0.539, -0.07, 0.32)
    forward = np.array((0.0, -0.13, -0.30))
    forward /= np.linalg.norm(forward)
    right = np.array((1.0, 0.0, 0.0))
    up = np.cross(-forward, right)
    mujoco.mju_mat2Quat(camera.quat, np.column_stack((right, up, -forward)).flatten())


class MemoryOcclusionEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, max_episode_steps: int = 1500, context_fps: int = 50,
                 resolution: int = RESOLUTION) -> None:
        super().__init__()
        if context_fps not in (10, 25, 50):
            raise ValueError("context_fps must be 10, 25, or 50")
        if resolution < 160:
            raise ValueError("resolution must be at least 160")
        self.context_fps = context_fps
        self.record_stride = 50 // context_fps
        self.resolution = resolution
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        configure_mvp_model(self.model)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=resolution, width=resolution)
        # MSAA blends segmentation IDs on some EGL drivers; RGB keeps its original sampling.
        samples = self.model.vis.quality.offsamples
        self.model.vis.quality.offsamples = 0
        self.segmentation_renderer = mujoco.Renderer(self.model, height=resolution, width=resolution)
        self.model.vis.quality.offsamples = samples
        self.segmentation_renderer.enable_segmentation_rendering()
        self.action_space = spaces.Box(self.model.actuator_ctrlrange[:, 0].astype(np.float32),
                                       self.model.actuator_ctrlrange[:, 1].astype(np.float32),
                                       dtype=np.float32)
        self.observation_space = spaces.Dict({
            "overview_rgb": spaces.Box(0, 255, shape=(resolution, resolution, 3), dtype=np.uint8),
            "overview_depth_m": spaces.Box(0.0, np.inf, shape=(resolution, resolution), dtype=np.float32),
            "wrist_rgb": spaces.Box(0, 255, shape=(resolution, resolution, 3), dtype=np.uint8),
            "reference_rgb": spaces.Box(0, 255, shape=(REFERENCE_SIZE, REFERENCE_SIZE, 3), dtype=np.uint8),
            "robot_joint_positions": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "robot_joint_velocities": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        })
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
        self.robot_joint_names = JOINT_NAMES
        joint_ids = [self.model.joint(MUJOCO_JOINTS[name]).id for name in JOINT_NAMES]
        self.robot_qpos_addresses = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        self.robot_qvel_addresses = np.array([self.model.jnt_dofadr[j] for j in joint_ids])
        self.target_qadr = {
            name: self.model.jnt_qposadr[self.model.joint(f"{name}_joint").id]
            for name in TARGETS
        }
        self.cover_qadr = {
            name: self.model.jnt_qposadr[self.model.joint(f"{name}_joint").id]
            for name in COVER_XY
        }
        self.phase = "reveal"
        self.layout = "standard"
        self.assignment: dict[str, str] = {}
        self.reveal_rgb: np.ndarray | None = None
        self.reveal_depth_m: np.ndarray | None = None
        self.context_rgb: np.ndarray | None = None
        self.context_depth_m: np.ndarray | None = None
        self.reference_images: dict[str, np.ndarray] = {}
        self.query_target: str | None = None
        self.selected_cover: str | None = None
        self.failure_reason: str | None = None
        self.success = False
        self.grasp_hold_steps = 0
        self.target_grasped = False
        self.initial_positions: dict[str, np.ndarray] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        self.layout = (options or {}).get("layout", "standard")
        if self.layout not in ("standard", "new_layout"):
            raise ValueError("layout must be standard or new_layout")
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.elapsed_steps = 0
        self.phase = "reveal"
        self.query_target = None
        self.selected_cover = None
        self.failure_reason = None
        self.success = False
        self.grasp_hold_steps = 0
        self.target_grasped = False
        self.assignment = {}
        for slot, (name, (x, y)) in enumerate(zip(self.np_random.permutation(TARGETS), SLOTS)):
            name = str(name)
            self.assignment[name] = "cover_a" if x < 0.55 else "cover_b"
            if self.layout == "standard":
                x += self.np_random.uniform(-0.012, 0.012)
            else:
                x += (0.015 if slot % 2 == 0 else -0.015) + self.np_random.uniform(-0.005, 0.005)
            y += self.np_random.uniform(-0.004, 0.004)
            if name in ("Butter", "Popcorn"):
                yaw = self.np_random.choice((0.0, np.pi)) + self.np_random.uniform(-0.45, 0.45)
            else:
                yaw = self.np_random.uniform(-np.pi, np.pi)
            qadr = self.target_qadr[name]
            self.data.qpos[qadr:qadr + 3] = (x, y, TARGET_HALF_HEIGHT[name] + 0.002)
            self.data.qpos[qadr + 3:qadr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
            if (options or {}).get("upright_targets") and name != "Milk":
                rest = np.array((np.sqrt(.5), np.sqrt(.5), 0, 0) if name == "Tuna"
                                else (np.sqrt(.5), 0, np.sqrt(.5), 0))
                quaternion = np.empty(4)
                mujoco.mju_mulQuat(quaternion, self.data.qpos[qadr + 3:qadr + 7], rest)
                self.data.qpos[qadr + 3:qadr + 7] = quaternion
                self.data.qpos[qadr + 2] = {"Butter": .020660, "Popcorn": .022136,
                                          "Tuna": .014144}[name] + .002
        self._place_covers(visible=False)
        rgb_frames, depth_frames = [], []
        for frame in range(REVEAL_FRAMES):
            mujoco.mj_step(self.model, self.data, 10)
            if (frame + 1) % self.record_stride == 0:
                rgb, depth = self._overview()
                rgb_frames.append(rgb)
                depth_frames.append(depth)
        self.reveal_rgb = np.stack(rgb_frames)
        self.reveal_depth_m = np.stack(depth_frames)
        self.context_rgb = self.reveal_rgb
        self.context_depth_m = self.reveal_depth_m
        self.reference_images = self._make_references(self.reveal_rgb[-1])
        self.initial_positions = self.target_positions()
        return self.observe(), self._info()

    def _place_covers(self, *, visible: bool) -> None:
        for name, qadr in self.cover_qadr.items():
            if visible:
                x, y = COVER_XY[name]
                z = 0.002
            else:
                x = -2.0 if name == "cover_a" else 2.0
                y, z = 0.0, 0.15
            self.data.qpos[qadr:qadr + 3] = (x, y, z)
            self.data.qpos[qadr + 3:qadr + 7] = (1, 0, 0, 0)
            dadr = self.model.jnt_dofadr[self.model.joint(f"{name}_joint").id]
            self.data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def occlude(self) -> dict[str, np.ndarray]:
        if self.phase != "reveal":
            raise RuntimeError("occlude requires reveal phase")
        starts = {"cover_a": (-0.15, -0.24), "cover_b": (1.25, -0.24)}
        rgb_frames, depth_frames = [], []
        for frame in range(COVER_FRAMES):
            progress = (frame + 1) / COVER_FRAMES
            travel = min(progress / 0.6, 1.0)
            lower = max((progress - 0.6) / 0.4, 0.0)
            travel = travel * travel * (3 - 2 * travel)
            lower = lower * lower * (3 - 2 * lower)
            for name, qadr in self.cover_qadr.items():
                start_x, start_y = starts[name]
                end_x, end_y = COVER_XY[name]
                self.data.qpos[qadr:qadr + 3] = (
                    start_x + travel * (end_x - start_x),
                    start_y + travel * (end_y - start_y),
                    0.22 + lower * (0.002 - 0.22),
                )
            mujoco.mj_forward(self.model, self.data)
            if (frame + 1) % self.record_stride == 0:
                rgb, depth = self._overview()
                rgb_frames.append(rgb)
                depth_frames.append(depth)
        hold_frames = HOLD_FRAMES // self.record_stride
        self.context_rgb = np.concatenate((
            self.reveal_rgb, np.stack(rgb_frames),
            np.repeat(rgb_frames[-1][None], hold_frames, axis=0),
        ))
        self.context_depth_m = np.concatenate((
            self.reveal_depth_m, np.stack(depth_frames),
            np.repeat(depth_frames[-1][None], hold_frames, axis=0),
        ))
        reveal_frames = REVEAL_FRAMES // self.record_stride
        self.reveal_rgb = self.context_rgb[:reveal_frames]
        self.reveal_depth_m = self.context_depth_m[:reveal_frames]
        self.phase = "occlude"
        return self.observe()

    def query(self, target: str | None = None) -> dict[str, np.ndarray]:
        if self.phase not in ("occlude", "query"):
            raise RuntimeError("query requires completed occlusion")
        target = str(self.np_random.choice(TARGETS)) if target is None else target
        if target not in TARGETS:
            raise ValueError(f"unknown target: {target}")
        self.query_target = target
        self.phase = "query"
        return self.observe()

    def choose_cover(self, cover: str) -> bool:
        if self.phase != "query":
            raise RuntimeError("cover choice requires query phase")
        if cover not in COVER_XY:
            raise ValueError(f"unknown cover: {cover}")
        self.selected_cover = cover
        self.phase = "execute"
        if cover != self.assignment[self.query_target]:
            self.failure_reason = "wrong_cover"
        return self.failure_reason is None

    def _overview(self) -> tuple[np.ndarray, np.ndarray]:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera="overview")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera="overview")
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def segmentation(self) -> np.ndarray:
        self.segmentation_renderer.update_scene(self.data, camera="overview")
        return self.segmentation_renderer.render().copy()

    def _make_references(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        segmentation = self.segmentation()
        references = {}
        for name in TARGETS:
            mask = segmentation[:, :, 0] == self.model.geom(f"{name}_visual").id
            ys, xs = np.where(mask)
            if len(xs) == 0:
                raise RuntimeError(f"target {name} is not visible in reveal frame")
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            crop, crop_mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            side = max(x1 - x0, y1 - y0) + 8
            canvas = np.full((side, side, 3), 128, dtype=np.uint8)
            top = (side - (y1 - y0)) // 2
            left = (side - (x1 - x0)) // 2
            patch = canvas[top:top + y1 - y0, left:left + x1 - x0]
            patch[crop_mask] = crop[crop_mask]
            references[name] = np.asarray(
                Image.fromarray(canvas).resize((REFERENCE_SIZE, REFERENCE_SIZE), Image.Resampling.NEAREST)
            )
        return references

    def observe(self) -> dict[str, np.ndarray]:
        rgb, depth = self._overview()
        self.renderer.update_scene(self.data, camera="wrist")
        wrist = self.renderer.render().copy()
        reference = (np.zeros((REFERENCE_SIZE, REFERENCE_SIZE, 3), dtype=np.uint8)
                     if self.query_target is None else self.reference_images[self.query_target])
        return {"overview_rgb": rgb, "overview_depth_m": depth,
                "wrist_rgb": wrist, "reference_rgb": reference.copy(),
                "robot_joint_positions": self.data.qpos[self.robot_qpos_addresses].astype(np.float32).copy(),
                "robot_joint_velocities": self.data.qvel[self.robot_qvel_addresses].astype(np.float32).copy()}

    def step(self, action: np.ndarray
             ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        if self.phase != "execute":
            raise RuntimeError("step is available only after query and cover choice")
        if self.success or self.failure_reason is not None:
            return (self.observe(), float(self.success),
                    self.failure_reason != "timeout", self.failure_reason == "timeout",
                    self._info())
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("action must be six actuator positions within control limits")
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, round(.04 / self.model.opt.timestep))
        self.elapsed_steps += 1
        self._evaluate()
        terminated = self.success or self.failure_reason is not None
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        if truncated:
            self.failure_reason = "timeout"
        return self.observe(), float(self.success), terminated, truncated, self._info()

    def _cover_in_drop_zone(self) -> bool:
        if self.selected_cover is None:
            return False
        pos = self.data.xpos[self.model.body(self.selected_cover).id]
        return bool(np.all(np.abs(pos[:2] - COVER_DROP_ZONE_XY) < (0.10, 0.10))
                    and -0.01 < pos[2] < 0.18)

    def _target_in_drop_zone(self) -> bool:
        if self.query_target is None:
            return False
        pos = self.target_positions()[self.query_target]
        return bool(np.all(np.abs(pos[:2] - TARGET_DROP_ZONE_XY) < (0.055, 0.055))
                    and -0.01 < pos[2] < 0.08 and not self._target_gripped())

    def _target_gripped(self) -> bool:
        collision = self.model.geom(f"{self.query_target}_collision").id
        jaws = {self.model.geom("link_6_left_jaw_collision_0").id,
                self.model.geom("link_6_right_jaw_collision_0").id}
        return any({contact.geom1, contact.geom2} & jaws
                   and collision in (contact.geom1, contact.geom2)
                   for contact in self.data.contact)

    def _wrong_object_gripped(self) -> bool:
        if self.data.ctrl[5] > -0.02:
            return False
        jaws = {self.model.geom("link_6_left_jaw_collision_0").id,
                self.model.geom("link_6_right_jaw_collision_0").id}
        for name in TARGETS:
            if name == self.query_target:
                continue
            collision = self.model.geom(f"{name}_collision").id
            touched = {jaw for contact in self.data.contact
                       for jaw in jaws
                       if collision in (contact.geom1, contact.geom2)
                       and jaw in (contact.geom1, contact.geom2)}
            if touched == jaws:
                return True
        return False

    def _evaluate(self) -> None:
        positions = self.target_positions()
        lower, upper = np.asarray(WORKSPACE_LOWER), np.asarray(WORKSPACE_UPPER)
        if self._wrong_object_gripped():
            self.failure_reason = "wrong_object_grasped"
            return
        for name, pos in positions.items():
            allowed_upper = TARGET_WORKSPACE_UPPER if name == self.query_target else upper
            if np.any(pos < lower) or np.any(pos > allowed_upper):
                self.failure_reason = "target_out_of_workspace"
                return
            if name != self.query_target and pos[2] > self.initial_positions[name][2] + 0.04:
                self.failure_reason = "wrong_object_grasped"
                return
        target_lifted = positions[self.query_target][2] > self.initial_positions[self.query_target][2] + 0.04
        if self._cover_in_drop_zone() and target_lifted and self._target_gripped():
            self.grasp_hold_steps += 1
        else:
            self.grasp_hold_steps = 0
        if self.grasp_hold_steps >= 10:
            self.target_grasped = True
        self.success = self.target_grasped and self._target_in_drop_zone()

    def _info(self) -> dict:
        return {"phase": self.phase, "elapsed_steps": self.elapsed_steps,
                "selected_cover": self.selected_cover,
                "cover_removed": self._cover_in_drop_zone(),
                "target_grasped": self.target_grasped,
                "target_placed": self._target_in_drop_zone() if self.query_target else False,
                "success": self.success, "failure_reason": self.failure_reason}

    def render(self) -> np.ndarray:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render().copy()

    def target_positions(self) -> dict[str, np.ndarray]:
        return {name: self.data.qpos[qadr:qadr + 3].copy()
                for name, qadr in self.target_qadr.items()}

    def close(self) -> None:
        self.segmentation_renderer.close()
        self.renderer.close()
