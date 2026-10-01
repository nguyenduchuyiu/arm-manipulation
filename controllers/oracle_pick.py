"""Reusable MuJoCo oracle: position IK, close gripper, lift through contact."""
from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .nexarm_mujoco_backend import MUJOCO_JOINTS


ARM_NAMES = tuple(MUJOCO_JOINTS)[:5]
TCP_LOCAL = np.array((0.539369, -0.039412, 0.230434))


def pick(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    grasp_point: np.ndarray,
    *,
    wrist: tuple[float, float] = (-0.8, 0.0),
    on_step: Callable[[str], None] | None = None,
    place_xy: tuple[float, float] | None = None,
    control_hz: int = 25,
    lift_height: float = .12,
    straight_lift: bool = False,
) -> float:
    """Pick and optionally place one body; return its held height gain in metres.

    ``grasp_point`` is the desired object contact point in world coordinates.
    The callback sees the pre-step observation and command applied for the next
    1/control_hz seconds. Motion uses contact, without attaching the object.
    """
    body_id = model.body(body_name).id
    gripper_id = model.body("link_6_gripper_base").id
    addrs = np.array([model.jnt_qposadr[model.joint(MUJOCO_JOINTS[n]).id]
                      for n in ARM_NAMES])
    dofs = np.array([model.jnt_dofadr[model.joint(MUJOCO_JOINTS[n]).id]
                     for n in ARM_NAMES])
    low, high = model.actuator_ctrlrange[:5].T
    q0 = data.qpos[addrs].copy()
    start_z = float(data.xpos[body_id, 2])
    clone = mujoco.MjData(model)
    wrist = np.asarray(wrist, dtype=np.float64)

    def fk(q3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clone.qpos[:] = data.qpos
        clone.qpos[addrs] = np.r_[q3, wrist]
        mujoco.mj_forward(model, clone)
        rotation = clone.xmat[gripper_id].reshape(3, 3).copy()
        return clone.xpos[gripper_id].copy() + rotation @ TCP_LOCAL, rotation

    def ik(point: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = least_squares(lambda q: 100 * (fk(q)[0] - point), initial[:3],
                               bounds=(low[:3], high[:3]), max_nfev=400)
        tcp, rotation = fk(result.x)
        if np.linalg.norm(tcp - point) > 0.008:
            raise RuntimeError(f"Oracle IK cannot reach {np.round(point, 3)}")
        return np.r_[result.x, wrist], rotation

    point = np.asarray(grasp_point, dtype=np.float64)
    _, rotation = ik(point, q0)
    jaw_axis = rotation[:, 1]
    grasp = point + 0.018 * jaw_axis
    approach = grasp + 0.07 * jaw_axis
    approach[2] = max(approach[2], 0.12)
    lift = grasp + (0.0, 0.0, lift_height)
    q_approach, _ = ik(approach, q0)
    q_grasp, _ = ik(grasp, q_approach)
    q_lift, _ = ik(lift, q_grasp)

    held_steps = 0
    jaw_geoms = {model.geom(name).id for name in
                 ("link_6_left_jaw_collision_0", "link_6_right_jaw_collision_0")}
    substeps = round(1 / control_hz / model.opt.timestep)
    if not np.isclose(substeps * model.opt.timestep, 1 / control_hz):
        raise ValueError("control period must be a multiple of the physics timestep")

    def move(stage, target, gripper, seconds):
        nonlocal held_steps
        start = data.qpos[addrs].copy()
        steps = round(seconds * control_hz)
        for step in range(steps):
            t = (step + 1) / steps
            desired = start + t * t * (3 - 2 * t) * (target - start)
            compensation = data.qfrc_bias[dofs] / model.actuator_gainprm[:5, 0]
            command = np.clip(desired + compensation, low, high)
            data.ctrl[:5] = command
            data.ctrl[5] = gripper
            if on_step is not None:
                on_step(stage)
            mujoco.mj_step(model, data, substeps)
            if stage in ("lift", "hold") and data.xpos[body_id, 2] - start_z > .04:
                touching = any(
                    (contact.geom1 in jaw_geoms and model.geom_bodyid[contact.geom2] == body_id)
                    or (contact.geom2 in jaw_geoms and model.geom_bodyid[contact.geom1] == body_id)
                    for contact in data.contact
                )
                held_steps += int(touching)

    def vertical_lift(seconds):
        rotation = data.xmat[gripper_id].reshape(3, 3).copy()
        start = data.xpos[gripper_id] + rotation @ TCP_LOCAL
        q = data.qpos[addrs].copy()
        for t in np.linspace(0, 1, 11)[1:]:
            destination = start + (0, 0, lift_height * t)

            def residual(values):
                clone.qpos[:] = data.qpos
                clone.qpos[addrs] = values
                mujoco.mj_forward(model, clone)
                r = clone.xmat[gripper_id].reshape(3, 3)
                tcp = clone.xpos[gripper_id] + r @ TCP_LOCAL
                return np.r_[100 * (tcp - destination), 5 * (r - rotation).ravel()]

            result = least_squares(residual, q, bounds=(low, high), max_nfev=400)
            if np.linalg.norm(residual(result.x)[:3]) > .8:
                raise RuntimeError("cannot lift vertically with the current grip orientation")
            q = result.x
            move("lift", q, -0.0255, seconds / 10)
        return q

    for stage, target, gripper, seconds in (
        ("approach", q_approach, 0.0, 3.0),
        ("engage", q_grasp, 0.0, 2.4),
        ("close", q_grasp, -0.0255, 1.6),
        ("lift", q_lift, -0.0255, 3.6),
        ("hold", q_lift, -0.0255, 1.6),
    ):
        if straight_lift and stage == "lift":
            q_lift = vertical_lift(seconds)
        elif stage == "hold":
            move(stage, q_lift, gripper, seconds)
        else:
            move(stage, target, gripper, seconds)
    gain = float(data.xpos[body_id, 2] - start_z)
    if place_xy is not None:
        if gain < 0.04 or held_steps < round(.2 * control_hz):
            raise RuntimeError(f"{body_name}: failed to lift ({gain:.3f} m)")
        tcp = data.xpos[gripper_id] + data.xmat[gripper_id].reshape(3, 3) @ TCP_LOCAL
        offset = tcp - data.xpos[body_id]
        destination = np.array((*place_xy, start_z + 0.003)) + offset
        above = destination + (0, 0, lift_height)
        q_above, _ = ik(above, q_lift)
        q_place, _ = ik(destination, q_above)
        move("transport", q_above, -0.0255, 4.0)
        move("lower", q_place, -0.0255, 3.0)
        move("release", q_place, 0.0, 1.5)
        move("retreat", q_above, 0.0, 2.0)
        move("home", q0, 0.0, 3.0)
    return gain
