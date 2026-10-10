"""Reusable MuJoCo oracle: position IK, close gripper, lift through contact."""
from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .nexarm_mujoco_backend import MUJOCO_JOINTS


ARM_NAMES = tuple(MUJOCO_JOINTS)[:5]
TCP_LOCAL = np.array((0.539369, -0.039412, 0.230434))


class GraspFailure(RuntimeError):
    """A completed grasp attempt did not securely lift its target."""


class IKFailure(RuntimeError):
    """The current physical geometry is unreachable by this position oracle."""


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
    Motion always uses joint-travel durations and gripper/contact feedback.
    Optional placement releases from the lifted height and waits for landing.
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
            raise IKFailure(f"Oracle IK cannot reach {np.round(point, 3)}")
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

    jaw_addresses = [model.jnt_qposadr[model.joint(f"{side}_jaw_slide_joint").id]
                     for side in ("left", "right")]

    def jaw_contacts():
        return {jaw for contact in data.contact for jaw in jaw_geoms
                if ((contact.geom1 == jaw and model.geom_bodyid[contact.geom2] == body_id)
                    or (contact.geom2 == jaw and model.geom_bodyid[contact.geom1] == body_id))}

    def ready(target):
        return (np.max(np.abs(data.qpos[addrs] - target)) < np.deg2rad(.4)
                and np.max(np.abs(data.qvel[dofs])) < .05)

    def motion_done(stage, target):
        if ready(target):
            return True
        if stage == "engage":
            # Contact can stop approach before IK. Closing and lifting still
            # require bilateral contact with the requested body.
            return (np.max(np.abs(data.qpos[addrs] - target)) < np.deg2rad(5)
                    and np.max(np.abs(data.qvel[dofs])) < .05)
        return False

    def step_command(stage, desired, gripper):
        nonlocal held_steps
        compensation = data.qfrc_bias[dofs] / model.actuator_gainprm[:5, 0]
        command = np.r_[np.clip(desired + compensation, low, high), gripper]
        if command.shape != (6,) or not np.isfinite(command).all():
            raise ValueError("invalid oracle command")
        data.ctrl[:] = command
        if on_step is not None:
            on_step(stage)
        mujoco.mj_step(model, data, substeps)
        if stage in ("lift", "hold") and data.xpos[body_id, 2] - start_z > .04:
            held_steps += int(bool(jaw_contacts()))

    def fast_motion(stage, path, gripper):
        lengths = np.abs(np.diff(path, axis=0)).max(axis=1)
        distances = np.r_[0., np.cumsum(lengths)]
        travel = distances[-1]
        # Move slowly near the grasp; smoothstep peaks at 1.5 times its average.
        speed_limit = .4 if stage == "engage" else .8
        steps = max(3, int(np.ceil(1.5 * travel / speed_limit * control_hz)))
        for index in range(steps):
            t = (index + 1) / steps
            position = t * t * (3 - 2 * t) * travel
            left = min(np.searchsorted(distances, position, side="right") - 1, len(path) - 2)
            fraction = (position - distances[left]) / max(lengths[left], 1e-12)
            desired = (1 - fraction) * path[left] + fraction * path[left + 1]
            step_command(stage, desired, gripper)
        for _ in range(round(.6 * control_hz)):
            if motion_done(stage, path[-1]):
                return
            step_command(stage, path[-1], gripper)
        if not motion_done(stage, path[-1]):
            error = np.rad2deg(np.max(np.abs(data.qpos[addrs] - path[-1])))
            speed = np.max(np.abs(data.qvel[dofs]))
            raise GraspFailure(f"{body_name}: {stage} did not reach its pose "
                               f"(error={error:.2f} deg, speed={speed:.3f} rad/s)")

    def move(stage, target, gripper):
        start = data.qpos[addrs].copy()
        if stage in ("close", "release"):
            consecutive = 0
            timeout = 1.6 if stage == "close" else .8
            for _ in range(round(timeout * control_hz)):
                step_command(stage, target, gripper)
                if stage == "close":
                    complete = (jaw_contacts() == jaw_geoms
                                and np.max(np.abs(data.qvel[dofs])) < .05)
                else:
                    left, right = data.qpos[jaw_addresses]
                    complete = left < .001 and right > -.001 and not jaw_contacts()
                consecutive = consecutive + 1 if complete else 0
                if consecutive >= 3:
                    return
            raise GraspFailure(f"{body_name}: {stage} did not complete")
        if stage == "hold":
            consecutive = 0
            for _ in range(round(.6 * control_hz)):
                step_command(stage, target, gripper)
                complete = data.xpos[body_id, 2] - start_z > .04 and jaw_contacts() == jaw_geoms
                consecutive = consecutive + 1 if complete else 0
                # Collection records pre-step states; eleven post-step checks
                # guarantee ten recorded bilateral-lift frames at the endpoint.
                if consecutive >= 11:
                    return
            raise GraspFailure(f"{body_name}: lift is not stable")
        fast_motion(stage, np.stack((start, target)), gripper)

    def vertical_lift():
        rotation = data.xmat[gripper_id].reshape(3, 3).copy()
        start = data.xpos[gripper_id] + rotation @ TCP_LOCAL
        q = data.qpos[addrs].copy()
        path = [q.copy()]
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
                raise IKFailure("cannot lift vertically with the current grip orientation")
            q = result.x
            path.append(q.copy())
        fast_motion("lift", np.stack(path), -.0255)
        return q

    for stage, target, gripper in (
        ("approach", q_approach, 0.0),
        ("engage", q_grasp, 0.0),
        ("close", q_grasp, -0.0255),
        ("lift", q_lift, -0.0255),
        ("hold", q_lift, -0.0255),
    ):
        if straight_lift and stage == "lift":
            q_lift = vertical_lift()
        elif stage == "hold":
            move(stage, q_lift, gripper)
        else:
            move(stage, target, gripper)
    gain = float(data.xpos[body_id, 2] - start_z)
    if place_xy is not None:
        if gain < 0.04 or held_steps < round(.2 * control_hz):
            raise GraspFailure(f"{body_name}: failed to lift ({gain:.3f} m)")
        tcp = data.xpos[gripper_id] + data.xmat[gripper_id].reshape(3, 3) @ TCP_LOCAL
        offset = tcp - data.xpos[body_id]
        destination = np.array((*place_xy, start_z + lift_height)) + offset
        q_drop, _ = ik(destination, q_lift)
        move("transport", q_drop, -0.0255)
        q_release = data.qpos[addrs].copy()
        move("release", q_release, 0.0)
        table_id = model.geom("table_top").id
        body_dof = model.jnt_dofadr[model.body_jntadr[body_id]]
        consecutive = 0
        for _ in range(control_hz):
            step_command("settle", q_release, 0.0)
            supported = any(
                (c.geom1 == table_id and model.geom_bodyid[c.geom2] == body_id)
                or (c.geom2 == table_id and model.geom_bodyid[c.geom1] == body_id)
                for c in data.contact)
            stable = (supported and not jaw_contacts()
                      and np.linalg.norm(data.qvel[body_dof:body_dof + 3]) < .05
                      and np.linalg.norm(data.qvel[body_dof + 3:body_dof + 6]) < .5)
            consecutive = consecutive + 1 if stable else 0
            if consecutive >= 3:
                return gain
        raise GraspFailure(f"{body_name}: dropped body did not settle on table")
    return gain
