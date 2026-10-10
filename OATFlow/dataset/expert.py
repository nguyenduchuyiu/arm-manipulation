"""Scene initialization and one physical expert shared by both collection modes."""
import mujoco
import numpy as np

from controllers.oracle_pick import GraspFailure, pick
from OATFlow.environment.env import COVER_XY, COVER_DROP_GOALS_XY, TARGETS, TARGET_HALF_HEIGHT
from OATFlow.environment.success import cover_deposited, target_lifted
from OATFlow.task import TASKS, task_id_from_state


def setup_scene(env, task_id, seed):
    rng = np.random.default_rng(seed)
    task = TASKS[task_id]
    mujoco.mj_resetData(env.model, env.data)
    centers = {name: np.array(xy) + rng.uniform((-.025, -.025), (.025, .025))
               for name, xy in COVER_XY.items()}
    cover = tuple(COVER_XY)[task["cover_side"]]
    assignment = {}
    for name in TARGETS:
        assignment[name] = cover if name in task["objects"] else next(c for c in COVER_XY if c != cover)
    env.assignment = assignment
    for owner in COVER_XY:
        names = [name for name in TARGETS if assignment[name] == owner]
        rng.shuffle(names)
        for index, name in enumerate(names):
            offset = rng.uniform((-.006, -.014), (.006, .014)) + (-.040 if index == 0 else .040, 0)
            address = env.target_qadr[name]
            yaw = rng.uniform(-np.pi, np.pi)
            env.data.qpos[address:address + 7] = (*centers[owner] + offset, TARGET_HALF_HEIGHT[name] + .002,
                                               np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    for name, address in env.cover_qadr.items():
        env.data.qpos[address:address + 7] = (*centers[name], .002, 1, 0, 0, 0)
    initial_arm = rng.uniform(-.06, .06, 5)
    env.data.qpos[env.robot_qpos_addresses[:5]] = initial_arm
    env.data.ctrl[:] = np.r_[initial_arm, 0]
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_step(env.model, env.data, 250)
    env.data.time = 0
    env.phase = "occlude"
    env.initial_positions = env.target_positions()
    if task_id_from_state(env, task["objects"]) != task_id:
        raise ValueError("oracle task ID disagrees with simulator assignment")
    return rng, cover


def execute(env, target, rng, record):
    env.query(target)
    cover = env.assignment[target]
    env.choose_cover(cover)
    initial_positions = env.target_positions()
    other_cover = next(name for name in COVER_XY if name != cover)
    other_start = env.data.xpos[env.model.body(other_cover).id, :2].copy()
    order = sorted(env.cover_qadr, key=lambda name: env.data.xpos[env.model.body(name).id, 0])
    drop_goal = COVER_DROP_GOALS_XY[tuple(COVER_XY)[order.index(cover)]]
    lift_gains = {}
    lift_contact_streaks = {}
    for index, body in enumerate((cover, target)):
        destination = drop_goal if index == 0 else None
        if index == 1:
            record("t_obj")
        point = env.data.xpos[env.model.body(body).id].copy()
        point[2] += .130 if index == 0 else .010
        point[:2] += rng.uniform(-.0015, .0015, 2)
        held_run, longest_held_run = 0, 0
        start_z = env.data.xpos[env.model.body(body).id, 2]

        def callback(stage):
            nonlocal held_run, longest_held_run
            for name, position in env.target_positions().items():
                if name != target and position[2] > initial_positions[name][2] + .04:
                    raise GraspFailure("wrong object lifted")
            record(body + "_" + stage)
            jaws = {env.model.geom(f"link_6_{side}_jaw_collision_0").id for side in ("left", "right")}
            body_id = env.model.body(body).id
            touching = {jaw for contact in env.data.contact for jaw in jaws
                        if ((contact.geom1 == jaw and env.model.geom_bodyid[contact.geom2] == body_id)
                            or (contact.geom2 == jaw and env.model.geom_bodyid[contact.geom1] == body_id))}
            if stage in ("lift", "hold") and env.data.xpos[body_id, 2] - start_z > .04:
                held_run = held_run + 1 if touching == jaws else 0
            else:
                held_run = 0
            longest_held_run = max(longest_held_run, held_run)

        gain = pick(env.model, env.data, body, point, place_xy=destination,
                    lift_height=.12, straight_lift=index == 0, on_step=callback)
        if gain < .04 or longest_held_run < 10:
            raise GraspFailure(f"{body}: lift gain {gain:.3f}, bilateral streak {longest_held_run} frames")
        lift_gains[body] = gain
        lift_contact_streaks[body] = longest_held_run
        if destination is not None and not cover_deposited(env.model, env.data, cover):
            raise GraspFailure(f"{body}: failed placement")
    mujoco.mj_forward(env.model, env.data)
    if not cover_deposited(env.model, env.data, cover) or not target_lifted(env.model, env.data, target, initial_positions[target][2]):
        raise GraspFailure(f"{target}: cover drop or terminal bilateral target lift failed")
    other_cover = next(name for name in COVER_XY if name != cover)
    if np.linalg.norm(env.data.xpos[env.model.body(other_cover).id, :2]
                      - other_start) > .015:
        raise GraspFailure(f"other cover displaced by {np.linalg.norm(env.data.xpos[env.model.body(other_cover).id, :2] - other_start):.3f} m")
    return dict(selected_cover=cover, cover_drop_goal_xy=list(drop_goal), lift_gains_m=lift_gains,
                bilateral_lift_streak_frames=lift_contact_streaks)
