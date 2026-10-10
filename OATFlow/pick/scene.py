"""Counterfactual four-object scenes and contact-based pick-only expert."""
from itertools import permutations

import mujoco
import numpy as np

from controllers.oracle_pick import GraspFailure, pick
from OATFlow.environment.env import TARGET_HALF_HEIGHT
from OATFlow.environment.success import jaw_contacts
from OATFlow.task import TARGETS

PERMUTATIONS = tuple(permutations(TARGETS))


def scene_spec(seed):
    # Separate streams keep initial joints independent of layout and query.
    spatial, robot = [np.random.default_rng(s) for s in
                      np.random.SeedSequence(seed).spawn(2)]
    centers = np.array(((.415, -.24), (.495, -.24), (.605, -.24), (.685, -.24)))
    centers += spatial.uniform((-.018, -.045), (.018, .045), (4, 2))
    offsets = spatial.uniform(-.006, .006, (4, 2))
    positions = centers + offsets
    if min(np.linalg.norm(a - b) for i, a in enumerate(positions) for b in positions[i + 1:]) < .055:
        raise ValueError("overlapping slots; choose another group seed")
    return dict(seed=int(seed), centers=centers.tolist(), positions=positions.tolist(),
                yaw=spatial.uniform(-np.pi, np.pi, 4).tolist(),
                initial_arm=robot.uniform((-.28, -.22, -.25, -.30, -.25),
                                          (.28, .22, .25, .15, .25)).tolist())


def permutation_spec(spec, permutation):
    """Vary object nuisance poses within a layout; paired queries stay identical."""
    rng = np.random.default_rng(np.random.SeedSequence([spec["seed"], permutation, 55055]))
    positions = np.asarray(spec["centers"]) + rng.uniform(-.006, .006, (4, 2))
    if min(np.linalg.norm(a - b) for i, a in enumerate(positions) for b in positions[i + 1:]) < .055:
        raise ValueError("overlapping randomized slots")
    return dict(spec, positions=positions.tolist(), yaw=rng.uniform(-np.pi, np.pi, 4).tolist())


def setup(env, spec, permutation):
    mujoco.mj_resetData(env.model, env.data)
    env._park_covers()
    for slot, name in enumerate(PERMUTATIONS[permutation]):
        adr = env.target_qadr[name]
        yaw = spec["yaw"][slot]
        env.data.qpos[adr:adr + 7] = (*spec["positions"][slot], TARGET_HALF_HEIGHT[name] + .002,
                                       np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    arm = np.array(spec["initial_arm"])
    env.data.qpos[env.robot_qpos_addresses] = np.r_[arm, 0.]
    env.data.ctrl[:] = np.r_[arm, 0.]
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_step(env.model, env.data, 250)
    env.data.time = 0.
    env.initial_positions = env.target_positions()
    for name, position in env.initial_positions.items():
        if abs(position[2] - TARGET_HALF_HEIGHT[name]) > .003:
            raise GraspFailure("initial robot pose disturbed an object")
    return env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy()


def execute(env, target, record):
    initial = env.target_positions()
    held, longest = 0, 0

    def callback(stage):
        nonlocal held, longest
        for name in TARGETS:
            if name != target and env.target_positions()[name][2] - initial[name][2] > .04:
                raise GraspFailure("expert lifted a distractor")
        gain = env.target_positions()[target][2] - initial[target][2]
        held = held + 1 if gain > .04 and len(jaw_contacts(env.model, env.data, target)) == 2 else 0
        longest = max(longest, held)
        record(target + "_" + stage)

    point = env.data.xpos[env.model.body(target).id].copy() + (0, 0, .010)
    pick(env.model, env.data, target, point, on_step=callback)
    if longest < 10 or env.target_positions()[target][2] - initial[target][2] <= .04:
        raise GraspFailure("missing stable bilateral target lift")
    return dict(max_bilateral_hold_frames=longest)
