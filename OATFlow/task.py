"""Cover choice and absolute expert action conventions."""
from itertools import combinations

import numpy as np


def normalized(values, limits):
    """Absolute positions: arm [-1, 1]; gripper 0 closed, 1 open."""
    result = (np.asarray(values) - limits[:, 0]) / np.diff(limits, axis=1)[:, 0]
    result[:5] = 2 * result[:5] - 1
    return np.clip(result, [-1] * 5 + [0], 1).astype(np.float32)


TARGETS = ("RedCube", "BlueCube", "GreenCube", "YellowCube")
PAIRS = tuple(combinations(TARGETS, 2))
TASKS = tuple(dict(task_id=2 * index + side, objects=list(pair), cover_side=side)
              for index, pair in enumerate(PAIRS) for side in range(2))


def cover_assignment_from_state(env):
    """Infer ownership from object centers inside the two closed covers."""
    interiors = {}
    for cover in env.cover_qadr:
        left, right, front, back, top = [env.model.geom(f"{cover}_{part}")
                                       for part in ("left", "right", "front", "back", "top")]
        lower = np.array((left.pos[0] + left.size[0], front.pos[1] + front.size[1], 0.))
        upper = np.array((right.pos[0] - right.size[0], back.pos[1] - back.size[1],
                          top.pos[2] - top.size[2]))
        body = env.model.body(cover).id
        interiors[cover] = (env.data.xpos[body], env.data.xmat[body].reshape(3, 3), lower, upper)
    assignment = {}
    for name in TARGETS:
        position = env.data.xpos[env.model.body(name).id]
        owners = []
        for cover, (center, rotation, lower, upper) in interiors.items():
            local = rotation.T @ (position - center)
            if np.all(local >= lower - 1e-6) and np.all(local <= upper + 1e-6):
                owners.append(cover)
        if len(owners) != 1:
            raise ValueError(f"{name} must be inside exactly one closed cover")
        assignment[name] = owners[0]
    return assignment


def task_id_from_state(env, pair):
    """Pair identity + left/right at the covered expert start, after shuffle."""
    pair = tuple(name for name in TARGETS if name in pair)
    if len(pair) != 2:
        raise ValueError("a task requires two distinct objects")
    assignment = cover_assignment_from_state(env)
    if assignment != env.assignment:
        raise ValueError("stored object/cover assignment disagrees with physical containment")
    cover = assignment[pair[0]]
    if assignment[pair[1]] != cover:
        raise ValueError("task objects must share a cover")
    order = sorted(env.cover_qadr, key=lambda name: env.data.qpos[env.cover_qadr[name]])
    return 2 * PAIRS.index(pair) + order.index(cover)


def physical_joint_action(action, limits):
    """Convert normalized absolute arm/gripper setpoints to actuator units."""
    action = np.asarray(action)
    fraction = np.concatenate(((action[..., :5] + 1) / 2, action[..., 5:6]), axis=-1)
    return limits[:, 0] + fraction * (limits[:, 1] - limits[:, 0])
