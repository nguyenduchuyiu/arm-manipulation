"""Cover choice and absolute expert action conventions."""
import numpy as np
from gymnasium import spaces


COVER_ACTION_SPACE = spaces.Discrete(2)
EXPERT_ACTION_SPACE = spaces.Box(np.array([-1] * 5 + [0], dtype=np.float32),
                                np.ones(6, dtype=np.float32))


def normalized(values, limits):
    """Absolute positions: arm [-1, 1]; gripper 0 closed, 1 open."""
    result = (np.asarray(values) - limits[:, 0]) / np.diff(limits, axis=1)[:, 0]
    result[:5] = 2 * result[:5] - 1
    return np.clip(result, [-1] * 5 + [0], 1).astype(np.float32)
