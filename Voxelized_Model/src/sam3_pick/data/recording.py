"""Collect dynamic shuffle contexts and contact-expert continuations for front/wrist RGB-D."""

import mujoco
import numpy as np
from sam3_pick.simulation.scene import save_segment

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def save(directory, store, offset=0, initial=None, terminal=None):
    n = save_segment(directory, store, offset=offset, initial=initial)
    with np.load(directory / "sim_state.npz") as z:
        state = {k: z[k] for k in z.files}
    for key in (
        "mocap_pos",
        "mocap_quat",
        "eq_active",
        "mocap_target",
        "integration_state",
    ):
        state[key] = np.asarray(store[key])
    state["integration_state_spec"] = int(STATE)
    if terminal is not None:
        state["terminal_integration_state"] = terminal
    np.savez_compressed(directory / "sim_state.npz", **state)
    return n
