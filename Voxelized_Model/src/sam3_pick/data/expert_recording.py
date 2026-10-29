"""Physics-driven cover shuffle: free cubes, externally driven covers, no object teleport."""

import mujoco
import numpy as np
from sam3_pick.simulation.scene import snapshot


def capture(scene, store, phase, valid, target=None):
    snapshot(scene, store, phase, valid)
    data = scene.data
    for key in ("mocap_pos", "mocap_quat", "eq_active"):
        store.setdefault(key, []).append(getattr(data, key).copy())
    store.setdefault("mocap_target", []).append(
        data.mocap_pos.copy() if target is None else target.copy()
    )
    state = np.empty(mujoco.mj_stateSize(scene.model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(scene.model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    store.setdefault("integration_state", []).append(state)
