"""Home-start four visible cubes; disjoint permutation and spatial holdouts."""

from sam3_pick.paths import ASSETS
import mujoco
import numpy as np
from sam3_pick.simulation.scene import NAMES, PERMS

HOME = ASSETS / "home.json"


def setup(scene, spec, permutation):
    model, data = (scene.model, scene.data)
    mujoco.mj_resetData(model, data)
    import json

    home = json.loads((ASSETS / "home.json").read_text())
    data.qpos[:8] = home["state_full_qpos"]
    data.qvel[:8] = home["state_full_qvel"]
    data.ctrl[:] = home["initial_ctrl"]
    for slot, color in enumerate(PERMS[permutation]):
        adr = scene.target_qadr[NAMES[color]]
        yaw = spec["yaw"][slot]
        data.qpos[adr : adr + 7] = (
            *spec["positions"][slot],
            0.018,
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        )
    for k, adr in enumerate(scene.cover_qadr.values()):
        data.qpos[adr : adr + 7] = (-2 if k == 0 else 2, 0, 0.15, 1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, 250)
    data.time = 0.0
    mujoco.mj_forward(model, data)
    for name in NAMES[:4]:
        assert abs(data.xpos[model.body(name).id, 2] - 0.016) < 0.003
    return (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
