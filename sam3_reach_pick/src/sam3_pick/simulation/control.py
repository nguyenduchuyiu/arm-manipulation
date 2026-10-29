"""25Hz actions,50Hz midpoint servo,500Hz MuJoCo physics."""

import mujoco


def step(model, state, held, command):
    for servo in ((held + command) * 0.5, command):
        state.ctrl[:] = servo
        mujoco.mj_step(model, state, 10)
    mujoco.mj_forward(model, state)
