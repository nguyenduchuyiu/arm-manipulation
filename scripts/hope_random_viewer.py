"""Watch ten random NexArm/HOPE rollouts in MuJoCo viewer."""
from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from envs import HopeNexArmEnv


def main() -> None:
    env = HopeNexArmEnv(
        image_width=96,
        image_height=96,
        max_episode_steps=60,
        settle_steps=0,
    )
    rng = np.random.default_rng(2026)

    front_id = env.model.camera("front").id
    print("Opening viewer for 10 live rollouts. Select wrist in the camera menu.", flush=True)
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = front_id
            for episode in range(10):
                _, info = env.reset(seed=episode)
                viewer.sync()
                target = info["target_object"]
                total_reward = 0.0
                for step in range(env.max_episode_steps):
                    if not viewer.is_running():
                        return
                    if step % 5 == 0:
                        action = rng.uniform(-0.5, 0.5, size=7).astype(np.float32)
                        action[6] = rng.choice((-1.0, 1.0))
                    start = time.monotonic()
                    _, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    viewer.sync()
                    time.sleep(max(0.0, 0.05 - (time.monotonic() - start)))
                    if terminated or truncated:
                        break
                print(
                    f"Episode {episode + 1:2d}/10: target={target}, "
                    f"steps={info['elapsed_steps']}, reward={total_reward:.0f}, "
                    f"reason={info['terminated_reason']}",
                    flush=True,
                )
            print("Rollouts finished; viewer stays open for inspection.", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
    finally:
        env.close()


if __name__ == "__main__":
    main()
