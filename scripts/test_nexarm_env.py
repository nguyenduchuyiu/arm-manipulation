from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from envs.nexarm_env import NexArmEnv


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    env = NexArmEnv(
        scene_path="assets/robot/scene.xml",
        image_height=384,
        image_width=384,
        frame_skip=10,
        max_episode_steps=200,
        randomize_object=False,
    )

    observation, info = env.reset(seed=0)

    assert env.observation_space.contains(observation)

    print("Action space:", env.action_space)
    print("Control dt:", env.control_dt)
    print("State:", observation["observation.state"])
    print("Front:", observation["observation.images.front"].shape)
    print("Wrist:", observation["observation.images.wrist"].shape)
    print("Object:", info["object_position"])
    print("Contacts:", info["ncon"])

    iio.imwrite(
        output_dir / "front.png",
        observation["observation.images.front"],
    )
    iio.imwrite(
        output_dir / "wrist.png",
        observation["observation.images.wrist"],
    )

    # Hold the current physical pose; do not sample random full-range actions.
    action = env.home_action.copy()
    observation, info = env.reset(seed=0)

    for _ in range(100):
        observation, reward, terminated, truncated, info = env.step(
            action
        )
        if terminated or truncated:
            break

    assert np.all(np.isfinite(observation["observation.state"]))

    iio.imwrite(
        output_dir / "front_after_hold.png",
        observation["observation.images.front"],
    )
    iio.imwrite(
        output_dir / "wrist_after_hold.png",
        observation["observation.images.wrist"],
    )

    print("Final state:", observation["observation.state"])
    print("Final object:", info["object_position"])
    print("Final contacts:", info["ncon"])
    print("Saved images under outputs/")

    env.close()


if __name__ == "__main__":
    main()
