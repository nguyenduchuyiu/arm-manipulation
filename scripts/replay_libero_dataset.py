"""Replay every episode in the downloaded LIBERO task and report success."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

from libero.libero.envs import TASK_MAPPING


DATASET = Path(
    "data/libero_sample/"
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5"
)
BDDL = Path(
    "LIBERO/libero/libero/bddl_files/libero_90/"
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet.bddl"
)
REPORT = Path("outputs/libero_replay_report.json")


def create_environment(env_args: dict):
    env_kwargs = deepcopy(env_args["env_kwargs"])
    env_kwargs.update(
        {
            "bddl_file_name": str(BDDL.resolve()),
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "ignore_done": True,
            "hard_reset": False,
            "control_freq": 20,
        }
    )
    return TASK_MAPPING[env_args["problem_name"]](**env_kwargs)


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    if not BDDL.exists():
        raise FileNotFoundError(BDDL)

    with h5py.File(DATASET, "r") as dataset:
        data = dataset["data"]
        env_args = json.loads(data.attrs["env_args"])
        env = create_environment(env_args)

        demo_names = sorted(
            data.keys(), key=lambda name: int(name.removeprefix("demo_"))
        )
        results = []
        for demo_name in demo_names:
            demo = data[demo_name]
            states = demo["states"][()]
            actions = demo["actions"][()]

            env.reset()
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            success_step = None
            max_state_error = 0.0
            for step, action in enumerate(actions):
                env.step(action)
                if step + 1 < len(states):
                    replay_state = env.sim.get_state().flatten()
                    max_state_error = max(
                        max_state_error,
                        float(np.linalg.norm(replay_state - states[step + 1])),
                    )
                if success_step is None and env._check_success():
                    success_step = step

            action_success = bool(env._check_success())

            env.sim.set_state_from_flattened(states[-1])
            env.sim.forward()
            recorded_final_success = bool(env._check_success())

            result = {
                "demo": demo_name,
                "steps": len(actions),
                "action_replay_success": action_success,
                "recorded_final_success": recorded_final_success,
                "success_step": success_step,
                "max_state_l2_error": max_state_error,
            }
            results.append(result)
            print(
                f"{demo_name}: action={action_success} "
                f"recorded={recorded_final_success} "
                f"max_state_error={max_state_error:.4f}"
            )

        env.close()

    action_successes = sum(result["action_replay_success"] for result in results)
    recorded_successes = sum(result["recorded_final_success"] for result in results)
    report = {
        "dataset": str(DATASET),
        "task": "close the top drawer of the cabinet",
        "num_demos": len(results),
        "action_replay_successes": action_successes,
        "action_replay_rate": action_successes / len(results),
        "recorded_final_successes": recorded_successes,
        "recorded_final_success_rate": recorded_successes / len(results),
        "results": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"action replay: {action_successes}/{len(results)}")
    print(f"recorded final states: {recorded_successes}/{len(results)}")
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
