# NexArm manipulation

## Simulation and data collection

### Random HOPE tabletop scene

`assets/hope-dataset` contains eight selected low resolution meshes from the
[NVIDIA HOPE dataset](https://github.com/swtyree/hope-dataset)
(upstream commit `621d855f58817f8edbb4367ee0efbb7786a59a66`).
The original mesh archive passed the HOPE downloader's MD5 check. `HopeNexArmEnv`
uses the existing NexArm model and controller, places three random HOPE objects
on a table, and includes a chair. The target is the object in the middle. At
every `reset(seed=...)`, object identities, positions, and yaw change.
Objects spawn 2 cm above the tabletop with separate footprints, then fall
under gravity. The live viewer shows the fall at the start of each episode.
The target is chosen from five thin packages that fit the NexArm gripper when
rotated. Milk, OrangeJuice, and Tuna can also appear as distractors.

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/mujoco-vla/bin/mjpython -m scripts.hope_random_viewer
```

This opens MuJoCo viewer first, shows ten seeded random rollouts at a readable
pace, and prints each reward and ending reason. The viewer stays open afterward.
It starts on the front camera; the wrist camera can be selected in the viewer
camera menu.
The front camera views the arm from above and in front; the wrist camera follows
the gripper. Actions and image observations use the same format as `NexArmEnv`.
`step` returns `(observation, reward, terminated, truncated, info)`: reward is 1
after the target stays at least 8 cm above the table for ten control steps,
otherwise 0. Failure ends the episode if the target leaves the workspace or
joint speed exceeds the safety threshold; the step limit sets `truncated`.
HOPE contacts use box proxies and a nominal 80 g mass per object.

```python
from envs import HopeNexArmEnv

env = HopeNexArmEnv()
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(env.home_action)
# RGB and aligned metric depth from the same front camera:
front_rgb = observation["observation.images.front"]
front_depth_m = observation["observation.depth.front"]
env.close()
```

The canonical robot assets live under `assets/robot`:

- `robot.xml`: MuJoCo kinematics, dynamics, collision, actuators, and wrist camera.
- `hope_scene.xml`: generated HOPE tabletop scene and front camera.
- `meshes/`: shared visual meshes.

The arm has five revolute joints and one parallel gripper. `NexArmEnv` uses the
LIBERO-style normalized action `[dx, dy, dz, dax, day, daz, gripper]` in
`[-1, 1]`. Translation scales to 5 cm, axis-angle rotation to 0.5 rad, and the
gripper uses `+1` for open and `-1` for closed. The LeRobot backend keeps its
hardware-compatible raw servo convention (`0..4095`).

`lerobot-nexarm/` is a separate checkout for leader/follower teleoperation,
recording demonstrations, and deploying policies on the physical NexArm. The
HOPE MuJoCo environment does not import it.

## Memory and occlusion episodes

The complete RGB-D memory task, physical expert, episode generator, and dataset
loader live in [`memory_occlusion/`](memory_occlusion/README.md). Run
`.venv/bin/python -m memory_occlusion.dataset.generate_samples` to create
four sample demonstrations.
