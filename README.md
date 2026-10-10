# NexArm manipulation

The main project is [OATFlow](OATFlow/README.md): pretrain a vision manipulation prior with FM or ACT, then train TCOW + FM end-to-end for object memory through cover shuffling. Its learned policies use absolute joint commands. Shared NexArm/HOPE environments and the standalone viewer are retained.

## Setup

Tested with Python 3.12, CUDA PyTorch 2.7.0 and MuJoCo 3.10.0 on Linux. Install FFmpeg system libraries for PyAV/TorchCodec (for example, `ffmpeg` on Debian/Ubuntu).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

# TCOW is a local upstream dependency, excluded from this repository.
git clone https://github.com/basilevh/tcow.git OATFlow/third_party/tcow
git -C OATFlow/third_party/tcow checkout a72e3e13a45e4156137328e5290f9e848d360367

# HOPE meshes are also fetched separately.
git clone https://github.com/swtyree/hope-dataset.git assets/hope-dataset
git -C assets/hope-dataset checkout 621d855f58817f8edbb4367ee0efbb7786a59a66
(cd assets/hope-dataset && python setup.py --meshes-eval)

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The HOPE [downloader](https://github.com/swtyree/hope-dataset/blob/621d855f58817f8edbb4367ee0efbb7786a59a66/setup.py) installs the low-resolution meshes used by the scenes. TCOW includes its modified TimeSformer; `OATFlow/policy/tracking.py` loads it directly.

Supply ImageNet ViT-B/16 weights, a prepared [SmolVLA base](https://huggingface.co/lerobot/smolvla_base) action-expert export, and TCOW initializer/configuration checkpoints. These are external artifacts, not files committed to this repository. The FM source revision is `d9f33c94a60fb382c90dea2164c96845bd955e28`; the action expert loader validates its exact tensor keys. See [prior training](OATFlow/prior/README.md) and [joint training](OATFlow/policy/README.md) for the required inputs and commands.

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
mjpython -m scripts.hope_random_viewer
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
