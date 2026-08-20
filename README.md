# NexArm manipulation

The canonical robot assets live under `assets/robot`:

- `robot.urdf`: kinematics model.
- `robot.xml`: MuJoCo dynamics, collision, actuators, and wrist camera.
- `scene.xml`: manipulation scene with the cube and front camera.
- `meshes/`: shared visual meshes.

The arm has five revolute joints and one parallel-gripper action. `NexArmEnv`
uses normalized absolute actions in `[-1, 1]`; the gripper convention is `+1`
for open and `-1` for closed. The LeRobot backend keeps its hardware-compatible
raw servo convention (`0..4095`).

Run the environment smoke test:

```bash
python test_nexarm_env.py
```

Run the physical-contact grasp oracle:

```bash
python scripts/record_pick_episode.py --out outputs
```

The recorder writes synchronized front/wrist MP4s and `pick_episode.npz` with
raw servo observations and actions.

Open the interactive grasp viewer:

```bash
mjpython scripts/interact_grasp.py
```
