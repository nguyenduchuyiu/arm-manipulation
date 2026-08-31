# NexArm manipulation

The canonical robot assets live under `assets/robot`:

- `robot.urdf`: kinematics model.
- `robot.xml`: MuJoCo dynamics, collision, actuators, and wrist camera.
- `scene.xml`: manipulation scene with the cube and front camera.
- `libero_cabinet_scene.xml`: one LIBERO cabinet task with physical drawer collision.
- `meshes/`: shared visual meshes.

The arm has five revolute joints and one parallel gripper. `NexArmEnv` uses the
LIBERO-style normalized action `[dx, dy, dz, dax, day, daz, gripper]` in
`[-1, 1]`. Translation scales to 5 cm, axis-angle rotation to 0.5 rad, and the
gripper uses `+1` for open and `-1` for closed. The LeRobot backend keeps its
hardware-compatible raw servo convention (`0..4095`).

Run the environment smoke test:

```bash
python scripts/test_nexarm_env.py
```

Run the physical-contact grasp oracle:

```bash
python scripts/record_pick_episode.py --out outputs
```

The recorder writes synchronized front/wrist MP4s and `pick_episode.npz`. Its
primary `action` array is the normalized 7D LIBERO format; physical joint state
and the original raw servo values are also retained for debugging.

Open the interactive grasp viewer:

```bash
mjpython scripts/interact_grasp.py
```

Retarget the downloaded LIBERO top-drawer demonstration to NexArm:

```bash
conda run -n mujoco-vla python scripts/retarget_libero_drawer.py
```

This writes `data/nexarm_libero/close_top_drawer_demo_0.hdf5` and a verification
video at `outputs/nexarm_libero_close_drawer.mp4`. The drawer moves only through
MuJoCo contact; the script rejects the episode unless the drawer reaches the
closed threshold. Source EE waypoints are smoothed and cubic-spline resampled
from 20 Hz to a 100 Hz controller loop; observations and actions remain recorded
at 20 Hz. A smooth terminal EE push compensates for the Panda-to-NexArm geometry
difference without accumulating joint commands.

Render NexArm retargets for all 50 demonstrations:

```bash
conda run -n mujoco-vla python scripts/render_nexarm_libero_retargets.py
```

The 50 labeled MP4 files and their success manifest are written under
`outputs/nexarm_libero_50/`.

Retarget all 500 LIBERO-Spatial demonstrations with physical collision:

```bash
conda run -n mujoco-vla python scripts/retarget_libero_spatial.py
```

The 10 NexArm HDF5 files are written to `data/nexarm_libero_spatial`; train on
episodes whose `success` attribute is true.

Replay all original Panda demonstrations with LIBERO's generated BDDL scene:

```bash
PYTHONPATH=LIBERO conda run -n libero-replay \
  python scripts/replay_libero_dataset.py
```

The replay writes its per-episode success and state-divergence report to
`outputs/libero_replay_report.json`.
