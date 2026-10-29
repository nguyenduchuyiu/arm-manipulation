# E118 — Direct object-cloud input through the original reach architecture

Status: COMPLETE — eight original-checkpoint rollouts and videos verified. No training or E120 feature/decoder path.

User requested feeding each object's cloud directly through the original FM
reach architecture. Load the entire E118 best checkpoint step120,000 strictly:
global XYZ normalization → per-point MLP → maxpool →960D goal projection → FM,
with the original measured joint/jaw state token. No weight from E120, no ViT,
no target-query decoder, no auxiliary head, no extra optimization.

E118 training source:32,017validated2cm voxel reach trajectories,25,613training
goals/1,651,684training chunks,120,000updates,B256 (~18.6data passes), scratch,
LR2e-4 after100step warmup,cosine to2e-5 at60k then constant;10D chunk-anchor EE
deltaXYZ,relative rotation6D,absolute jaw,H25. The original task reaches an
isolated voxel center and closes the gripper in air; it never learns object lift.

Test eight predefined E117 home-start scenes: g000/p01 and g001/p00, all four
targets each. Respectively unseen permutation/seen layout and new layout/seen
permutation relative to E120. This is a diagnostic subset, not a benchmark.

Each input is all valid depth pixels inside the eroded current query's **frame0
GT mask**, with median±60mm outlier rejection, back-projected using front camera
calibration into the robot base frame. This is an oracle object-selection test.
Keep this cloud fixed throughout each rollout, matching original static reach
goals. Preserve all object points; duplicate padding enables a variable-size
batch without changing maxpool output. Verify padded and unpadded conditioning.
No GT object center is a policy input; centers are scoring/overlay only.

H25/K10 (same execution length as current picking test),Euler10,noise seed0,
300frames,25Hz policy/50Hz servo/500Hz physics. All300frames are executed for every
case. No scripted grasp/lift/offset and no post-hoc goal correction. Report final
and minimum EE error to the initial body center, current-body error, position-only
10mm tolerance, bilateral contacts and physical lift. These are not the original
collision-aware stable air-reach success metric; contact with the table/objects
changes dynamics. Object clouds come from32mm cubes with another camera viewpoint,
whereas E118 was trained on20mm isolated voxel clouds.

Render all eight cases without outcome selection. Native front RGB overlaid with
the actual fixed input cloud plus oblique scene/input cloud/EE path. Cyan is the
actual cloud fed to FM; orange is the initial target center (diagnostic only).
12.5FPS samples every second25Hz state and preserves trajectory duration.
Source traces and all points retained in per-caseNPZ files; verify full video
decode and unchanged trace/checkpoint hashes.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.reach.object_rollout
```

Artifacts: `e118_cloud/test_pick_k10/config.json`, `input_verification.json`,
`results.json`, eight per-caseMP4/NPZ files, `video_verification.json`.

Actual result:0/8 final position within10mm; median final EE-to-initial-target error60.646mm;0bilateral contact cases;0pick successes. Rollouts18.33s, all eight video decodes and trace hashes passed. This is zero-shot transfer to32mm cubes on a table, not an E118 held-out voxel-reach benchmark.

Overview videos: `g000_p01_overview.mp4` and `g001_p00_overview.mp4` (red/blue/green/yellow in reading order).

- `g000_p01_t0.mp4`: RedCube, 200input points, final error72.12mm, min42.64mm.
- `g000_p01_t1.mp4`: BlueCube, 318input points, final error61.46mm, min40.70mm.
- `g000_p01_t2.mp4`: GreenCube, 217input points, final error52.00mm, min44.40mm.
- `g000_p01_t3.mp4`: YellowCube, 224input points, final error44.13mm, min37.80mm.
- `g001_p00_t0.mp4`: RedCube, 236input points, final error61.16mm, min54.34mm.
- `g001_p00_t1.mp4`: BlueCube, 269input points, final error84.87mm, min36.73mm.
- `g001_p00_t2.mp4`: GreenCube, 224input points, final error60.13mm, min33.05mm.
- `g001_p00_t3.mp4`: YellowCube, 199input points, final error59.42mm, min39.07mm.

Coverage audit: nearest E118 training goal is16.8–36.8mm from these target centers; this audit does not establish why the policy fails. No target correction or retraining was applied. See `coverage_audit.json`.
