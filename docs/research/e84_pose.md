# e84_pose

Status: preparing. Training: /workspace/checkpoints/e84_pose.

Change e83's action to 10 channels: world-axis delta XYZ, world-axis relative
rotation in 6D, and absolute right-jaw command. Both anchors are the measured EE
pose at the beginning of each chunk. For k=0..24, target source state is
min(2*(start+k+1), T). Delta R = R_target @ R_anchor.T. Decode R_target = delta R
@ R_anchor. Rotation 6D stores the first two columns, column by column; identity
is [1,0,0,0,1,0]. It is not an Euler-angle subtraction. First action targets the
end of the first 40ms; its delta rotation need not be identity. Padded actions
repeat the final label and are excluded by the source mask.

Orientation comes from exact FK of recorded measured full qpos, checked against
recorded grasp-site XYZ for every source state. Actuator commands are not used
as joint poses. Reconstruction of both XYZ and rotation is audited.

Training planned, not started: original e08 H25/25Hz/stride2 chunks, unchanged
voxel-disjoint e79 split (seed78), B256, 60,000 updates, BF16, fused AdamW
(beta0.9/0.95, wd1e-4), clip1. LR2e-4/min2e-5 with the recorded e79 schedule.
Policy input: goal XYZ in workspace base frame and a10D measured state:
base XYZ (3), rotation6D (6), measured right-jaw position in metres (1).
No arm joints enter the policy. Gripper state is the measured jaw at chunk-start
source state2*start, not the previous or current actuator command. Online rollout
reads the right-jaw qpos at each replan. Pose state/action channels and goal
XYZ use train-only mean/std normalization. Gripper uses a fixed-range transform
for both input and the FM training target: g_norm=-1-2*g/0.0255, open=-1,
half-closed=0, closed=+1. It remains an absolute target, never a delta.
Dataset observations/labels remain in metres. Policy.sample returns physical
metres after g=-0.0255*(g_norm+1)/2, so existing evaluation and IK controllers
receive physical commands and must not denormalize a second time. Gripper
mean/std entries are identity to avoid a second transform in the shared FM head.
Workspace base origin comes from e08 config.base_origin_world_m; base axes match
world axes in this scene, so p_base=p_world-origin and R_base=R_world. Train,
offline evaluation and online rollout all use this same conversion. Online pose
comes from measured grasp_site position/orientation; no FK is inside the policy.

Source e70_cf10/latest.pt; restore action backbone/context projections only.
New pose9+gripper1 proprio MLP, state projection and action10 projections are initialized
fresh. Joint state weights/statistics are excluded from checkpoint loading.
All learned modules train; no FK buffers. Pose action, pose state and goal
normalization are recomputed from train chunks only. Gripper uses the fixed
physical range, independent of the training-set distribution. Source e79 is used for the split and
comparison artifacts, not input normalization.

Full data, source weights, baseline artifacts and the installed OATFlow package
remain unavailable on this instance; no full training run is claimed. A CPU
forward/backward and sampling check passed using the original FlowMatchingHead
code loaded from git HEAD and the real recorded sample. The temporary CPU test
environment does not change the project's training environment.

Pose IK uses position and rotation-matrix residuals, weight0.05/sqrt(2)m for the
rotation matrix (approximately0.05m/rad for small rotations), posture1e-4m/rad,
bounded least_squares/max40 evaluations, analytic Jacobian, and bias/gain servo
compensation. Rollout anchor is the measured current position/orientation,
fixed during each executed chunk; no accumulation of deltas. Training evaluation
retains H25/K15, flow10, 300frame budget, 5k validation intervals, and e79 heldout
scene indices/noise0,1. Existing physical success is a position/stability/gripper
metric; report rotation prediction and IK residual separately.

Completed sample: one existing e08 test_k64 trajectory v038800, 168 source frames
at50Hz, exported as42H25 pose chunks. Artifacts:
`workspace_analysis/results/e84_pose/test_sample_4/`, including state_ee_pose_base[N,10] and target_base_m. No new policy training.

Completed controller replay:
`workspace_analysis/results/e84_pose/test_replay_3/replay.mp4`. Recorded labels
and recorded chunk-start anchors, H25/K10, 25Hz target/50Hz midpoint servo/500Hz
physics;84video frames,3.36s,25fps. Left panel recorded measured states; right
new dynamic pose-IK replay. Original NexArm STL meshes recovered from git HEAD are used for rendering.
Rendering uses a separate model; mass/inertia/joints/actuators/equalities checked
identical to the physics model. Camera and lighting changes are render-only. No checkpoint,
flow sampling or noise seed. Every executed decoded pose matches the source.
Mean/max tracking:0.268/0.588mm and0.208/0.535deg. Final goal error0.0082mm,
final reference orientation error0.00315deg; no penetration or clipping.
This verifies one recorded feasible path, not arbitrary six-DOF poses on a
five-joint arm. The recorded-anchor replay does not measure a learned policy's
closed-loop performance.

Checks:100 random rotation6D roundtrips; analytic pose Jacobian vs finite
differences; pose export/masks/padding for eight odd/even source lengths;
sample recorded XYZ vs measured-qpos FK; video stream decode/metadata;
train-only pose/goal normalization; fixed-range gripper normalization; base-origin translation invariance;
offline and online measured-pose agreement; actual loss/backward and sample
shape[B,25,10]; rejection of6D joint input; synthetic old joint6 checkpoint
loading preserves the freshly initialized state encoder. After adding measured
gripper state, checked offline/online state10 equality (including closing/hold),
measured-versus-commanded gripper distinction, model gripper sensitivity and
nonzero gripper-column gradient, loss/backward/sampling, and old9D input rejection.

Reproduce from repository root, using fresh output paths:

```bash
/venv/main/bin/python workspace_analysis/sample_ee_delta.py workspace_analysis/results/e08_arm/test_k64/trajectories/v038800.npz workspace_analysis/results/e84_pose/test_sample_5
MUJOCO_GL=egl /venv/main/bin/python workspace_analysis/replay_ee_pose.py workspace_analysis/results/e84_pose/test_sample_5 workspace_analysis/results/e84_pose/test_replay_4
```

Fixed-range gripper check: the five specified open/25%/50%/75%/closed values
map to[-1,-0.5,0,0.5,1] and roundtrip to metres. Check the normalized model input,
normalized FM training target, physical-metre sampling output, no double
normalization, finite loss/backward and action sampling. Stored labels retain
absolute physical units.

Selected implementation: state10 and action10 with anchor-relative rotation6D; fixed-range gripper retained. The restored sample labels and state are byte-equal to the original6D export. Input/output projections, loss/backward, sampling and SO(3) reconstruction were checked after restoration. e85 rotvec artifacts are historical only; no training was run.
