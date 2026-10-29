# e113 — Two-view mask-conditioned voxel XYZ budget

Follow-up: the requested10-epoch run is complete; see
[training configuration and results](e113_train_README.md).

Status: complete. Budget/count/render/training benchmark, not full training.
No accuracy or convergence claim. Model: shared frozen ImageNet ViT-B/16,
320x320 RGB for top+left, all 400 patch tokens per view. Two binary current-view
target masks are average-pooled to patch occupancy (not applied to RGB), embedded
and added to projected visual tokens with learned view/spatial embeddings.
Four trainable transformer decoder layers, width384/6heads/MLP1536, one XYZ
query reading all 800 visual tokens; XYZ regression head. No temporal memory,
joint input, FM, actions or H/K. ViT stays eval/frozen.

Encoder source: robotics_assets/pretrained/vision/vit_b_16-c867db91.pth;
torchvision ViT-B/16 ImageNet, original positional embeddings interpolated to
20x20. All modules after ViT train from scratch. Bench optimizer AdamW lr1e-4,
BF16, TF32, no gradient checkpointing. MSE on normalized XYZ; repeated small
batches measure runtime, not independent epochs or held-out performance.

Candidate positions reuse the old 2cm reach voxel grid, status2 accepted
centers. One existing 32mm red cube, one target pose per voxel is the baseline
count; orientation/background/jitter variants multiply that count. Reachable EE
centers are not automatically valid object placements or observable in both
cameras. Count whole-cube frustum coverage, then target-vs-scene penetration
at home, with all other objects and covers parked away. This is not an exact
rendered visibility count. The one-object renderer pilot requires >=16 visible
target pixels in both views. Full collection may reject additional robot occlusions.

Suspended targets are static geometric localization examples; no physics steps,
IK solves, trajectory collection, or FM chunks are generated. Base-origin
conversion and exact camera intrinsics/extrinsics are archived. Cameras are
fixed and implicit in learned view embeddings, not explicit ray geometry input.

The benchmark creates a fresh output folder and copies this README. Retain only
metadata, measured timing and a few illustrative images. No persistent feature
cache or benchmark checkpoint is created.

Run:
```
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -u -m OATFlow.localization.voxel_budget --output /workspace/robotics_assets/experiments/e113_voxel/budget
```

## Actual counts and timing (RTX 3090)

The grid has 72,030 candidate centers, 32,017 dynamically validated EE targets.
For a single 32mm cube at yaw0, 15,882 accepted centers keep its whole geometry
inside the top camera, 29,569 inside left, and 15,848 inside both. Removing
cube-vs-robot/table penetration at the fixed home leaves 15,603. Rendered robot
occlusion can remove more: this is a conservative whole-frustum filter, not
a completed dataset manifest. Using every reach voxel requires changing camera
coverage, allowing partial/offscreen observations, or another explicit setup.
No camera was changed during the benchmark.

The final run is `robotics_assets/experiments/e113_voxel/budget_2/summary.json`.
256 real paired RGB/mask samples were rendered, with one rejection for visibility.
Single-worker rendering: 2.22s/256pairs (~115pairs/s), no physics rollout.
Frozen ViT encoding: 1.33s/256pairs (~193pairs/s), batch16pairs.
10,073,475 trainable decoder/projection/mask/head parameters; shared frozen
ViT-B/16 retains 400 spatial patch tokens per view (800 total), no pooling.

Cached-GPU training B128: ~52ms/update. RAM-cache training B128, including CPU
sample gathering and transfer to GPU: ~178ms/update. Measured20updates per
GPU-cached batch size16/32/64/128,20online updates B32,40RAM-cached updates B128;
each case also has5warmup updates. No meaningful accuracy evaluation was run.
Peak training CUDA allocation ~2.90GiB on the small pilot, not including a full
dataset loaded onto GPU. A full feature cache is intended to remain in host RAM.
Encoder tensor hash before/after is identical and all encoder gradients absent.

| Scope, one pose per voxel | Samples | FP16 features | uint8 masks | Render + encode once | Train/epoch from RAM (80% train) |
|---|---:|---:|---:|---:|---:|
| Current-camera filtered count | 15,603 | 17.86GiB | 2.98GiB | 3.60min | 17.47s |
| All reach voxels, nominal coverage | 32,017 | 36.64GiB | 6.11GiB | 7.40min | 35.83s |

These are extrapolated kernel/transfer costs from real pilot data. Validation,
checkpoint saving, full-scale memory behavior, data compression/loading and
additional visibility rejection are excluded; allow overhead. For current
camera coverage, budget roughly20-30s/epoch,3-5min/10epochs,30-50min/100epochs,
plus initial data generation/encoding. This estimates runtime, not epochs needed
to converge. More yaw/jitter/background variants multiply sample/cache counts.
Camera calibration is in `cameras.json`. Four preview images show paired RGB
and target mask overlays. No checkpoint/features were written; all pilot tensor
caches and repeatedly updated benchmark weights were discarded on process exit.
