# Reproduce: voxel reach → pick BC → SAM3 + FM

This is the current **E118 → E131 → E133** pipeline. Reach learns from a target point cloud; pick finetunes the same point encoder + FM; SAM3 supplies target masks at inference.

## 0. Environment and retained assets

Run from `/workspace/arm-manipulation`, using `.venv/bin/python`. Set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`. Run long commands in tmux with `set -o pipefail` and `2>&1 | tee -a <log>`.

Keep these external assets under `/workspace/robotics_assets/`:

| Asset | Used for |
|---|---|
| `robot/nexarm/` | Robot meshes, `physics_reference.xml`, IK seeds and settled home. |
| `models/reach_joint_delta_ee/{config.json,normalization.json,split.npz,evaluation_indices.npz}` | Original coordinate convention, train-only normalization and fixed voxel/evaluation splits. E118 reuses metadata, not learned weights. |
| `experiments/e117_wide/camera_samples/scene.xml` | Approved pick scene/cameras used by `wide.py`. |
| `experiments/e122_full/` label/split files listed in step 4 | Existing pick cloud/action cache consumed by E131. |

Paths are currently module constants. For a fresh run, use fresh outputs and update the relevant `ROOT`, `DATA`, `SOURCE`, `REFERENCE` constants consistently. Existing results are preserved; several commands reject nonempty output directories. The repository is not a standalone bootstrap without the retained assets/cache above.

## 1. Voxelize workspace and collect reach trajectories

```bash
.venv/bin/python -m OATFlow.data_tools.voxel_grid --output /workspace/robotics_assets/datasets/reach_voxel_grid --edge 0.02 --workers 8
.venv/bin/python -m OATFlow.data_tools.audit_voxels /workspace/robotics_assets/datasets/reach_voxel_grid --workers 8 --replay 16
.venv/bin/python -m OATFlow.data_tools.export_ee /workspace/robotics_assets/datasets/reach_voxel_grid /workspace/robotics_assets/datasets/reach_voxel_grid/ee_delta_25hz_h25_s2 --workers 8
.venv/bin/python -m OATFlow.data_tools.export_joints --chunks /workspace/robotics_assets/datasets/reach_voxel_grid/ee_delta_25hz_h25_s2
```

2 cm voxel centers → position-only IK → collision-checked joint path / RRT-Connect → MuJoCo rollout, close and hold. Reference result: **72,030 centers → 34,452 IK-valid → 32,017 validated trajectories**. Export **2,065,104 H25 chunks**, 25 Hz, start stride 2, measured joint inputs and masked tail padding.

Code: [voxel_grid.py](../../OATFlow/data_tools/voxel_grid.py), [audit_voxels.py](../../OATFlow/data_tools/audit_voxels.py), [export_ee.py](../../OATFlow/data_tools/export_ee.py), [export_joints.py](../../OATFlow/data_tools/export_joints.py). Keep the retained split only when trajectory/chunk ordering matches it; a changed dataset requires rebuilding split/statistics/evaluation indices.

## 2. Collect and render pick data

```bash
.venv/bin/python -m OATFlow.top_pick.wide --workers 8
.venv/bin/python -m OATFlow.top_pick.wide_render --workers 8
.venv/bin/python -m OATFlow.top_pick.wide_finish
```

Output: `datasets/pick_wide_rgbd/`. Four 32 mm cubes, XY uniform in `[0.30,0.78] × [-0.42,-0.10]` m, separation ≥55 mm, random yaw; reject layouts where the physical expert fails. Start at home with all objects visible. **40 layouts / 3,712 demos: train 2,048, val 256, test 1,408**, with held-out permutations/layouts. Front RGB + 2 mm depth, wrist RGB, instance masks, calibration and physical expert trajectories are saved. Wrist RGB is not used by E131/E133.

Code: [wide.py](../../OATFlow/top_pick/wide.py), [wide_render.py](../../OATFlow/top_pick/wide_render.py), [wide_finish.py](../../OATFlow/top_pick/wide_finish.py); expert: [pick/scene.py](../../OATFlow/pick/scene.py), [oracle_pick.py](../../controllers/oracle_pick.py). Collection resumes with `wide --resume`; rendering skips verified completed episodes. Run the audit without `--retire-old`.

## 3. Train the reach checkpoint from scratch — E118

```bash
.venv/bin/python -m OATFlow.reach.cloud_data
.venv/bin/python -u -m OATFlow.reach.cloud_train
```

For each accepted voxel, render an isolated **2 cm cube** with analytic calibrated front-camera depth, quantize to 2 mm and back-project **128 points in robot-base coordinates**. No per-cloud centering. Input: cloud + five measured arm joints and jaw. Output: **H25 × 10**, EE delta XYZ + relative rotation 6D + absolute jaw.

Train all parameters from random initialization: **120,000 updates, batch 256**, BF16, flow-matching loss. Output: `experiments/e118_cloud/train/best.pt`; the retained selected checkpoint is **step 120,000**. Add `cloud_train --resume` to resume. The trainer includes validation and final reach testing.

Code: [cloud_data.py](../../OATFlow/reach/cloud_data.py), [cloud_train.py](../../OATFlow/reach/cloud_train.py), [cloud_eval.py](../../OATFlow/reach/cloud_eval.py). Architecture: [pointcloud.py](../../OATFlow/reach/pointcloud.py), [model.py](../../OATFlow/reach/model.py), [flow.py](../../OATFlow/reach/flow.py). `reach/train.py` is a continuation trainer for the historical XYZ policy, not this scratch-training entry point.

## 4. Finetune picking with GT target clouds — E131

Prepare target clouds from **front GT instance mask × depth + calibration**, using [object_cloud.py](../../OATFlow/top_pick/object_cloud.py): erode 1 px, reject depth outliers, sample 128 base-frame points. Task ID selects the target mask externally.

The current trainer reads `experiments/e122_full/{labels.npz,cache_config.json,cache_jobs.json,partitions.json,train_manifest.jsonl,val_manifest.jsonl,test_manifest.jsonl,train/indices.npz}`. Keep these files to reproduce the existing run. They contain **294,640 train chunks + 2,048 sampled validation chunks**, with 512 fixed validation chunks scored. No RGB feature cache is needed by E131.

Cache-building code: [full_train.py](../../OATFlow/top_pick/full_train.py) `partition()` for the full split; [cloud_data.py](../../OATFlow/top_pick/cloud_data.py) `jobs_and_labels()`, `action_labels()`, `Clips(clean_depth_labels=True)`; [cloud_train.py](../../OATFlow/top_pick/cloud_train.py) `cache()` and validation-index selection. There is currently **no standalone point-cloud-only cache CLI**: rebuilding this cache needs an adapter around those functions. Running `full_train.py` directly also runs the older RGB/XYZ experiment and is not required for this recipe.

```bash
.venv/bin/python -u -m OATFlow.top_pick.object_train
.venv/bin/python -u -m OATFlow.top_pick.object_evaluate
```

Load the **entire E118 best checkpoint**, then train point encoder + state encoder + FM jointly: **23,520 updates, batch 128, ~10.22 epochs**, BC flow loss only; FM LR `2e-5`, point/goal LR `1e-4`. Preserve reach normalization. Output: `experiments/e131_object/train/final.pt`, **step 23,520**. Add `object_train --resume` to resume. Rollouts use this final checkpoint, not the validation-best step 17,250. Code: [object_train.py](../../OATFlow/top_pick/object_train.py), [object_evaluate.py](../../OATFlow/top_pick/object_evaluate.py).

## 5. Connect SAM3 to the frozen pick policy — E133

Use `.venv-sam3` with [sam3_requirements.txt](../../OATFlow/task_mask/sam3_requirements.txt), the shared PyTorch runtime, and HF credentials approved for `facebook/sam3`. [sam3_runtime.py](../../OATFlow/task_mask/sam3_runtime.py) downloads the pinned official weights automatically and runs BF16.

For the exact existing tests, retain the original XY16 manifest/states at `e122_full/test_xy/`; generate the expanded OOD394 manifest/states with `.venv/bin/python -m OATFlow.top_pick.object_ood`. [sam3_fusion.py](../../OATFlow/top_pick/sam3_fusion.py) `prepare` checks all three suites, even if `run --suite heldout` is selected later.

```bash
.venv/bin/python -m OATFlow.top_pick.sam3_fusion prepare
.venv-sam3/bin/python -u -m OATFlow.top_pick.sam3_fusion run
.venv/bin/python -m OATFlow.top_pick.sam3_video
```

Task ID → `red cube / blue cube / green cube / yellow cube` → SAM3 masks on **front + left RGB** → masked depth backprojection → **3 mm voxel fusion → 128 points** → frozen E131 point encoder + FM, with measured joints. No additional training. [two_view.py](../../OATFlow/top_pick/two_view.py) defines cameras/calibration/fusion; [sam3_rollout.py](../../OATFlow/top_pick/sam3_rollout.py) executes trajectories; [sam3_video.py](../../OATFlow/top_pick/sam3_video.py) replays saved traces/masks/clouds.

Output: `experiments/e133_fuse/`. Test **H25/K15/Euler10, seed 0, limit 300 frames**, 25 Hz policy / 50 Hz servo / 500 Hz physics. Success requires target lift >4 cm and two-jaw contact for 10 frames. `run --resume` continues interrupted batches and skips completed suites. Reference results: **1,314/1,408 held-out (93.32%)**, **194/197 spatial ID (98.48%)**, **183/197 spatial OOD (92.89%)**. Exact configs, hashes, summaries and saved trajectories remain beside each test.
