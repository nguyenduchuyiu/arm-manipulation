# Checkpoint cleanup

Completed 2026-10-22T10:27:13.825849+00:00. Dataset, source code, pretrained downloads, logs, evaluation metrics and videos were not deleted.

Checkpoint allocation: 35.68 GiB -> 13.26 GiB. Freed 22.42 GiB. 20 redundant snapshot paths removed; 24 distinct retained checkpoint payloads.

Keep policy: retain best checkpoints for each run/architecture and distinct checkpoints already evaluated; preserve the exact E103 10epoch final model and E122 final model. Full optimizer resume state remains for E122 and E123. Other retained checkpoints support inference or further fine-tuning with a fresh optimizer; exact optimizer-state continuation is unavailable for removed snapshots. Identical complete checkpoint payloads share hardlinks. Optimizer-only removal was verified to preserve all model tensor values and other checkpoint metadata.

## Main checkpoints

| Purpose | Step | Path |
|---|---:|---|
| Original XYZ FM reach | 115000 | `/workspace/robotics_assets/models/reach_joint_delta_ee/best.pt` |
| Point-cloud FM reach | 120000 | `/workspace/robotics_assets/experiments/e118_cloud/train/best.pt` |
| E103 10epoch baseline | 22320 | `/workspace/robotics_assets/experiments/e103_center/train_2/final.pt` |
| E122 full-data policy (81.25% K15) | 23520 | `/workspace/robotics_assets/experiments/e122_full/train/final.pt` |
| E123 MobileSAM current training | 1280 | `/workspace/robotics_assets/experiments/e123_mask/latest.pt` |

## All distinct retained checkpoints

| Path (other aliases recorded in JSON) | GiB | Step | Optimizer |
|---|---:|---:|---|
| `/workspace/robotics_assets/experiments/e102_top/train/best.pt` | 0.720 | 2500 | False |
| `/workspace/robotics_assets/experiments/e103_center/train/best.pt` | 0.720 | 2500 | False |
| `/workspace/robotics_assets/experiments/e103_center/train_2/best.pt` | 0.720 | 10000 | False |
| `/workspace/robotics_assets/experiments/e103_center/train_2/final.pt` | 0.720 | 22320 | False |
| `/workspace/robotics_assets/experiments/e111_bc/test_k10/checkpoint.pt` | 0.541 | 10600 | False |
| `/workspace/robotics_assets/experiments/e111_bc/train/best.pt` | 0.541 | 15500 | False |
| `/workspace/robotics_assets/experiments/e113_voxel/resume100/best.pt` | 0.038 | 9215 | False |
| `/workspace/robotics_assets/experiments/e113_voxel/train/best.pt` | 0.038 | 970 | False |
| `/workspace/robotics_assets/experiments/e114_mask/train_3/best.pt` | 0.454 | 4608 | False |
| `/workspace/robotics_assets/experiments/e115_direct/test_full/checkpoint.pt` | 0.391 | 29420 | False |
| `/workspace/robotics_assets/experiments/e115_direct/test_k10/checkpoint.pt` | 0.391 | 9268 | False |
| `/workspace/robotics_assets/experiments/e118_cloud/test_k15/checkpoint.pt` | 0.392 | 75000 | False |
| `/workspace/robotics_assets/experiments/e118_cloud/test_pick_k10/checkpoint.pt` | 0.392 | 120000 | False |
| `/workspace/robotics_assets/experiments/e119_pick/test_k10/checkpoint.pt` | 0.721 | 2940 | False |
| `/workspace/robotics_assets/experiments/e119_pick/train/best.pt` | 0.721 | 1250 | False |
| `/workspace/robotics_assets/experiments/e120_fusion/test_k10/checkpoint.pt` | 0.722 | 23520 | False |
| `/workspace/robotics_assets/experiments/e120_fusion/train/best.pt` | 0.722 | 1250 | False |
| `/workspace/robotics_assets/experiments/e121_patches/train/best.pt` | 0.722 | 2750 | False |
| `/workspace/robotics_assets/experiments/e122_full/test_k10/checkpoint.pt` | 1.529 | 23520 | True |
| `/workspace/robotics_assets/experiments/e122_full/train/best.pt` | 0.722 | 12250 | False |
| `/workspace/robotics_assets/experiments/e123_mask/best.pt` | 0.042 | 1280 | True |
| `/workspace/robotics_assets/models/point_tracking/best.pt` | 0.454 | 1250 | False |
| `/workspace/robotics_assets/models/point_tracking_local/best.pt` | 0.454 | 1250 | False |
| `/workspace/robotics_assets/models/reach_joint_delta_ee/best.pt` | 0.391 | 115000 | False |

Full audit: `checkpoint_cleanup.json`. That file records removed paths, compaction, deduplication, and content hashes. Existing `storage_inventory.*` and `large_files.tsv` describe the earlier pre-cleanup snapshot.
