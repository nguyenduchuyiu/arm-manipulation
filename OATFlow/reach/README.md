# Joint-input delta-EE reach

`ReachPolicy` loads robotics_assets/models/reach_joint_delta_ee/best.pt strictly.
Input base goalXYZ3 and measured5arm joints(rad)+right jaw(m). OutputH25x10:
measured chunk-start deltaXYZ3,relative rotation6D6,absolute gripper1. Compose
R_target=delta_R@R_anchor,not a difference of6Dvectors. Use simulation.pose.PoseIK
for bounded5joint pose control. The model is a reach/grip policy; it has no
learned object-lift or cover-removal stage.

Goal/joint/first9action channels use saved train-only mean/std. Jaw uses
-1-2*g/.0255 and inverse -.0255*(g_norm+1)/2; it is never delta. Sampling uses
flow10 by default. Current best is115000; latest120000 includes AdamW/RNG/sampler.

```bash
python -m OATFlow.reach.train --updates 1000 --output /tmp/new_reach
MUJOCO_GL=egl python -m OATFlow.tracking.pick_video --reach /tmp/new_reach/latest.pt --output /tmp/new_reach_video
```

--resume defaults to the curated latest.pt. This minimal continuation trainer
restores the fixed train split,normalization,optimizer,RNG and sampler; uses
masked10D flow-matching loss,original batch256,constant restoredLR2e-5,BF16,clip1.
It saves latest only; no validation/best selection is implied. Train-only data
are the unified voxel chunks,including aligned measured joint inputs. Two
updates and one+resume+one were verified to produce identical weights,AdamW and
RNG. New training results require evaluation before replacing the retained best.
