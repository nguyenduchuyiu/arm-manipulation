# e85_rvec

Status: sample export and controller replay complete; not selected for training.
The user selected the restored10D rotation6D output. No e85 model was trained.
Current shared code uses e84 rotation6D; the settings below describe the historical
rotvec sample only. Artifacts are preserved, not overwritten.

Replace e84's output relative rotation6D with a3D rotation vector in radians.
Action has7channels: chunk-anchor delta XYZ (metres), relative rotvec (radians),
absolute right-jaw command (metres). For each target t in H25:
omega[t]=Log(R_target[t] @ R_anchor.T), where R_anchor is the measured EE
orientation at the beginning of the chunk. Decode R_target[t]=Exp(omega[t])
@ R_anchor. The anchor is fixed throughout the chunk; this is not consecutive
frame integration. Labels use the principal log, with angle up to pi. Exact
180degree rotations have the usual axis-sign ambiguity. No Euler or6D vector
subtraction is used. Target0 is the measured pose after the first40ms, so its
rotvec need not be zero. Odd terminal frames clamp to the actual terminal state;
padding repeats the final label and is excluded from statistics/loss.

Policy input remains10D: measured EE base XYZ, absolute rotation6D, measured jaw.
Goal input remains base XYZ. Base origin is e08 config.base_origin_world_m;
base/world axes are aligned in this scene. No arm joint input or FK inside the
policy. Offline observations and live rollout use the same state conversion.

Pose input9, pose output6 and goal XYZ use per-channel train-only mean/std.
Gripper input/output use fixed range: g_norm=-1-2*g/0.0255, open=-1 and closed=+1.
Inverse: g=-0.0255*(g_norm+1)/2. Stored data remain in physical units.
Policy.sample returns physical units after inverse normalization, including
rotvec radians and jaw metres; IK receives reconstructed absolute EE rotation.
Shared mean/std entries for gripper are identity to avoid double normalization.

Planned training, not run: original e08 H25/25Hz/stride2 data and e79 voxel-disjoint
split(seed78), B256,60000updates, BF16, fused AdamW(beta0.9/0.95,wd1e-4),clip1,
LR2e-4/min2e-5 with recorded e79 warm100/6kcosine/restart200/60kcosine schedule.
Source e70_cf10/latest.pt; restore FM backbone and context projections, initialize
state10 encoder/projection and new action7 projections. All learned modules train.
No source optimizer or joint/action statistics are restored. Model selection and
heldout protocol remain the e84/e79 settings:5k intervals,2048validation chunks,
32validation goals, H25/K15,flow10,300frames;128heldout goals/noise0,1, initial
and expert-state chunks,32train rollouts. Physical success remains a position/
stability/gripper metric; rotation error is reported separately.

Full source dataset/checkpoints/baseline artifacts are still unavailable, so
the full export and60k training have not started. Sample/test artifacts live at
workspace_analysis/results/e85_rvec/. Earlier e84 artifacts remain intact.

Completed sample:42H25 chunks, action7/state10, exported from original e08
test_k64/v038800. Completed original-mesh dynamic IK replay:84frames at25fps,
H25/K10,500Hz physics, mean tracking0.268mm/0.208deg, no penetration or clipping.
Artifact paths:workspace_analysis/results/e85_rvec/test_sample/ and test_replay/.
No learned checkpoint, flow sampling or noise seed was used for physical replay.
CPU checks passed for rotvec reconstruction/anchors/padding and model action7
forward/backward/sampling before the output representation was reverted.
