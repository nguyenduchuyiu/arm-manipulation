# E118 best-checkpoint held-out rollout and videos

Status: COMPLETE. Rollout, six individual videos, overview and full media checks passed. User requested rollout and visualization of the current best E118.
Use an immutable hardlinked snapshot of train/best.pt; record its actual step/hash
at launch, never choose checkpoints on these test results. Observed best before
launch:step75000. Main120k scratch training continues independently.

Training source: E118 PointCloudReach, all modules random seed0,32,017reach goals,
1,651,684train chunks, batch256, full120kbudget, warmup100/cosine60k then2e-5.
Input128base-frame points from front depth of isolated2cm voxel + measured joints;
point encoder/state encoder/FM all train, no XYZ auxiliary head, no pretrained
weights. Main protocol:e118_cloud_README.md; actual train config ../train/config.json.

Test first16 of original128fixed held-out goals, disjoint from training voxels.
Select these and first6video cases before outcomes. H25/K15,Euler10,noise0,300frames,
25Hzactions/50Hzmidpointservo/500Hzphysics, same bounded PoseIK. Same reconstructed
success rule as E118 validation: final25frames inside error threshold, jaw closed
within1mm, EE speed<10mm/s, no penetration>0.2mm throughout rollout. This16case
subset is not a full128goal evaluation or grasp success rate. Target cube is a
visual marker only; robot reaches and closes in air, no inserted physical object.

Save full qpos/EE/commands, results and input/checkpoint hashes. Render6fixed
cases plus synchronized overview at12.5FPS, retaining original timing. Real robot
visual mesh, orange voxel/GT center, cyan input points and green measured EE path.
Input cloud inset is diagnostic; target center overlay never enters policy.
Complete video decoding/framecount and displayed EE replay checked.

Artifacts:e118_cloud/test_k15 (fresh folder created by tool). Tmux ssh_tmux:e118_test,
log:e118_cloud/test_k15.log. Code:OATFlow/reach/cloud_rollout.py.
Run:MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.reach.cloud_rollout
Exact snapshot/test settings:config.json; results:rollout.json/.npz;
videos:case_01..06.mp4 and overview.mp4; verification:video_checks.json.
Results on this16goal subset:5mm6/16(37.5%),10mm13/16(81.25%),2mm2/16;
median final5.180mm,mean6.535mm,p90 12.466mm,no collisions.
First6video final errors:6.16,8.37,2.59,6.38,10.23,1.84mm; all chosen before outcomes.
Main training and E116pause unchanged.

Final media verified:7videos,151frames each at12.5FPS; source25Hz timing
preserved. Visual-mesh forward-kinematics replay agrees with recorded EE within2um.
Lighting/closer-camera re-render uses the same saved physical rollout. Final render
source hash retained separately in config.json; original test/checkpoint hashes unchanged.
