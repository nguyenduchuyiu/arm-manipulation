# E114 depth-colored camera-aligned point cloud

Status: complete. Visualization edit, no training or inference. Source geometry and
causal masks unchanged from e114_mask/test_pointcloud_3, epoch9/step4608 of frozen
TCOW checkpoint e114_mask/train_3/best.pt. Training configuration remains as recorded
there; no new epochs, batch, LR, H/K, flow steps or noise settings apply.

Cases g000/p01/t1 (621frames) and g036/p01/t3 (606frames), full recorded expert
sequences at25FPS. Same front camera, perspective, crop and scale as test_pointcloud_4.
Two panels, RGB+mask and point cloud; no naive comparison.

Scene points colored by axial camera depth (not world height or per-frame RGB),
using turbo_r with FIXED0.5--1.25m scale across both clips and all frames. Near is
warm/red, far is cool/blue; colors clip outside scale. A metric colorbar is displayed.
Dark background, stride2 scene sampling, larger points improve robot/table/cover
visibility. Target remains cyan when measured or orange when inferred. Inferred
surface is shown through occluders and remains the existing fixed-height hypothesis.

Original camera/reconstruction/ideal-depth limitations: e114_front_README.md;
matched projection protocol: e114_camera_view_README.md. No real RGB-D sensor test.

Validation: full video decode/frame counts, camera projection roundtrip, source
mask/cloud/metric hashes unchanged, and visual inspection of screenshots.

The original run used `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.tracking.pointcloud_camera_video`.
Exact source is saved beside outputs. The current script exports native dense
depth, described in e114_raw_depth_README.md.
Output: `/workspace/robotics_assets/experiments/e114_mask/test_pointcloud_5`.

Both videos fully decoded,621/606frames. Projection roundtrip1.14e-13pixels;
source mask/cloud/metric hashes unchanged. Initial screenshot visually inspected:
robot and tabletop are visible with the fixed metric depth colors.
