# E114 dense depth context plus tracked target cloud

Status: complete. Visualization-only combination: full native colorized depth for
robot/table/cover context, with the saved target point cloud projected on top.
No sparse scene-point grid and no naive comparison. Both panels retain identical
front-camera projection/crop/scale,1440x860 video,25FPS.

Background: every native pixel, fixed turbo_r0.5--1.25m camera-depth scale,
nearest display enlargement. Target: actual saved measured/inferred points from
e114_mask/test_pointcloud_3, cyan measured / orange inferred, with a white outline
for contrast. Inferred target is deliberately shown through cover; background
depth is still the observed front surface. No hidden-object depth is measured.

TCOW e114_mask/train_3/best.pt epoch9/step4608, unchanged. No training or inference;
no new epochs, batch, LR, H/K, flow or noise settings. Training config remains
in original run. Same saved causal masks/clouds and expert state replay:
g000/p01/t1 BlueCube621frames and g036/p01/t3 YellowCube606frames.
Original camera/geometry/ideal-depth assumptions: e114_front_README.md.

Checks: full decode/frame count, calibrated projection roundtrip, unchanged input
mask/cloud/metric hashes, and visual inspection of hidden/visible screenshots.

The original run used `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.tracking.pointcloud_camera_video`.
Its exact source is saved beside outputs. The current script visualizes the
decoded uint16 file; see e114_uint16_view_README.md.
Output: `/workspace/robotics_assets/experiments/e114_mask/test_pointcloud_7`.

Both videos fully decoded (621/606frames); source mask/cloud/metric hashes
unchanged, projection roundtrip1.14e-13pixels. Hidden screenshot inspected:
dense depth context retained and orange target cloud visible with white outline.
