# E114 RGB / native depth video

Status: complete. Display-only change: replace sparse point-cloud rendering with
the original dense depth map, directly colorized at every native pixel. No point
sampling, point splatting, geometry completion or inferred-target overlay.

Same front camera as test_pointcloud_3: fixed position(0.54,-0.66,0.60)m,
look-at(0.54,-0.12,0.035)m,FOV55deg.320x320 RGB/depth, crop y40:280; paired
672x504 panels in1440x860 video,25FPS. Depth uses nearest resizing to preserve
original depth edges. Fixed turbo_r scale0.5--1.25m for all frames and both cases,
warm near / cool far; out-of-range colors clip. Numeric depth is not altered.

Left panel overlays saved causal front-view TCOW masks. Right panel is sensor
depth only, with no target estimate drawn through occluders. MuJoCo depth is ideal
and is regenerated from recorded qpos; this is not a real RGB-D sensor evaluation.

Cases g000/p01/t1 BlueCube621frames and g036/p01/t3 YellowCube606frames.
Includes reveal/cover/shuffle/uncover/lift using recorded expert actions. Checkpoint
e114_mask/train_3/best.pt,epoch9/step4608,unchanged. No training or inference;
training parameters remain recorded in that run. No new epochs/batch/LR/H/K/
flow steps/noise settings apply. Causal masks reused from test_pointcloud_3.

Checks: full decoding/frame counts, correct camera transforms/crop, projection
roundtrip and unchanged source masks/cloud/metric file hashes. Inspect screenshot.
Prior camera/model limitations: e114_front_README.md.

The original run used `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.tracking.pointcloud_camera_video`.
Exact source is saved beside outputs. The current script restores the target
cloud over the dense-depth background, described in e114_depth_target_README.md.
Output: `/workspace/robotics_assets/experiments/e114_mask/test_pointcloud_6`.

Both videos fully decoded,621/606frames; source mask/cloud/metric hashes
unchanged. Camera roundtrip1.14e-13pixels. Initial screenshot inspected: dense,
continuous depth raster with visible robot/table/cube outlines, no dot lattice.
