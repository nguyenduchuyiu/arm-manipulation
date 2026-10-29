# E114 matched RGB / point-cloud camera

Status: complete. Visualization-only edit of e114_mask/test_pointcloud_3.
Two side-by-side panels: front RGB plus saved TCOW mask, and saved measured/inferred
target cloud. The previous comparison panel is removed. Point clouds use the exact
RGB camera extrinsics, perspective intrinsics, center crop and display scale.
No independently chosen visualization azimuth/elevation or orthographic projection.

Source camera: world position(0.54,-0.66,0.60)m, look-at(0.54,-0.12,0.035)m,
FOV55 degrees.320x320 rendering cropped y40:280, both panels scaled2.1.
Both panels672x504, complete video1440x860 at25FPS.

Source checkpoint e114_mask/train_3/best.pt epoch9/step4608, frozen. No training,
inference, feature cache, or new control policy. Masks and point-cloud estimates
are reused exactly from test_pointcloud_3. Training configuration is unchanged;
no new epochs/batch/LR/H/K/flow/noise settings apply. Recorded expert actions.
Cases: g000/p01/t1 BlueCube621frames, g036/p01/t3 YellowCube606frames.

RGB and gray scene cloud rendered from the recorded simulation state. Target cloud
retains causal stride3 prediction timing; no future interpolation. Cyan is measured
surface; orange is a fixed-height, translated initial-surface hypothesis displayed
through occluders. No missing geometry is completed. Ideal depth, no sensor noise.
The unchanged top-trained TCOW can drift on this new camera, especially under cover.
Original reconstruction protocol and limitations: e114_front_README.md.

Checks: camera transform equals stored cloud calibration; full-scene projection/
backprojection roundtrip<0.001pixel; exact decoded frame counts; input masks, clouds
and metric files hash unchanged. Inspect initial/hidden/reappearance PNGs.

The original run used `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.tracking.pointcloud_camera_video`.
Exact source is saved beside outputs. The current script exports the depth-colored
variant described in e114_depth_view_README.md.
Output: `/workspace/robotics_assets/experiments/e114_mask/test_pointcloud_4`.

Validation: both MP4s fully decoded (621/606 frames), camera transform matches
stored cloud calibration, full-scene roundtrip error1.14e-13pixels. Source mask,
cloud and metric hashes unchanged. Hidden-target screenshot visually inspected.
