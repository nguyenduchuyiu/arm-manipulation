# E114: RGB-D target surface visualization

Status: completed. Diagnostic replay, no training and no FM control.

Source: `/workspace/robotics_assets/experiments/e114_mask/test_videos`.
TCOW checkpoint: e114_mask/train_3/best.pt, epoch 9, step 4608. Original target-only mask
fine-tuning retained; this visualization reuses stored causal predictions, no parameter updates.
Training hyperparameters remain those recorded in e114_mask/train_3/config.json.
No training epochs, batch, learning rate, action horizon, flow steps or noise seed apply to this replay.

Cases: first predetermined BlueCube unseen-permutation case g000/p01/t1 and YellowCube
both-unseen case g036/p01/t3 from the existing video selection. Full sequence from initial
visible prompt through covering, shuffle, expert uncovering and lifting. Recorded expert
actions; 25 FPS, 621 and 606 frames respectively. Clouds update only at existing causal
mask prediction endpoints (stride 3 plus segment terminal); no future interpolation.

Sensor: replay saved qpos in the original visual scene and render ideal top-camera axial
depth at 320x320; center crop y=40:280 exactly matches TCOW. Use calibrated camera and
robot-base transform for backprojection. This is synthetic depth, not a real RGB-D test.
Saved RGB is the original compressed recording.

Estimator inputs: current predicted amodal target mask, RGB/depth, calibrated camera,
initial visible mask prompt. No subsequent GT masks, visibility labels, body positions,
container identities, or object poses enter the estimator.

Pink cloud: naive predicted amodal mask times current depth (can select cover).
Cyan cloud: eroded predicted mask, filtered by color similarity to initial target.
Color gate uses chromaticity L2 < 0.16; visible support >=25% of initial valid target pixels
and >=12 pixels. This is a hand-designed gate for the colored blocks, not generic segmentation.
Orange cloud: initial observed surface points translated to the current amodal-mask centroid
ray's intersection with the initial surface-height plane. No full 3D completion, rotation,
occluder-motion estimation, or calibrated uncertainty. If mask is empty, hold last estimate.
Hidden fixed-height translation is an explicit tabletop prior and fails for hidden lifting,
rotation, bad mask centroids, or incompatible geometry.

GT body pose is read AFTER each estimate for scoring only. Reference is the initial observed
surface median transformed with the object, not object center or grasp goal. It is deliberately
not used in the point-cloud estimator. Metrics compare this reference and predicted surface
center, with hidden/visible strata from GT only after estimation.

Outputs: two MP4s, initial/hidden/reappearance PNGs, packed variable-length clouds in NPZ,
per-frame metrics JSON, configuration and summary. Checks: projection/backprojection
roundtrip <0.001 pixel; full video decode and exact frame count. Visual inspection required.

The original run used `MUJOCO_GL=egl .venv/bin/python -u -m
OATFlow.tracking.pointcloud_video`. Its exact source is stored beside the outputs.
The current script now renders a new front observation camera; see
[e114_front_README.md](e114_front_README.md).

Results: all 621/606 frames decoded. Projection/backprojection maximum error
2.84e-14 pixels. On 114/112 fully occluded prediction endpoints, median surface-reference
error is 1.19/6.22 mm versus naive mask-times-depth 48.61/48.23 mm.
These are two demonstrations with ideal depth and a tabletop translation prior,
not a general XYZ localization benchmark or a real-sensor/grasp success result.
Videos include a local 3x surface detail, fixed metric grid and coordinate axes.

Alternate view: `test_pointcloud_2` uses azimuth -145 degrees and elevation
18 degrees (original: 35/42 degrees). This rotates the visualization to the
opposite side and lowers the viewing angle; the observation camera, masks,
geometry estimates and scoring are identical. Each output folder includes its
exact source snapshot and config.
Alternate video rendering status: complete. Both MP4s fully decoded (621/606
frames); every saved geometry array and per-frame metric matches the original
view exactly. Inspected the hidden-target snapshot from the new viewing angle.
