# E114 front-oblique RGB-D and TCOW replay

Status: complete. New observation camera, not a rotation of an existing cloud.
No training or FM inference; unchanged e114_mask/train_3/best.pt epoch9/step4608,
originally fine-tuned on top-camera RGB. This is an unseen-camera diagnostic,
not a checkpoint trained for this viewpoint. Training configuration is recorded
in the checkpoint and e114_mask/train_3 artifacts. No epochs/batch/LR/action
horizon/flow steps apply to training here; TCOW inference batch4, BF16.

Camera fixed in world at (0.54,-0.66,0.60)m, looking at (0.54,-0.12,0.035)m.
FOV55 degrees; 320x320 RGB/depth, center crop y40:280 to240x320. About46 degrees
downward, frontal rather than top-down. Camera is independent of target GT.
Both RGB input to TCOW and depth source are rendered from this camera.

Cases: g000/p01/t1 BlueCube (621frames) and g036/p01/t3 YellowCube (606frames).
Includes initial visible prompt, covering, shuffle, expert uncovering and lifting.
25FPS recorded expert replay; no newly controlled actions. Existing stride3 plus
segment-terminal sampling times reused, but all TCOW predictions recomputed.
Each prediction uses30 sampled frames from the causal prefix, initial frame and
current frame included; only frame0 visible-mask prompt. No subsequent GT mask,
visibility, object pose, container ID, or future image is a model input.

Ideal MuJoCo depth supplies observed surface geometry, with no real sensor noise.
Same initial-template/color-gate/plane-memory estimator as the top-camera video:
chromaticity L2<0.16, eroded predicted mask, support>=25% initial valid pixels and
>=12pixels. Insufficient support transports initial surface to amodal-mask-centroid
ray intersection with its initial height plane. Empty mask holds last estimate.
This is a handcrafted colored-object tabletop prior, not full-shape completion;
hidden lifting/rotation and mask drift invalidate it. Reference point is the
initial observed surface median attached to object, not object center/grasp target.

GT segmentation rendered separately to obtain frame0 prompt and evaluate mask IoU/
visibility. Later GT never enters inference or reconstruction. Body poses are used
only to produce sensor observations and subsequently score surface-reference error.

Outputs: MP4, screenshots, point clouds, new predicted masks and GT-only evaluation
masks, per-frame JSON, summary/config and exact source snapshots. No large features
cached. Check full decode, frame count, projection roundtrip and visual tracking.

Run: `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.tracking.pointcloud_video`
Output: `/workspace/robotics_assets/experiments/e114_mask/test_pointcloud_3`.

Both videos decoded in full (621/606 frames). Projection roundtrip max error
5.68e-14 pixels, checkpoint SHA256 unchanged. Initial surface templates now contain
201/234 points and include side faces (top-camera templates:110/102 points).
Mask IoU mean:60.43%/66.88%; fully occluded endpoints31.64%/56.52%.
Fully occluded surface-reference median error:22.66/2.09mm versus naive
mask-times-depth51.35/55.17mm. These use95/97 hidden endpoints defined from the
new viewpoint, so strata differ from the top-camera diagnostic. The blue case
shows substantial tracker viewpoint shift under occlusion. No hidden endpoint
was incorrectly classified as measured by the color-support gate.
Visible/partial median reference error0.34/0.82mm under ideal simulated depth.
Two clips only; not a general sensor or grasp benchmark. See summary.json and
per-frame JSON for complete results, including tracking failures.
