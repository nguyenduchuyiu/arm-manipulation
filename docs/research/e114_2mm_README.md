# E114 two-millimetre depth compression and visualization

Status: complete. Same exact621-frame320x240 depth reference and same FFV1 flags
as test_depth_compress, now quantized to2mm/unit. Reference is decompressed from
the bit-exact depth_float32.zlib; no new depth render. Common valid range remains
(0,6.5535]m, code0 invalid, matching0.1mm and1mm comparisons. Codec must decode
all uint16 values exactly; geometric error measured separately against float32.

Case g000/p01/t1 BlueCube,24.84s at25FPS, ideal MuJoCo front-camera depth.
No training or tracker/policy inference. No new checkpoint, epochs/batch/LR/H/K/
flow/noise settings. Existing TCOW epoch9/step4608 masks and target cloud reused
only in preview. Original training metadata remains in e114_mask/train_3.

Outputs compressed gray16le MKV, results.json, config.json and source. Preview
under view_uint16 decodes this actual file, pairs it with RGB, and overlays the
original saved target cloud (not recomputed using2mm depth). Cyan measured /
orange inferred, white outline, fixed turbo_r0.5--1.25m background, dark invalid.
Camera/crop/perspective unchanged; full dense depth, no scene-dot grid/naive panel.

Checks: codec pixel equality, depth/XYZ MAE/max/P95 vs valid original pixels,
file size reduction vs0.1mm and1mm; preview full decode/frame count/calibration;
source hashes unchanged; screenshot inspection. Ideal simulated data does not
establish real-camera compression or grasp performance. Float32 reference and
all earlier outputs retained.

Run: `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.tracking.depth_2mm`
Output: `/workspace/robotics_assets/experiments/e114_mask/test_depth_compress_2`.

Measured payload1,892,269bytes (1.892269MB),50.41x smaller than uint16 raw;
14.68% smaller than1mm and40.30% smaller than0.1mm. Equivalent0.274GB/hour
only if other clips compress identically. Codec reconstruction exactly equals
all original integer codes. Quantization depth MAE0.51436mm / max1.00006mm,
XYZ MAE0.54790mm / max1.16828mm over the same46130797 valid pixels.
Invalid count1562003 unchanged. Full preview decoded621frames; original
depth and saved mask/cloud/metric hashes unchanged. Initial screenshot inspected.
Target overlay remains original saved geometry, not recomputed from2mm depth.
