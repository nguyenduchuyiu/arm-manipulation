# E114 depth compression benchmark

Status: complete. One actual full depth sequence, no training or model inference.
Case g000/p01/t1 BlueCube,621frames at25FPS (24.84s),320x240 native depth crop
from front camera (same camera as test_pointcloud_3). Full reveal/cover/shuffle/
uncover/lift recorded expert replay, ideal MuJoCo depth without sensor noise.
TCOW checkpoint is irrelevant to compression; no epochs/batch/LR/H/K/flow settings.

Compare original float32 depth compressed as zlib level6 with uint16 depth at
0.1mm and1mm units compressed in FFV1 gray16le Matroska. Also test actual16-bit
PNG frames, compress_level6, packaged in ZIP_STORED (no double compression).
FFV1 flags: level3,coder1,context1,g1,slices4,slicecrc1,threads4. Every compressed
file is fully decoded and checked against its encoded representation.

Loss must be separated: float32 zlib should be bit-exact for ALL pixels; FFV1/PNG
must preserve uint16 values exactly. Quantization error relative to float32 is
reported independently (depth MAE/RMSE/P95/max and backprojected XYZ MAE/max).
Both uint16 variants use the SAME common valid depth range (0,6.5535]m; beyond
this range encode invalid0. Original float32 preserves all original depths.
Far-plane background pixels in the simulator may exceed this sensor range;
their count/fraction is reported and they are excluded from geometry-error
metrics, rather than being misrepresented as codec error.

Outputs config.json,results.json,exact source,compressed payloads and README.
No raw scratch arrays are persisted. Compression ratio includes file/container
overhead and uses same-native-format raw bytes. Real sensor noise can reduce
compressibility substantially; this one synthetic clip is not a real-camera
storage guarantee. Encoding/decoding times include disk/pipe overhead.

Run: `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.tracking.depth_compress`
Output: `/workspace/robotics_assets/experiments/e114_mask/test_depth_compress`.

Actual results (decimal MB): raw float32 190.7712MB; float32+zlib12.635005MB,
15.10x smaller, decoded bit-exact for all pixels. Raw uint16 95.3856MB;
FFV1 0.1mm3.169473MB (30.10x), PNG16 0.1mm4.695255MB (20.32x),
FFV1 1mm2.217952MB (43.01x). All encoded integer pixels decoded exactly.
Quantization against float32 over46130797 valid pixels:0.1mm representation
depth MAE0.02506mm / max0.05007mm,XYZ mean0.02670mm / max0.05820mm;
1mm representation depth MAE0.25682mm / max0.50006mm,XYZ mean0.27404mm /
max0.58699mm. These are unit conversion errors, not codec loss.
1562003 pixels (about3.28%) exceed6.5535m and become invalid0 in uint16;
original far-plane background range215.09584m. Float32+zlib retains them exactly.
No synthetic noise added; compression ratios do not predict noisy real-camera
data without further measurement. Full details/timings:results.json/config.json.
