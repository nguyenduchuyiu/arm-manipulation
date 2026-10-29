# E128 downward top RGB-D camera

Status: COMPLETE.

Replay the E127 source episode g000/p00/t0 (RedCube), 138 saved states at
25 Hz. Add a fixed camera at world [0.54, -0.10, 1.20] m, looking vertically
down at the table center. Image right is +world X; image up is +world Y.
Retain the front pinhole intrinsics and FOV (42.65386164 degrees), 320x240.
Height 1.20 m fits the 1.10x0.86 m tabletop inside the field of view.

Source data: /workspace/robotics_assets/datasets/pick_wide_rgbd.
Output: /workspace/robotics_assets/experiments/e128_top.
No model training, source weights, frozen/trainable modules, epochs, batch,
learning rate, action generation, policy checkpoint, H/K or flow settings
apply. Stored states are replayed without physics stepping or new actions.

Save front/top RGB, optical-Z depth as uint16 FFV1 (2 mm/unit, zero invalid),
visible instance masks and framewise camera-to-world/base calibration.
Use the same 0.25-1.25 m depth display range for both views. The depth is ideal
simulated geometry, with no real sensor range/noise model. The original
dataset and E127 artifacts remain unchanged.

Run `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.data_tools.top_episode`
in tmux with stdout/stderr streamed through tee. Main artifacts: comparison.mp4,
sample_000.png, sample_099.png, sample_137.png, rgb_top.mp4, depth_top.mkv,
calibration.npz, masks.npz, scene.xml, config.json and verification.json.
Verify unchanged source hashes, exact decoded depth codes, calibration,
and complete decoding of all 138 video frames. Results below.


Results: all 138 frames decode successfully for both RGB/depth pairs and the
comparison video. Each of the four objects remains visible (>=8 pixels) in
all 138 top-view frames of this episode. This threshold does not imply full
object visibility; top-view initial areas are only 80/86/86/85 pixels
(Red/Blue/Green/Yellow), versus 257/379/282/277 in the front view.
Top depth quantization MAE is 0.1950 mm and maximum error 1.00011 mm.
Exact decoded-depth hashing, unchanged source hashes, vertical optical axis,
extrinsics and projected tabletop-corner coverage all pass.

The top view gives clear tabletop coverage in this one episode, but has
fewer object pixels and less side-surface geometry. It does not establish
dataset-wide robustness or improved localization/manipulation.
