# E127 — Rear RGB-D camera on one saved pick episode

Status: COMPLETE.

Render a second fixed camera with the same intrinsics, height and downward pitch
as the existing front camera, moved to the opposite side of its look-at point.
Front world position[0.54,-0.66,0.60]m; rear[0.54,0.42,0.60]m. Shared look-at
[0.54,-0.12,0.035]m,42.65386164degree vertical FOV,320x240,25Hz. Rotation is180degree
yaw about that look-at point; no image flipping, crop, transparency or scene
geometry changes. The rear view can be occluded by the robot, intentionally.

Data: original E117 wide-pick episode g000/p00/t0,RedCube,138frames/5.52seconds.
Replay saved physical states only, no new simulation rollout, IK or control.
Copy original front RGB/depth unchanged; add rear RGB and ideal optical-Z depth.
Depth uint16,2mm/unit,invalid code0,FFV1 gray16le. Quantization error <=1mm;
FFV1 decoded-depth hash verified exact. No sensor noise/focus/range hardware
simulation. Native camera images stay upright and calibration-compatible.

Save front/rear intrinsics, per-frame camera-to-world/base poses and timestamps
in calibration.npz. Original front extrinsics are checked against source;
rear pose is checked constant. Save visible instance masks for both views.
Source scene/episode/calibration/video/mask/state hashes verified unchanged.

Artifacts: comparison.mp4 (front/rear RGB top, matched depth views bottom),
rgb_front.mp4,rgb_rear.mp4,depth_front.mkv,depth_rear.mkv,scene.xml,
calibration.npz,masks.npz,copied states/labels/supervision,config.json,
summary.json,verification.json,sample_{frame}.png and rear RGB/depth images.
Depth display uses a shared fixed0.25–1.25m turbo_r scale; zero/invalid is dark.
Display clipping/resizing affects visualization only, never stored depth.

Run `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.data_tools.rear_episode` in
repository tmux with stdout/stderr piped through tee. Code:
OATFlow/data_tools/rear_episode.py. Output:
/workspace/robotics_assets/experiments/e127_back. Exact hashes/settings:config.json.
No learned policy/checkpoint or training; H/K, flow steps, noise seed and training
epochs/LR are not applicable. The original expert uses absolute joint commands;
this render generates no new actions. Results complete.


## Results

Rendered138frames/5.52s. Both RGB views, both depth streams and comparison fully
decoded with matching frame counts/FPS. Rear FFV1 uint16 decoded hash matches the
pre-encoding codes exactly. Rear quantization MAE0.523mm,max1.00006mm. Calibration
has identical front/rear intrinsics and138synchronized poses in world/base frames.
Original source scene/state/RGB/depth/masks/calibration hashes are unchanged.

Initial visible object pixels(Red,Blue,Green,Yellow):front[257,379,282,277],
rear[247,75,239,220]. Visible>=8pixel frames:front[138,138,138,138],
rear[116,110,138,138]. Exact mirrored rear placement sees additional sides but
has real robot occlusion, especially Red/Blue; no attempt was made to optimize
or shift the camera after observing this. No reconstruction accuracy or grasp
improvement is claimed from one rendered episode.

Comparison:comparison.mp4. Native rear RGB:rgb_rear.mp4. Metric raw rear depth:
depth_rear.mkv;decode gray16le and multiply by0.002metres. Calibration:calibration.npz.
Frames0/99/137are also saved as comparison/rear RGB/rear depth preview PNGs.
