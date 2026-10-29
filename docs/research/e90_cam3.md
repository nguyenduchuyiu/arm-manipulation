# e90_cam3

Status: complete. Preview rendering only; no training, checkpoint, optimizer,
frozen/trainable module, epoch, batch, LR, policy H/K or flow sampling applies.

Re-render4existing e55_pick expert episodes using their recorded full qpos/qvel
at exactly original timestamps and frame count.320x320RGB per camera,25Hz,
three fixed external views; no new trajectories or modifications to original data.
Episodes:g000/p00_t0,g011/p14_t1,g036/p11_t2,g039/p12_t3, covering4targets and
slots0/1/2/3,2train and2held-out test layouts. Original target success is dataset
metadata, not a new learned rollout result. Verify source normalized joint states
against recorded physical qpos and reconstructed robot actuator limits.

Workspace design box:world X[0.29,0.79],Y[-0.37,0.13],Z[0,0.30]m,
center[0.54,-0.12,0.15],tabletopZ0. Top camera[0.54,-0.12,0.85],looking straight
down,FOV55deg. Oblique left[0.09,-0.55,0.65],right[0.99,-0.55,0.65],both aimed
at workspace volume center,FOV65deg. Obliques tilted51.224deg from vertical,
fixed outside workspace/front table edge. Full box must fit all three frusta.

Use full original OATFlow environmental geometry, original NexArm STL meshes
and original2scene lights plus headlight0.3/0.4/0.15, without flat headlight
replacement. Reconstruct configured robot from serialized e86 model and original
OATFlow scene worldbody, with the same50qpos/44qvel saved state convention,
four boxes and2parked covers. New camera definitions affect rendering only.

Calibration: MuJoCo ideal pinhole intrinsics,zero lens distortion,OpenCV camera
axes(+Xright,+Ydown,+Zforward); export camera_from_world/world_from_camera and
camera_from_base/base_from_camera. Base origin[0.539369022999,0.0639727628473,0],
axes aligned with world. Pixel centers use integer coordinates0..319; principal
point159.5,159.5. Export K, resolution,FOV,cam position/quaternion,aim,tilt,height.
Validate world-camera roundtrip, workspace corner projections and rendered
fiducial centroid reprojection. Three streams use identical qpos/timestamps.

Output:workspace_analysis/results/e90_cam3. Each case has3raw320x320videos,
combined960x416preview,original frame indices/timestamps and README/config.
Combined all-episode comparison.mp4,preview contact sheet,camera_layout.png,
scene.xml,calibration.json/calibration.npz,calibration_checks.json,
config.json,summary.json/provenance hashes. Original e55 videos stay intact.
Tmux:huy:cam3; log:workspace_analysis/logs/e90_cam3.log.
Run from repository root:

```bash
MUJOCO_GL=egl PYTHONPYCACHEPREFIX=/tmp/e90_cam3_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u workspace_analysis/rerender_cam3.py
```

All state, camera, calibration and video checks passed; summary.json.


Completed:4expert episodes,563synchronized frames,12rawRGB streams,320x320at25Hz. All original normalized joint observations match reconstructed qpos exactly(max difference0). All3cameras cover8corners of the design box. Rendered9fiducials per camera; reprojection max0.437258px. Intrinsics/extrinsics exported in calibration.json/calibration.npz. All per-stream frame counts/resolutions/FPS and combined MP4 decode pass.
