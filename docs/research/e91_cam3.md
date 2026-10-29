# e91_cam3

Status: complete; all3,840 episodes rendered and audited. Logs: workspace_analysis/logs/e91_cam3.log and e91_audit.log.
Initial12 workers; resumed with24 after1,482 checked complete episodes to
measure throughput. Render settings unchanged; incomplete streams rerendered.
CPU affinity144, actual cgroup quota15.36CPU cores; encoding is CPU limited.
Returned to12 workers because24 reduced head-training and render throughput.

Render all 3,840 e55_pick episodes (544,324 synchronized frames) through the
three calibrated e90 cameras. Original NexArm meshes, OATFlow lighting, saved
full simulator states, unchanged trajectories. RGB 320x320, 25Hz; top85cm,
left/right65cm. No training or learned rollout in this export. No action change.

Artifacts: `/workspace/datasets/e91_cam3`. Exact rig/calibration in scene.xml
and calibration.json; source metadata/config.json; source-relative episode
directories contain rgb_top/left/right.mp4 and labels.npz. Labels are actual
centers in metres, all four object IDs at each frame. Base origin in calibration,
axes aligned to world. Source state hash and decode checks in complete.json.
Rendering is resumable only via completed, checked episode markers.

The e92 training subset retains all 24 permutations and all four query
trajectories from 8 original train groups, 2 validation groups and 2 test groups.
Groups and paths in subset.json. Rendering prioritizes these episodes.

Run: `MUJOCO_GL=egl PYTHONPYCACHEPREFIX=/tmp/e91_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u workspace_analysis/render_cam3_dataset.py --workers 12`
Resume: same command with `--resume`. Full progress: progress.json.
Code: workspace_analysis/render_cam3_dataset.py. Calibration checked against
rendered fiducials; every stream checked with full decode, dimensions, FPS,
frame count. Full source hash/label/ID/timestamp audit passed: audit.json.

Results:544,324synchronized frames,11,520fully decoded RGB streams, train/val/test3,072/384/384episodes.
Video bytes1,709,406,799 (~1.59GiB). All3,840source state SHA256 hashes match.
Maximum base-label rounding error1.49011e-8m. Intrinsic/extrinsic projection fiducial checks pass.

Preview: preview.mp4 (4episodes, all4target IDs, train/val/test); source refs in preview.json.
Audit command: `MUJOCO_GL=egl PYTHONPYCACHEPREFIX=/tmp/e91_cache .venv/bin/python workspace_analysis/audit_cam3_dataset.py`.
