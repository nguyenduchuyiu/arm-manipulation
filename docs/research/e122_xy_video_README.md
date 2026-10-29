# E122: outside-X failure videos

Status: COMPLETE.

Replay two BlueCube failures from the completed paired XY test: x_low/t1/outside at X0.27m and x_high/t1/outside at X0.81m, beyond pick-training X[0.30,0.78]m. Selection fixed before rendering: one same-color example per tested edge. These are diagnostic examples, not a new success-rate estimate.

Training unchanged: E122 final step23520,2048demos/32layouts,stride1,batch128,23520updates (~10.217epochs),frozen ViT-B/16,trained geometry/query/cloud/FM,FM initialized from E118 reach. Exact optimizer/LR/action configuration is in the parent config and copied into config.json. No training in this render.

Original rollout: CPU FP32,H25/K15,Euler10,seed0,300frames,25Hzwaypoints/50Hzservo/500Hzphysics. Replay saved initial+300qpos frames at25FPS (12.04seconds). No new physics or action sampling. Recompute auxiliary cloud at actual pre-action replans0,15,...,285 with the same checkpoint on CPU FP32; hold between replans. Re-rendered float32states can introduce small raster differences. Target mask/GT are for diagnostics only; policy uses scene RGB-D/calibration,target ID,joints and the decoder hidden.

Panels: front camera with orange GT visible target cloud and cyan auxiliary predicted cloud; oblique robot/table replay and EE path; target-cloud zoom; localization/contact/lift measurements. Surface Chamfer/centroid errors are not object-center XYZ errors. Preserve and verify checkpoint/results/trajectory hashes; fully decode each video; verify cloud cadence. Files:config.json,progress.json,verification.json,contact_sheet.png,fail_x_low.mp4,fail_x_high.mp4,and diagnostic arrays/metrics. Reproduce with MUJOCO_GL=egl .venv/bin/python -m OATFlow.top_pick.xy_video in tmux, tee metadata/e122_xy_video.log. The fresh output is test_xy/videos_1.

Both videos fully decoded;301frames each;all source hashes unchanged and20cloud replan frames verified per clip.
