# E165 — Target-conditioned point-cloud flow matching

Status: 18k training and evaluation complete. This directory owns all variant code.

Data: unchanged E157, physically rerendered independently sized red/blue/green/yellow cubes, robot/chair/rug and original cam_front (RGB 320×240). Train1024, val128, five heldout splits128 each. Full-scene4096 XYZ with frozen cached SAM3 `vision_features`256×72×72 sampled per point; 4 task IDs, GT128 visible points/task. SAM3 weights unchanged. No GT masks, centers, pose or size as model input.

Architecture: original width128 MLP + 2-layer point TransformerEncoder and 2-layer128-query task-conditioned TransformerDecoder retained and trainable, yielding Z[32,128,128] for batch8 scenes/32 tasks. Replace XYZ MLP with width128 two-layer TransformerDecoder: each layer has full128-point self-attention, cross-attention to all128 Z tokens, FF384, heads4, dropout0. No point-index positional encoding. Linear velocity output FP32, zero initialized. SAM3 is frozen; all grounding/flow parameters train from scratch seed135, no warm start. Output128 points/task.

Flow: x0~N(0,I), x1=2*GT_m, exact unsquared Euclidean Hungarian assignment x0↔x1 (user selected OT pairing), t=1-(Beta(1.5,1)*.999+.001). xt=(1-t)x0+t*x1_matched; u=x1_matched-x0. **Only velocity MSE**, averaged across points/XYZ/tasks. No EMD/pair auxiliary training loss. This finite-set coupling minimizes unsquared distance, not squared-cost Wasserstein-2. GT/matching used only in supervision. Prior has physical sigma0.5m; fixed global isotropic factor2, no GT centering or per-axis normalization.

Augmentation: Haar SO(3) + uniform translation[-0.5,0.5]m, same transform for all scene/target points, per-point SAM features stay attached. Apply before fixed XYZ*2. No coordinate scaling augmentation; physical size variation already rendered in E157. Same data seed135 and augmentation seed137 as E161.

Train: 18,000 updates, batch8 scenes/32 task queries, AdamW3e-4→3e-5 cosine, warmup100, wd0.01, clip1. BF16 attention/MLP, FP32 parameters/optimizer, velocity output/MSE/state, TF32 off. About140.625 train-scene passes. Validate every500, select minimum sampled EMD on fixed128 E137 SE3 validation scenes; clean validation diagnostic. Different objective and output head from E161, so gains cannot be attributed solely to attention architecture.

Inference/evaluation: Gaussian→XYZ with20 positive Euler steps, divide output by2, compute Z once. Common fixed CPU noise seed165 (4 clouds reused across scenes), no GT information. For coordinate-SE3 tests rotate noise bank by R, no prior translation; report coupling explicitly, no claim of guaranteed equivariance. Canonical five heldout splits and four E137 IID SE3 transforms. Metrics: exact128-point EMD, directional CD, centroid/aligned CD, robust extents, 4mm-voxel visible-surface coverage and target selection. Baseline retained E161 best18k checkpoint. Save predictions and best-vs-GT viewer. No physical-scale paired intervention in this initial run.

Reference FM: `OATFlow/reach/flow.py` (time law, path convention, positive Euler); fresh point head returns positive velocity directly (no inherited pretrained action sign). Copied original grounding/data/SE3/metrics helper provenance in `sources.json`. Run on isolated branch `exp/e165-fm`; exact executing Git commit and source SHA256 recorded in each run's config/README. Artifacts stay outside checkout.

Commands (existing uv environment):
```
/workspace/arm-manipulation/.venv-concerto/bin/python /workspace/robotics_assets/experiments/e165_fm/code/run.py verify
/workspace/arm-manipulation/.venv-concerto/bin/python /workspace/robotics_assets/experiments/e165_fm/code/run.py benchmark
/workspace/arm-manipulation/.venv-concerto/bin/python /workspace/robotics_assets/experiments/e165_fm/code/run.py train
```
Training automatically evaluates the selected best checkpoint. Logs stream in attached `ssh_tmux:e165-log` and `/workspace/robotics_assets/experiments/e165_fm/run.log`.

Executing Git commit: `51a4a493b9fc14833b2e09d61a581ba61ea68078`. Exact hashes/configuration: `config.json`; benchmark: `../benchmark.json`.

Results: /workspace/robotics_assets/experiments/e165_fm/RESULTS.md

# E165 — Point-cloud Flow Matching

Status: complete. See train/config.json, evaluation/summary.json and code/README.md for exact configuration.

Data E157: train1024/val128; frozen SAM3 per-point features +4096 full-scene XYZ, 4 tasks,128 target points. Original grounding trainable; two-layer128-wide SA+CA velocity head. Full SE(3), no coordinate scale. Scratch18k, batch8 scenes, OT/Hungarian path + velocity MSE only. Inference20 Euler steps; fixed Gaussian seed165.

| Model / split | EMD mm | Aligned CD mm | pred→GT mm | GT→pred mm | Coverage@5mm % | Selection % |
|---|---:|---:|---:|---:|---:|---:|
| E165 / iid | 15.588 | 7.577 | 9.863 | 8.890 | 30.42 | 100.00 |
| E165 / SE3 | 16.348 | 7.627 | 9.957 | 9.575 | 28.58 | 99.85 |
| E165 / positions | 18.014 | 7.580 | 10.790 | 9.839 | 26.81 | 100.00 |
| E165 / rotations | 16.322 | 7.309 | 10.320 | 8.680 | 31.21 | 100.00 |
| E165 / sizes | 18.385 | 7.278 | 11.760 | 11.352 | 24.97 | 99.61 |
| E165 / combinations | 16.478 | 7.664 | 10.107 | 9.387 | 28.61 | 100.00 |
| E161 / iid | 11.574 | 6.210 | 8.293 | 6.164 | 51.01 | 100.00 |
| E161 / SE3 | 12.073 | 6.323 | 8.349 | 6.232 | 49.96 | 100.00 |
| E161 / positions | 12.999 | 6.302 | 8.852 | 6.646 | 46.89 | 100.00 |
| E161 / rotations | 12.386 | 6.194 | 8.642 | 6.191 | 51.02 | 100.00 |
| E161 / sizes | 15.316 | 6.397 | 10.955 | 8.068 | 42.94 | 99.80 |
| E161 / combinations | 11.618 | 6.298 | 8.283 | 6.217 | 50.48 | 100.00 |

Changed decoder and objective together; comparison does not isolate architecture. One fixed sampling-noise bank only. Ground-truth labels describe visible surfaces; no physical target-only resize intervention in this initial run.

Executing Git commit: `51a4a493b9fc14833b2e09d61a581ba61ea68078`. Checkpoint steps: {'E165': 17000, 'E161': 18000}.

Audited: checkpoint hashes, saved metrics/finite predictions,20 independent EMD assignments/model, identical GT/SE3 inputs, viewer HTTP200 (4.84MB). Paired128-scene bootstrap10k: E165 minus E161 IID EMD95CI[3.732,4.298]mm; SE3[4.029,4.528]mm. Training33.53min. See server audit.json.
