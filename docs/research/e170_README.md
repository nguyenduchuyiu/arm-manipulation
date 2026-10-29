# E170 — Gaussian FM with frozen target Z and full point memory

Status: training/evaluation complete; see RESULTS.md.

Change: source128×3 is independent standard Gaussian N(0,I) in metres. Frozen E168best16500 SAM3/point encoder/task decoder produces memory4096×128 and task-conditioned Z128×128. FM cross-attends to concatenated adapted Z+memory (4224 tokens), self-attends among128 particle tokens, predicts128×3 velocity. Shared XYZ/time projection, no learned particle-index embedding. No GT center/Rg, masks, pose, shape parameters or explicit XYZ bypass in FM conditioning. No output centering/scaling or sphere projection. Task enters through Z only.

Data: immutable E157 rendered RGB320×240, fullscene4096 XYZ with robot/chair/rug, four colored cubes;1024 train,128 val,128 each IID/positions/rotations/sizes/combinations. GT128 visible target points per task. Exact E169 canonical+4 fullSO3/translation views retained; transform scene/GT together, preserve feature-point association. Freeze/cache encoder separately for every view; never rotate hidden features. Five finite views, not online unlimited SE3. Cache memory shared across4tasks plus Z per task, FP16 storage≈9.84GiB; extracted with original E168 BF16 autocast, tested against original FP32 output with rtol5e-4/atol6e-8 rounding bound. Copied labels contain GT statistics for metrics only.

Weights: `/workspace/robotics_assets/experiments/e168_normfm/train/best.pt`,step16500,SHA256 `65527d6657e57aef6552c01bf1d33c2c9014c88a08a4c332693342f71e776207`. Source862080 grounding parameters frozen; SAM3 features already immutable. New FM and memory/Z adapters freshly initialized seed170; exact trainable count in verify.json/config.json. No trained flow or center/Rg head loaded.

Train:18,000steps, B8scenes×4tasks=32targets; FP32 parameters/attention/AdamW/MSE/Euler, TF32off. Only velocity MSE in m². Exact unsquared Euclidean Hungarian source↔GT assignment128points, straight CFM path, timestep1−(Beta(1.5,1)×.999+.001). AdamW3e-4→3e-5 cosine,100warmup,wd.01,clip1. Data/modelseed170; Gaussianseed167. Match GT in augmented world coordinates. No EMD/pair/geometry auxiliary losses.

Validation:128 valscenes×4tasks, fixed fullSE3view1 every500steps; select minimum world EMDmm; zero-velocity step0 eligible, last18k also retained.20Eulersteps. Fixed Gaussian bank seed167, independently distributed per task; rotated for each view, origin unchanged. Canonical val diagnostic also recorded. Test fulltrain/val/5OOD canonical and4IIDSE3views. EMD, both Chamfer directions, centroid/alignedCD, robust XYZ extent,Rg error, visible-surface coverage2/5mm,targetselection (predicted centroid nearest of4GT128centroids). No strict stochastic equivariance claim from these fixed-origin noise evaluations. E169 oracle normalized sphere run is not a controlled baseline.

Code: dedicated worktree branch `exp/e170-memory`; commit recorded alongside artifacts. Selective own copies and hashes in sources.json; no imports from other experiment code. Cache→verify→train/evaluate in attached `ssh_tmux:e170-log`, stdout+stderr saved to `/workspace/robotics_assets/experiments/e170_memory/run.log`. Cache stores immutable source features only, no training gradients; KV projections recalculated differentiably each optimizer batch, reused across20samplingsteps.

Run with `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py {cache,verify,train,evaluate}` sequentially from experiment root. Trainer creates fresh train directory. Auto-evaluation after18k; best-vs-GT viewer published at existing `http://localhost:17070/e170.html` on completion. No result claimed yet.

Artifact status: training/evaluation complete; see RESULTS.md. Executing source commit: `b449074087e643ec367cec5fb79596c8a3ed09cd`.

# E170 — Gaussian FM conditioned on frozen Z and full memory

Frozen E168best16500 encoder/task decoder. Cache1792 scenes×5 SE3 views: shared memory4096×128 + task-conditioned Z128×128. Fresh FM/adapters, FP32,18k steps, only world velocity MSE with exact unsquared Hungarian pairing. Standard Gaussian128×3 → world target128×3; no GT centroid/Rg or output normalization.

| Method / split | EMD mm ↓ | Centroid mm ↓ | Rg error mm ↓ | Coverage5mm % ↑ | Selection % ↑ |
|---|---:|---:|---:|---:|---:|
| E170_best/train | 11.801 | 8.055 | 2.282 | 45.39 | 100.00 |
| E170_best/val | 12.231 | 8.754 | 2.308 | 43.52 | 100.00 |
| E170_best/iid | 12.713 | 9.242 | 2.332 | 42.41 | 100.00 |
| E170_best/SE3 | 14.140 | 9.988 | 2.519 | 36.50 | 99.95 |
| E170_best/positions | 14.017 | 10.749 | 2.454 | 37.10 | 100.00 |
| E170_best/rotations | 14.023 | 10.512 | 2.329 | 38.60 | 100.00 |
| E170_best/sizes | 15.819 | 12.123 | 4.514 | 37.63 | 99.80 |
| E170_best/combinations | 12.571 | 8.792 | 2.390 | 41.94 | 100.00 |
| E170_last/train | 11.434 | 7.541 | 2.216 | 45.99 | 100.00 |
| E170_last/val | 11.864 | 8.250 | 2.261 | 43.85 | 100.00 |
| E170_last/iid | 12.320 | 8.670 | 2.228 | 43.00 | 100.00 |
| E170_last/SE3 | 14.637 | 10.512 | 2.327 | 35.94 | 99.90 |
| E170_last/positions | 14.007 | 10.762 | 2.392 | 37.14 | 100.00 |
| E170_last/rotations | 14.022 | 10.539 | 2.321 | 38.51 | 100.00 |
| E170_last/sizes | 15.473 | 11.737 | 4.368 | 38.52 | 99.80 |
| E170_last/combinations | 12.105 | 8.229 | 2.274 | 42.83 | 100.00 |

Checkpoint steps: {'E170_best': 17500, 'E170_last': 18000}
Best selected by world EMD on fixed validation SE3 view1 every500 steps. Initial zero-velocity model step0 eligible; last18k reported separately.
Frozen E168 memory and task-conditioned Z only; no GT centroid/Rg, explicit XYZ bypass or masks in FM condition.
Canonical+4 cached full SE3 views, not unlimited online augmentation.
Standard Gaussian N(0,I) in metres, fixed seed167, FP32,20 Eulersteps; no output projection.
Target selection: predicted centroid nearest to one of four GT128 centroids; shape/coverage evaluated separately.
E169 used oracle centroid/Rg and normalized sphere source; it is not a controlled baseline for this GT-free run.
Source commit: `b449074087e643ec367cec5fb79596c8a3ed09cd`.

Completion audit (2026-10-28): status trained/evaluated,18000steps; independent exact Hungarian EMD over512 IID queries12.713074mm and centroid9.242373mm match summary; both saved checkpoint hashes match. Viewer localhost:17070/e170.html HTTP200,4865259bytes. Train elapsed39.223min, peak allocated7.445GiB.
