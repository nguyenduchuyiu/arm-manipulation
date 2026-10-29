# E171 — Train Gaussian FM without SE(3)

Status: training/evaluation complete; see RESULTS.md.

Purpose: controlled E170 ablation. Train a fresh Gaussian FM on canonical scene coordinates only. Leave completed E170 code/checkpoints/cache untouched. Same model580483trainable parameters, frozen E168best16500 grounding862080parameters; task-conditioned Z128×128 + full point memory4096×128, two-layer128-wide4-head particle self-attention and condition cross-attention. No particle-index embedding, GT centroid/Rg, masks, explicit XYZ bypass or output normalization. Source Gaussian128×3 N(0,I)metres; output target128×3 XYZ.

Data: E157 unchanged RGB320×240/fullscene4096XYZ with robot/chair/rug,4colored cubes and128visible GT target points. Train1024scenes/4096tasks;128val,128each IID/positions/rotations/sizes/combinations. Only view0 (R=I,t=0) is used in training. Reuse E170 immutable frozen feature cache at `/workspace/robotics_assets/experiments/e170_memory/cache` read-only, no duplicate feature storage. Existing5views remain available for validation/test; SE3 retained as OOD evaluation. Frozen E168 encoder itself was originally SE3-trained; this ablates augmentation of the fresh FM/adapters, not encoder pretraining.

Training: fresh18000steps; B8scenes×4tasks=32queries; FP32 FM/attention/optimizer/MSE/Euler, TF32off; cacheFP16castFP32. Exact unsquared Euclidean Hungarian noise↔GT matching, only world velocity MSE(m²), linear CFM, timestep1−(Beta(1.5,1)×.999+.001). AdamW3e-4→3e-5cosine,100warmup,wd.01,clip1. Source E168best16500 SHA25665527d6657e57aef6552c01bf1d33c2c9014c88a08a4c332693342f71e776207, no trained FM weights loaded. Same E170 model/data seed170,noise167. Exact equality to E170 initial model asserted. Consume original view RNG draws then force0, preserving scene-ID sequence and downstream noise/timestep random streams. Canonical inputs repeat more often within fixed budget, an inherent consequence of removing augmentation.

Selection/test: canonical valview0 worldEMD every500steps chooses best, zero-velocity step0 eligible. SE3 valview1 is diagnostic only. Primary comparison is fixed E171last18000 vs E170last18000 to avoid different checkpoint-selection criteria (E170best selected by SE3val). Autoevaluate best/last on all canonical splits and4IIDSE3OODviews;128points,20Eulersteps,seed167fixedGaussian bank. Report EMD,centroid error,alignedChamfer,directionalChamfer,unitshapeEMD,robust XYZ extents,Rgerror,surfacevoxel4mm coverage@2/5mm,nearestGTcentroid targetselection. Same exact GT and per-query deltas checked for comparison. Single matched seed, no multi-run statistical claim.

Code is isolated on `exp/e171-nose3` from65c7f27, own selective E170 copies, source hashes in sources.json. No cross-experiment code imports. Model/flow byte equality to E170 checked; source commit and actual configuration saved in train/config.json; cache provenance remains E170b449074. Root verify.json records preflight source/cache hashes, exact initialstate, canonical sampling and fullGPU updates.

Run sequentially with `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py verify` then `... code/run.py train`; train autoevaluates. Dedicated attached tmux `ssh_tmux:e171-log`, same stdout/stderr saved to `/workspace/robotics_assets/experiments/e171_nose3/run.log`. Fresh train directory created by trainer. Best-vs-GT visualization will use existing forwarded `http://localhost:17070/e171.html`. No result claimed yet.

Artifact status: training/evaluation complete; see RESULTS.md. Executing source commit: `2a157f1162e04d8ea9f36209dd2b8a29609717c6`.

# E171 — No SE3 during FM training; frozen Z + full memory

Frozen E168best16500 encoder/task decoder. E170 cache reused read-only; training1024 scenes×canonical view0 only: shared memory4096×128 + task-conditioned Z128×128. Fresh FM/adapters, FP32,18k steps, only world velocity MSE with exact unsquared Hungarian pairing. Standard Gaussian128×3 → world target128×3; no GT centroid/Rg or output normalization.

| Method / split | EMD mm ↓ | Centroid mm ↓ | Rg error mm ↓ | Coverage5mm % ↑ | Selection % ↑ |
|---|---:|---:|---:|---:|---:|
| E171_best/train | 9.920 | 6.670 | 1.758 | 48.16 | 100.00 |
| E171_best/val | 10.505 | 7.526 | 1.835 | 45.62 | 100.00 |
| E171_best/iid | 10.883 | 7.832 | 1.819 | 44.20 | 100.00 |
| E171_best/SE3 | 312.919 | 312.399 | 5.778 | 1.58 | 43.80 |
| E171_best/positions | 14.679 | 12.634 | 1.993 | 33.90 | 100.00 |
| E171_best/rotations | 12.925 | 9.364 | 1.920 | 38.16 | 100.00 |
| E171_best/sizes | 14.098 | 11.245 | 3.003 | 38.09 | 99.80 |
| E171_best/combinations | 11.338 | 8.269 | 2.085 | 42.47 | 100.00 |
| E171_last/train | 10.309 | 7.186 | 1.868 | 49.16 | 100.00 |
| E171_last/val | 10.854 | 7.991 | 1.915 | 47.01 | 100.00 |
| E171_last/iid | 11.266 | 8.382 | 1.874 | 45.16 | 100.00 |
| E171_last/SE3 | 312.990 | 312.437 | 6.217 | 1.57 | 43.99 |
| E171_last/positions | 14.223 | 11.972 | 2.264 | 35.70 | 100.00 |
| E171_last/rotations | 13.632 | 10.117 | 1.997 | 38.27 | 100.00 |
| E171_last/sizes | 13.923 | 11.060 | 3.029 | 40.27 | 99.80 |
| E171_last/combinations | 11.304 | 8.308 | 1.934 | 44.78 | 100.00 |

Checkpoint steps: {'E171_best': 17500, 'E171_last': 18000}
Best selected by canonical validation view0 EMD every500. Primary controlled comparison uses last18000 for both E170 and E171, without checkpoint-selection differences.
Frozen E168 memory and task-conditioned Z only; no GT centroid/Rg, explicit XYZ bypass or masks in FM condition.
FM trained on canonical view0 only. Four cached SE3 test views are OOD; frozen source E168 encoder was originally trained with SE3.
Standard Gaussian N(0,I) in metres, fixed seed167, FP32,20 Eulersteps; no output projection.
Target selection: predicted centroid nearest to one of four GT128 centroids; shape/coverage evaluated separately.
Primary comparison: both last18000 checkpoints. E171 best uses canonical val whereas E170 best used SE3 val; best-vs-best has selection differences.
Source commit: `2a157f1162e04d8ea9f36209dd2b8a29609717c6`.

## Fixed18k comparison

| Run/split | EMD mm ↓ | AlignedCD mm ↓ | Unit-shape EMD ↓ | Coverage5mm % ↑ |
|---|---:|---:|---:|---:|
| E170_SE3_last/train | 11.434 | 5.821 | 0.4235 | 45.99 |
| E170_SE3_last/iid | 12.320 | 5.824 | 0.4242 | 43.00 |
| E170_SE3_last/SE3 | 14.637 | 6.919 | 0.4900 | 35.94 |
| E170_SE3_last/positions | 14.007 | 6.039 | 0.4289 | 37.14 |
| E170_SE3_last/rotations | 14.022 | 6.229 | 0.4681 | 38.51 |
| E170_SE3_last/sizes | 15.473 | 5.946 | 0.4388 | 38.52 |
| E170_SE3_last/combinations | 12.105 | 5.978 | 0.4295 | 42.83 |
| E171_noSE3_last/train | 10.309 | 5.053 | 0.3559 | 49.16 |
| E171_noSE3_last/iid | 11.266 | 5.066 | 0.3564 | 45.16 |
| E171_noSE3_last/SE3 | 312.990 | 9.150 | 0.5692 | 1.57 |
| E171_noSE3_last/positions | 14.223 | 5.235 | 0.3604 | 35.70 |
| E171_noSE3_last/rotations | 13.632 | 6.106 | 0.4582 | 38.27 |
| E171_noSE3_last/sizes | 13.923 | 5.048 | 0.3754 | 40.27 |
| E171_noSE3_last/combinations | 11.304 | 5.193 | 0.3610 | 44.78 |

One matched seed; frozen context source unchanged and already SE3-trained. Removing augmentation repeats canonical examples more often within the same18k budget. Shape conclusions should consider alignedCD/unitshape/coverage, not only worldEMD.

Completion audit: trained/evaluated18000steps; both checkpoint hashes pass; independent Hungarian on512IIDlast queries11.266mm matches summary; GT exact E170 array. Viewer localhost:17070/e171.html HTTP200,4865369bytes. Train elapsed39.129min, peak7.445GiB.
