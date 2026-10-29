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
