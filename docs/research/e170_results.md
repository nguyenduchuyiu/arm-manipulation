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
