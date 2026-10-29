# E175 — Cross-scene SAM3 feature swap

Completed frozen inference using best checkpoint step12,000, SHA256 `aebc05fc6e3afff34d28fd283b12a65e1ca97f82666c6fbcfe5d56affd206a52`. No training or model changes. Diagnostic source branch `diag/e175-swap`, commit `0212ee7419d261c3958a6f7795409d31a088c1d1`; owned code at `/workspace/robotics_assets/experiments/e175_equiv/test_swap/code`. Data/model/training remain E175: 1,024 train scenes, canonical inputs/no SE(3) augmentation, 4,096 full-scene XYZ, raw frozen SAM3 point features, task ID, 128 predicted points, 772,018 trained equivariant encoder/decoder parameters, 50k updates, B8 scenes, AdamW3e-4→3e-5, exact unsquared Hungarian EMD. Best remains 12k.

## Test configuration

All 128 IID scenes × 4 tasks = 512 queries per condition. Donor A = recipient B xor1: 64 disjoint pairs evaluated in both directions. XYZ, original recipient UV graph, task ID, and evaluation GT remain from B. All model parameters frozen, FP32/TF32off, no inference normalization/alignment. Four modes: own aligned features, donor features by raw point index, donor features matched by nearest image pixel to each recipient point, own features shuffled over points (seed17562). No GT used by inference. UV transfer keeps image coordinates meaningful despite independently sampled point indices; it does not create a physically consistent observation or preserve object identity between scenes.

## Results

| Features with recipient B XYZ | EMD mm | Centroid error mm | Visible-surface coverage@5mm % | Nearest-centroid selection % | Aligned CD mm |
|---|---:|---:|---:|---:|---:|
| B aligned (baseline) | 4.419 | 2.047 | 91.754 | 100.00 | 2.779 |
| A by point index | 410.983 | 403.696 | 0.012 | 24.61 | 52.299 |
| A by nearest UV | 181.209 | 180.499 | 0.951 | 31.84 | 11.803 |
| B shuffled across points | 396.582 | 389.055 | 0.094 | 24.41 | 52.200 |

Raw cross-scene swap raises mean EMD 93.01×; UV transfer still raises it 41.01×. Every one of 512 queries in both swapped modes is worse than baseline by over 2×. Own-scene shuffling gives 89.75× mean EMD. UV transfer distances: mean2.103px, median2px, p954.123px, max12px (320×240 source image). Self UV transfer is exactly identity. Baseline predictions are bit-exact to the previously audited full IID best predictions (max drift0mm). Target selection is a forced nearest-centroid four-way decision; it can select the nominal target while prediction is far from all objects, so EMD/coverage are the primary failure evidence.

The UV-matched result is still strongly broken, so raw-array sampling mismatch alone does not explain failure. Model depends on correct visual feature–geometry association rather than being insensitive to SAM3 feature replacement. This counterfactual does not by itself demonstrate correct metric-shape grounding, semantic-only feature use, or immunity to spatial shortcuts. Swapped inputs intentionally conflict, and fixed dataset backgrounds do not test robustness to unseen clutter.

## Artifacts and checks

- Run: `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py` from testroot (creates fresh results; do not rerun over saved output). Attached tmux `ssh_tmux:e175-swap`, stdout/stderr mirrored to `run.log`.
- Exact config/checkpoint/source IDs, aggregate and per-task metrics: `/workspace/robotics_assets/experiments/e175_equiv/test_swap/results/summary.json`.
- Per-query paired metrics: `results/metrics.json`; all raw clouds, donor mapping, UV matching indices/distances and permutation indices: `results/predictions.npz`.
- Scene B0/A1 original RGB: `results/recipient_B.png`, `results/donor_A.png`. Static `results/swap.png` and interactive four-task `results/swap.html`, no corrections.
- Viewer: http://localhost:17070/e175_swap.html (HTTP200 verified). Static plot visually inspected; GT black, baseline blue, index swap orange, UV swap magenta.
- Source hashes match checkpoint model; checkpoint step12k checked; all 4×128×4×128 predictions finite; self UV mapping identity; baseline exact. All inputs/graph/tasks except specified features held fixed.
