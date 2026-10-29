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
