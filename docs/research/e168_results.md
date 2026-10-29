# E168 — Point-cloud Flow Matching

Status: complete. See train/config.json, evaluation/summary.json and code/README.md for exact configuration.

Data E157: train1024/val128; frozen SAM3 per-point features +4096 full-scene XYZ, 4 tasks,128 target points. All grounding/center/Rg/flow heads scratch-trained; two-layer128-wide SA+CA velocity head. Full SE(3), no coordinate scale. Scratch18k, batch8 scenes, Predicted centroid+Rg sphere, Unit-shape EMD/Hungarian FM, endpoint centroid0/Rg1, world output=predictedcentroid+predictedRg*shape; dimensionless velocity/center/Rg losses1/1/1 (regression reference20mm). Inference20 Euler steps; fixed sphere-direction seed167; Gaussian baseline seed165.

| Model / split | EMD mm | Centroid mm | Rg MAE mm | Unit-shape EMD | Aligned CD mm | Coverage@5mm % | Selection % |
|---|---:|---:|---:|---:|---:|---:|---:|
| E168 / iid | 12.749 | 7.284 | 0.969 | 0.498 | 7.527 | 31.94 | 100.00 |
| E168 / SE3 | 12.701 | 7.689 | 0.956 | 0.489 | 7.352 | 34.46 | 100.00 |
| E168 / positions | 14.503 | 9.979 | 1.685 | 0.492 | 7.696 | 29.39 | 100.00 |
| E168 / rotations | 13.875 | 9.418 | 1.425 | 0.511 | 7.408 | 31.81 | 100.00 |
| E168 / sizes | 15.246 | 10.557 | 1.639 | 0.503 | 7.137 | 33.85 | 99.80 |
| E168 / combinations | 12.592 | 6.858 | 1.046 | 0.497 | 7.637 | 31.28 | 100.00 |
| E168_prior / iid | 12.589 | 7.284 | 0.969 | 0.491 | 7.540 | 32.13 | 100.00 |
| E168_prior / SE3 | 12.724 | 7.689 | 0.956 | 0.491 | 7.504 | 33.23 | 100.00 |
| E168_prior / positions | 14.381 | 9.979 | 1.685 | 0.487 | 7.748 | 28.98 | 100.00 |
| E168_prior / rotations | 13.819 | 9.418 | 1.425 | 0.512 | 7.548 | 31.34 | 100.00 |
| E168_prior / sizes | 15.138 | 10.557 | 1.639 | 0.496 | 7.152 | 33.85 | 99.80 |
| E168_prior / combinations | 12.374 | 6.858 | 1.046 | 0.487 | 7.606 | 31.91 | 100.00 |
| E167 / iid | 13.539 | 9.068 | 1.028 | 0.491 | 7.482 | 30.56 | 100.00 |
| E167 / SE3 | 14.129 | 9.860 | 1.025 | 0.491 | 7.480 | 30.62 | 100.00 |
| E167 / positions | 15.749 | 12.203 | 1.737 | 0.487 | 7.693 | 27.78 | 100.00 |
| E167 / rotations | 14.783 | 10.861 | 1.425 | 0.512 | 7.491 | 29.53 | 100.00 |
| E167 / sizes | 16.694 | 12.856 | 1.760 | 0.496 | 7.041 | 27.79 | 99.80 |
| E167 / combinations | 13.431 | 8.781 | 1.097 | 0.487 | 7.519 | 29.89 | 100.00 |
| E165 / iid | 15.588 | 10.650 | 2.450 | 0.571 | 7.577 | 30.42 | 100.00 |
| E165 / SE3 | 16.348 | 11.644 | 2.593 | 0.575 | 7.627 | 28.58 | 99.85 |
| E165 / positions | 18.014 | 14.136 | 2.567 | 0.564 | 7.580 | 26.81 | 100.00 |
| E165 / rotations | 16.322 | 12.213 | 2.236 | 0.567 | 7.309 | 31.21 | 100.00 |
| E165 / sizes | 18.385 | 13.989 | 4.067 | 0.591 | 7.278 | 24.97 | 99.61 |
| E165 / combinations | 16.478 | 11.847 | 2.422 | 0.572 | 7.664 | 28.61 | 100.00 |
| E161 / iid | 11.574 | 7.366 | 2.210 | 0.457 | 6.210 | 51.01 | 100.00 |
| E161 / SE3 | 12.073 | 7.966 | 2.153 | 0.463 | 6.323 | 49.96 | 100.00 |
| E161 / positions | 12.999 | 9.548 | 2.197 | 0.452 | 6.302 | 46.89 | 100.00 |
| E161 / rotations | 12.386 | 8.592 | 1.858 | 0.469 | 6.194 | 51.02 | 100.00 |
| E161 / sizes | 15.316 | 11.394 | 3.471 | 0.466 | 6.397 | 42.94 | 99.80 |
| E161 / combinations | 11.618 | 7.417 | 2.349 | 0.456 | 6.298 | 50.48 | 100.00 |

E168 enforces strict centroid/Rg/shape factorization; geometry losses use fixed20mm reference. E167 is the world-space counterpart, with its actual checkpoint/budget recorded. E165/E161 use different prior/loss/head. These are not single-factor comparisons. One fixed sampling-noise bank only. Ground-truth labels describe visible surfaces; no physical target-only resize intervention in this initial run.

Executing Git commit: `6487c2c314ca7933fd0521f6330b65961c41f24a`. Checkpoint steps: {'E168': 16500, 'E167': 14500, 'E165': 17000, 'E161': 18000, 'E168_prior': 16500}.


Completion audit and interpretation:
- Scratch18k complete, best16500, training24.58min. Fixed20Eulerstep/seed167 evaluation complete; checkpoints/finite clouds/summary means/independent EMD checks passed.
- IID output EMD12.749mm versus its own sphere prior12.589mm; paired128-scene bootstrap95CI for output-minus-prior[0.134,0.186]mm. Unit-shape EMD0.49771 versus sphere0.49079. No shape benefit on IID.
- SE3 output EMD12.701mm versus prior12.724mm; delta95CI[-0.0481,0.00049]mm includes0. No clear EMD improvement. Coverage34.46% versus prior33.23%.
- Centroid/Rg heads: IIDcentroid7.284mm, RgMAE0.969mm/R²0.934; SE3centroid7.689mm,RgMAE0.956mm/R²0.9365. Predicted centroid/Rg preserved from prior to output (maxdrifts0.00000235/0.00000321mm). Selection100% on IID/SE3.
- E16118k remains better: IIDEMD11.574mm/coverage51.01%, versus E16812.749mm/31.94%. E167 reference was snapshotted while training at14900, best14500; not a completed18k comparison.
- Particle-index requirement verified: shared XYZ/time projections, no particle-index embedding, dense SA and CA to Z. Learned grounding queries are condition tokens, not particle-index embeddings.
- Read-only permutation test, best16500, IIDscenes0/1×4tasks,20steps: FP32velocitymaxerror2.03e-6, worldpointmaxdrift0.000156mm. Reordering Z alone also invariant within FP32precision. BF16 endpoint sensitivity measured: maxpointdrift1.089mm (rawunitRMS~1.005, not a near-zero normalization collapse). No claim of exact BF16numerical invariance. Test source and failed original BF16tolerance trace retained under test_perm; no model changes.
