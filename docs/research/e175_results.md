# E175 — Equivariant encoder + direct coordinate decoder, original scenes50k

Frozen original SAM3 point features; fresh trainable scalar/vector encoder and invariant task-query decoder with equivariant coordinate head. No SE3/scale augmentation.50kupdates,EMDonly,4096fullsceneinputs→128targetpoints.18kmilestone saved.

| Model/split | EMD mm | Centroid mm | AlignedCD mm | Coverage5mm % |
|---|---:|---:|---:|---:|
| E175_18k/train | 3.094 | 0.684 | 2.295 | 94.58 |
| E175_18k/iid | 4.451 | 2.133 | 2.770 | 91.94 |
| E175_18k/SE3 | 4.451 | 2.133 | 2.770 | 92.13 |
| E175_18k/positions | 5.028 | 2.588 | 3.061 | 88.86 |
| E175_18k/rotations | 6.337 | 2.703 | 4.078 | 69.91 |
| E175_18k/sizes | 5.562 | 3.097 | 3.088 | 84.49 |
| E175_18k/combinations | 4.660 | 2.271 | 2.862 | 91.16 |
| E175_best/train | 3.476 | 1.028 | 2.439 | 93.90 |
| E175_best/iid | 4.419 | 2.047 | 2.779 | 91.75 |
| E175_best/SE3 | 4.418 | 2.047 | 2.779 | 91.87 |
| E175_best/positions | 4.962 | 2.503 | 3.056 | 89.00 |
| E175_best/rotations | 6.320 | 2.738 | 4.076 | 69.81 |
| E175_best/sizes | 5.624 | 3.174 | 3.102 | 84.27 |
| E175_best/combinations | 4.597 | 2.199 | 2.852 | 91.14 |
| E175_last/train | 2.333 | 0.224 | 1.926 | 94.54 |
| E175_last/iid | 4.791 | 2.275 | 2.871 | 90.14 |
| E175_last/SE3 | 4.791 | 2.275 | 2.871 | 90.37 |
| E175_last/positions | 5.479 | 2.828 | 3.207 | 86.65 |
| E175_last/rotations | 6.826 | 3.163 | 4.223 | 68.16 |
| E175_last/sizes | 5.910 | 3.321 | 3.230 | 82.56 |
| E175_last/combinations | 5.041 | 2.475 | 3.002 | 88.79 |
| E174_last18k/train | 5.768 | 3.492 | 3.305 | 77.38 |
| E174_last18k/iid | 7.766 | 5.999 | 3.446 | 62.83 |
| E174_last18k/SE3 | 345.555 | 345.221 | 10.346 | 0.57 |
| E174_last18k/positions | 11.332 | 9.936 | 3.798 | 43.28 |
| E174_last18k/rotations | 11.190 | 8.162 | 5.409 | 40.46 |
| E174_last18k/sizes | 10.737 | 9.025 | 3.556 | 48.66 |
| E174_last18k/combinations | 8.095 | 6.287 | 3.594 | 60.24 |

## Pointwise coordinate equivariance

| Model | RMS mm | Max mm |
|---|---:|---:|
| E175_18k | 0.001170 | 0.005730 |
| E175_best | 0.001178 | 0.005779 |
| E175_last | 0.001095 | 0.005700 |

All train inputs canonical only; SO3/translation used for tests, never updates.
SameRGB/SAM3/pixelgraph under global coordinate-frame changes; physical target-only rotations are a different test.
Coordinate constraints do not guarantee reconstruction quality. FP32 numerical error reported pointwise, no GT alignment.
New3Dencoder is trainable fromscratch; E174 used a pretrained frozen non-equivariant encoder. Different architecture/pretraining/budget;18kmilestone has a differentLRhorizon.
Axis-aligned voxel coverage may vary under rotation; pointwise drift/EMD are primary equivariance checks.
No GTcenter,size,masks orcameraextrinsics as modelinputs; GTvisible128points onlysupervision/metrics.

Checkpointsteps: {'E175_18k': 18000, 'E175_best': 12000, 'E175_last': 50000}
Codecommit `552a2a353aaae14f3b2e8faa90fb6a20d5781a41`.
