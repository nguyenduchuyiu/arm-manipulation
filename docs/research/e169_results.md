# E169 — Oracle centroid/Rg, frozen cached Z, fresh flow-only training

GT centroid/Rg are supplied to every method. Geometry diagnostic, not GT-free inference. Source E168best16500 frozen; cache1792scenes×5views×4tasks. Only fresh513923parameter PointFlow trained18k with velocityMSE, FP32. No center/size losses or upstream gradient. Source/source-cache/checkpoint hashes in configs.

| Method / split | Unit-shape EMD ↓ | Oracle EMD mm ↓ | CD mm ↓ | Coverage5mm % ↑ |
|---|---:|---:|---:|---:|
| oracle_sphere/train | 0.4900 | 9.883 | 7.401 | 38.59 |
| oracle_sphere/val | 0.4925 | 9.780 | 7.319 | 39.08 |
| oracle_sphere/iid | 0.4908 | 9.936 | 7.430 | 38.58 |
| oracle_sphere/SE3 | 0.4908 | 9.936 | 7.430 | 38.63 |
| oracle_sphere/positions | 0.4867 | 9.897 | 7.415 | 38.30 |
| oracle_sphere/rotations | 0.5117 | 9.958 | 7.476 | 38.39 |
| oracle_sphere/sizes | 0.4956 | 9.341 | 7.009 | 56.28 |
| oracle_sphere/combinations | 0.4868 | 10.059 | 7.497 | 37.71 |
| E169_best/train | 0.3780 | 7.618 | 5.426 | 56.88 |
| E169_best/val | 0.3830 | 7.589 | 5.401 | 56.63 |
| E169_best/iid | 0.3787 | 7.642 | 5.444 | 56.90 |
| E169_best/SE3 | 0.4972 | 10.060 | 7.255 | 40.57 |
| E169_best/positions | 0.3865 | 7.855 | 5.604 | 54.82 |
| E169_best/rotations | 0.4553 | 8.878 | 6.259 | 48.22 |
| E169_best/sizes | 0.3939 | 7.327 | 5.240 | 65.02 |
| E169_best/combinations | 0.3811 | 7.869 | 5.590 | 54.85 |
| E169_last/train | 0.3985 | 8.031 | 5.898 | 50.99 |
| E169_last/val | 0.4023 | 7.982 | 5.849 | 51.36 |
| E169_last/iid | 0.3986 | 8.064 | 5.921 | 50.72 |
| E169_last/SE3 | 0.5011 | 10.141 | 7.358 | 39.54 |
| E169_last/positions | 0.4033 | 8.201 | 6.032 | 49.44 |
| E169_last/rotations | 0.4759 | 9.269 | 6.620 | 44.98 |
| E169_last/sizes | 0.4140 | 7.750 | 5.676 | 62.72 |
| E169_last/combinations | 0.3982 | 8.218 | 6.024 | 49.20 |
| E168_frozen/train | 0.4892 | 9.858 | 7.344 | 38.46 |
| E168_frozen/val | 0.4903 | 9.732 | 7.251 | 38.89 |
| E168_frozen/iid | 0.4902 | 9.913 | 7.368 | 38.71 |
| E168_frozen/SE3 | 0.4882 | 9.889 | 7.333 | 39.56 |
| E168_frozen/positions | 0.4849 | 9.859 | 7.350 | 38.28 |
| E168_frozen/rotations | 0.5039 | 9.804 | 7.296 | 39.28 |
| E168_frozen/sizes | 0.4957 | 9.310 | 6.950 | 56.38 |
| E168_frozen/combinations | 0.4884 | 10.082 | 7.482 | 37.43 |
| E169_best_shuffledZ/train | 0.3796 | 7.660 | 5.447 | 56.76 |
| E169_best_shuffledZ/val | 0.3843 | 7.625 | 5.412 | 56.64 |
| E169_best_shuffledZ/iid | 0.3803 | 7.685 | 5.468 | 56.56 |
| E169_best_shuffledZ/SE3 | 0.5085 | 10.296 | 7.445 | 38.69 |
| E169_best_shuffledZ/positions | 0.3872 | 7.878 | 5.615 | 54.87 |
| E169_best_shuffledZ/rotations | 0.4553 | 8.879 | 6.257 | 48.30 |
| E169_best_shuffledZ/sizes | 0.3952 | 7.402 | 5.294 | 64.86 |
| E169_best_shuffledZ/combinations | 0.3826 | 7.903 | 5.609 | 54.75 |

Checkpoint steps: {'E169_best': 14500, 'E169_last': 18000, 'E168_frozen': 16500, 'E169_best_shuffledZ': 14500}
A best step0 means the initial sphere won validation; last18k is reported separately.
Frozen Z may lack shape/orientation information; cached-pose diagnostic does not by itself establish flow capacity or deployment robustness.
Source commit: `907e501265502fac07d7d9554453ca8d8cdc182d`.
