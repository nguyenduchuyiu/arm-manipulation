# E174 — TaskID + frozen full point memory → direct128XYZ, EMD

Frozen E168 pointmemory4096×128 from E170cache, fresh497411parameter task-conditioned128query TransformerDecoder+XYZMLP, no Z/noFM/noise/timestep. FP32,18ksteps, exactunsquaredHungarianEMD only. Canonical E157fullscene inputs and128visibleGTpoints; noGTcenter/size/masks.

| Method / split | EMD mm ↓ | Centroid mm ↓ | Rg error mm ↓ | Coverage5mm % ↑ | Selection % ↑ |
|---|---:|---:|---:|---:|---:|
| E174_best/train | 5.898 | 3.640 | 1.059 | 75.75 | 100.00 |
| E174_best/val | 7.525 | 5.753 | 1.120 | 63.23 | 100.00 |
| E174_best/iid | 7.682 | 5.913 | 1.154 | 62.87 | 100.00 |
| E174_best/SE3 | 344.591 | 344.252 | 11.634 | 0.56 | 39.26 |
| E174_best/positions | 11.096 | 9.713 | 1.532 | 44.06 | 100.00 |
| E174_best/rotations | 11.023 | 7.999 | 1.592 | 40.87 | 100.00 |
| E174_best/sizes | 10.566 | 8.822 | 1.592 | 49.24 | 100.00 |
| E174_best/combinations | 8.011 | 6.232 | 1.194 | 60.44 | 100.00 |
| E174_last/train | 5.768 | 3.492 | 0.975 | 77.38 | 100.00 |
| E174_last/val | 7.599 | 5.855 | 1.059 | 63.70 | 100.00 |
| E174_last/iid | 7.766 | 5.999 | 1.085 | 62.83 | 100.00 |
| E174_last/SE3 | 345.555 | 345.221 | 11.350 | 0.57 | 38.82 |
| E174_last/positions | 11.332 | 9.936 | 1.433 | 43.28 | 100.00 |
| E174_last/rotations | 11.190 | 8.162 | 1.524 | 40.46 | 100.00 |
| E174_last/sizes | 10.737 | 9.025 | 1.568 | 48.66 | 100.00 |
| E174_last/combinations | 8.095 | 6.287 | 1.121 | 60.24 | 100.00 |

Checkpoint steps: {'E174_best': 15500, 'E174_last': 18000}
Best selected by canonical validation view0 EMD every500. Primary controlled comparison uses last18000 for both E171 and E174, without checkpoint-selection differences.
Only taskID+frozen full point memory; no Z,rawXYZ,GTcentroid/Rg/masks as decoder input.
Fresh direct decoder trained canonicalview0 only;4SE3testviews OOD; frozen sourceencoder originallySE3trained.
Single deterministic forward,FP32; no source noise,timestep,flowintegration or outputnormalization.
Target selection: predicted centroid nearest to one of four GT128 centroids; shape/coverage evaluated separately.
E174 direct decoder/EMD versus E171FM/E173FMplusglobal: same frozenmemory/data/noSE3/budget, different architecture/conditioning and objective. Best selection canonicalval for all.
Source commit: `a4fe7f3cc1270809e6c8bbf43f4549d342612498`.

## Fixed18k comparison

| Run/split | EMD mm ↓ | AlignedCD mm ↓ | Unit-shape EMD ↓ | Coverage5mm % ↑ |
|---|---:|---:|---:|---:|
| E171_FM_last/train | 10.309 | 5.053 | 0.3559 | 49.16 |
| E171_FM_last/iid | 11.266 | 5.066 | 0.3564 | 45.16 |
| E171_FM_last/SE3 | 312.990 | 9.150 | 0.5692 | 1.57 |
| E171_FM_last/positions | 14.223 | 5.235 | 0.3604 | 35.70 |
| E171_FM_last/rotations | 13.632 | 6.106 | 0.4582 | 38.27 |
| E171_FM_last/sizes | 13.923 | 5.048 | 0.3754 | 40.27 |
| E171_FM_last/combinations | 11.304 | 5.193 | 0.3610 | 44.78 |
| E174_directEMD_last/train | 5.768 | 3.305 | 0.2275 | 77.38 |
| E174_directEMD_last/iid | 7.766 | 3.446 | 0.2360 | 62.83 |
| E174_directEMD_last/SE3 | 345.555 | 10.346 | 0.6101 | 0.57 |
| E174_directEMD_last/positions | 11.332 | 3.798 | 0.2616 | 43.28 |
| E174_directEMD_last/rotations | 11.190 | 5.409 | 0.3925 | 40.46 |
| E174_directEMD_last/sizes | 10.737 | 3.556 | 0.2779 | 48.66 |
| E174_directEMD_last/combinations | 8.095 | 3.594 | 0.2430 | 60.24 |

One seed. Same frozen fullmemory/data/canonicaltraining/budget/LR as E171; architecture/conditioning and objective change. Use centroid/alignedCD/unitshape/coverage jointly to assess geometry.
