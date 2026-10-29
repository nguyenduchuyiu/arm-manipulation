# E174 — Grounding decoder + direct XYZ head

Status: training/evaluation complete; see RESULTS.md.

User-requested parallel decoder baseline. Input only taskID0..3 and full frozen point memory4096×128.128learned output queries + trainabletaskembedding4×128 → 2layerTransformerDecoder(width128,4heads,FF384,GELU,dropout0,normfirst,finalLN) → sharedXYZMLP128→128GELU→3 →128points.497411trainableparameters, all newly initializedseed170. This is the original grounding-decoder/XYZ-head architecture, trained from scratch; source pointencoder/SAM3 remain frozen. Do not load frozen task-conditioned Z or trained decoder weights. No rawXYZ/GTcentroid/Rg/mask/pose/shape parameters as decoder input; no noise,timestep,FM,integration,outputnormalization or centering.

Memory source immutable E170cache, extracted by E168best16500 frozen pointencoder381568parameters (the cached encoder/taskdecoder source bundle has862080parameters; its taskdecoder/Z are unused here). Source checkpoint SHA25665527d6657e57aef6552c01bf1d33c2c9014c88a08a4c332693342f71e776207. Frozen SAM3 features already incorporated per point. Load canonical trainview0 memory only (~1GiB onGPU), valview0/1 (~0.25GiB); no new feature caches. Other requested views loaded only for evaluation. E173 runs in parallel; E172 remainspaused.

Data E157 unchanged RGB320×240/fullscene4096XYZ withrobot/chair/rug,4coloredcubes,GT128visible targetpoints.1024train×4tasks,128val and128each IID/positions/rotations/sizes/combinations. NoSE3 training, canonicalview0 only;4IIDSE3views OODdiagnostic. Source frozen encoder was originallySE3trained.

Train18000steps, B8scenes×4tasks=32outputs,FP32decoder/attention/AdamW/EMD,TF32off,cacheFP16castFP32. Loss only exactunsquaredEuclideanHungarianEMD mean matched distance(metres); correspondence detached, gradient through matched distances. No Chamfer, pairwise,centroid,globalvelocity or other auxiliary loss. AdamW3e-4→3e-5cosine,100warmup,wd.01,clip1. Model/dataseed170; scene draws match E171/E173 by consuming their discardedviewdraws. One deterministic forward perquery, no samplingseed.

Validate canonical128×4 every500steps, best byworldEMD (initialstep0eligible); SE3valview1 diagnostic only. Autoevaluate best/last18k on train/val/5heldoutcanonicalsets+4IIDSE3views. Report EMD,centroid,alignedCD,directionalCD,unitshapeEMD,Rgerror,robustextents,GTsurfacevoxel4mmcoverage@2/5mm,targetselection. Compare savedE171FM18k and E173FM+global when available; if E173 still running record incomplete rather than wait. Architecture/conditioning and training objective differ, so this is not an isolated loss/architecture ablation. One seed.

Own isolated branch exp/e174-direct from65c7f27, selective source hashes in sources.json,no cross-experiment code imports. Preflight checks taskID/alltasks wiring,memory permutation invariance,EMDvsSciPy/GTpermutation,canonicalGT/frozencache and full B8FP32backward. Sourcecommit saved beside artifacts. Run `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py verify` then `... code/run.py train`, creates freshtrain dir and autoevaluates. Attachedtmux ssh_tmux:e174-log,stdout+stderr mirrored to run.log. Runtime measured under jointGPU load; no estimate claimed beforebenchmark. Existing forwarded viewer localhost:17070/e174.html oncompletion.

Artifact status: training/evaluation complete; see RESULTS.md. Executing source commit: `a4fe7f3cc1270809e6c8bbf43f4549d342612498`.

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

Completion audit: trained/evaluated18000steps, best15500, elapsed33.329min under initiallysharedGPUload. Bothcheckpoint hashes match. Independent exactHungarian on512IIDlast queries reproduces7.766462mm; same GT arrays as E171. Viewer localhost:17070/e174.html HTTP200,4865546bytes. E173 interrupted and nowexplicitlypaused at userrequest; no18kE173 comparison yet.
