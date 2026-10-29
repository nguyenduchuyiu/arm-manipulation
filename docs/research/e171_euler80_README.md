# E171last18k — Euler40/80 inference against Euler20

Status: complete.

User-requested sampler-resolution test: exact E171last18000checkpoint, same fixedGaussian128particles/noise167, same frozenZ+4096pointmemory. ForwardEuler with20/40/80steps over t=0→1,dt=1/N. Source particles are reset identically for each resolution; do not append extra steps beyondt=1. No retraining, GTcentroid alignment, scale/orientation correction or sourceweights/cache modification. E17250k remains paused per user request.

Test all128IIDscenes×4tasks=512queries,128output/GTpoints,B8scenes,FP32sampler/attention withTF32off. GT used only for evaluation. Reuse immutable E170canonicalview0memory/Z/labels. Model architecture/training unchanged:580483trainable FM/adapters in original E171,18000steps,AdamW3e-4→3e-5,HungarianvelocityMSE,noSE3; no optimizer/LR in this inference diagnostic.

Record centroid trajectories and endpoints, then exactHungarianEMD,raw/centroidalignedChamfer,directionalChamfer,Rgerror,visibleGTsurfacevoxel4mm coverage@5mm. Mean centroid error curves compared on common FMtime axis; zoomfinal15%. Baseline20 must reproduce old prediction bit-exact and originalEMD/coverage; GT/sourceparticles identical. Save perqueryNPZ,CSV,PNG/SVG/HTML. One fixednoise/checkpoint; no generalization claim from noise seeds.

Own isolated diag/e171-euler80worktree under test_euler80. Selective source copies/hashes in sources.json; no imports from other experiment code. Run `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py` from testroot; fresh results directory, attachedtmux ssh_tmux:e171-e80,stdout/stderr mirrored in run.log. Sourcecommit saved beside artifacts. Existing forwarded viewer localhost:17070/e171_euler80.html; canonical notes docs/research/e171_euler80.md.

Source commit `e4d18c231ae8fbfcb47d7893230476fedb44f4f0`. See results/summary.json and results/RESULTS.md.

# E171last18k — Euler20/40/80

Same512IIDqueries,noise167,weights andsource; fullFM interval[0,1],dt=1/N. No training or GTalignment.

| Eulersteps | Centroid mm ↓ | EMD mm ↓ | CD mm ↓ | AlignedCD mm ↓ | Coverage5mm % ↑ |
|---:|---:|---:|---:|---:|---:|
| 20 | 8.382 | 11.266 | 6.719 | 5.066 | 45.16 |
| 40 | 8.712 | 11.530 | 6.867 | 5.069 | 43.39 |
| 80 | 8.755 | 11.709 | 6.980 | 5.203 | 42.85 |

Baseline20 endpoint reproduced bit-exact, EMD/coverage checked. Single checkpoint/fixednoise; no inference beyond t=1.
Source commit `e4d18c231ae8fbfcb47d7893230476fedb44f4f0`.

Completion audit:20/40/80 all use t0→1 and identical initialparticles. Euler20endpoint bit-identical to original evaluation; baseline EMD/coverage reproduced. PNGvisuallyinspected; HTML localhost:17070/e171_euler80.html HTTP200,4829119bytes. On512IIDqueries, increasing steps did not improve mean centroid/EMD/coverage. E172 remains paused per user request.
