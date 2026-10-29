# E175 — Full evaluation audit after50k

Status: complete.

Audit completed model evaluations, no retraining/newinference/weight changes. E175trained50000steps on1024originalscenes/noSE3,772018trainableequivariantencoder/decoderparameters,onlySAM3frozen,EMDloss,B8,FP32,AdamW3e-4→3e-5. Evaluate/check threeactualcheckpoints:milestone18000,best12000,last50000. Source full inference already completed automatically in parent evaluation; this independentCPU audit rechecks results rather than overwriting them.

Scope: fulltrain1024/val128/each5heldout128scenes,4tasks/128points/query,plus4globalSE3views of128IIDscenes. Percheckpoint:7168canonical+2048SE3=9216queries. Every savedcloud shape/finiteness/GT and every summarymetric aggregate checked. Independently recompute exactHungarianEMD andcentroid on allval/heldout/SE3queries,validate GT against rawE157,targettransform androtationorthogonality,pointwiseequivariance,checkpoint/SAM3/graphhashes. Coverage@5mm recomputed independently with GTsurface4mmvoxel representatives for4fixedscenes×4tasks percanonicalsplit/model. Train metrics use allsavedarrays; no extra model inference needed. Existing12kpreview shouldmatchfinalbest within FP32 tolerance.

Source shapes use sameRGB/SAM3/pixelgraph under coordinate-frame changes. GlobalSE3 equivariance is not physicaltarget-only rotation generalization. Modelconstraint/framework/pretraining and50kLRhorizon differ from E174; compare18kmilestone as temporalreference, not purearchitectureablation. User's pausedE172/E173 remainstopped.

Own isolateddiag/e175-fullworktree,one selective calibrationhelper andsourcehash in sources.json; no imports from parentexperimentcode. Run `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py` from testroot. CPUaudit saves audit.json,stdout/stderr mirrored in run.log andattachedtmux ssh_tmux:e175-full. Canonical report docs/research/e175_results.md and e175_full.md. No trainedweights/checkpoints/cache duplicated.

Auditpassed; codecommit `ca81a10a52fcf576e7c445a41502aa228e5f63df`. Counts {'canonical_queries': 21504, 'SE3_queries': 6144, 'independent_EMD_queries': 15360, 'coverage_spotchecks': 336}. See audit.json.

# E175 — Full test after50k

Data:1024train/128validation/640heldoutscenes,4tasks per scene,4096fullsceneXYZ+frozenSAM3features,pixelcorrespondence graph;128targetpoints. Sameoriginalcamera/robot/chair/rug. Clutter is present but held fixed; changedclutter/camera are not tested.

Architecture:772018trainableparameters,scalar/vector3Dencoder and equivariant directcoordinate decoder; noFM. OnlySAM3frozen. Trainoriginalcanonicalscenes only, noSE3/scaleaugmentation. EMDloss,FP32,B8,AdamW3e-4→3e-5,50000updates,elapsed119.358min.

Bestselected by canonicalvalidation at12000steps (valEMD4.373554mm). Last50000 is not better: IIDEMD4.791mm vsbest4.419; trainEMD falls3.476→2.333mm while validation4.374→4.745mm. Use best12k for inference.18kmilestoneIIDEMD4.451mm.

| Heldout/best12k | Queries | EMD mm ↓ | Centroid mm ↓ | AlignedCD mm ↓ | Coverage5mm % ↑ | Targetselection % |
|---|---:|---:|---:|---:|---:|---:|
| iid | 512 | 4.419 | 2.047 | 2.779 | 91.75 | 100.00 |
| positions | 512 | 4.962 | 2.503 | 3.056 | 89.00 | 100.00 |
| rotations | 512 | 6.320 | 2.738 | 4.076 | 69.81 | 100.00 |
| sizes | 512 | 5.624 | 3.174 | 3.102 | 84.27 | 100.00 |
| combinations | 512 | 4.597 | 2.199 | 2.852 | 91.14 | 100.00 |
| SE3 | 2048 | 4.418 | 2.047 | 2.779 | 91.87 | 100.00 |

Targetselection is predictedcentroid nearest to the correctone of4GTcentroids; geometry/coverage measured separately. GlobalSE3test keeps RGB/SAM3/pixelgraph correspondence fixed, transforms sceneXYZ andGT together. It differs from physicaltarget-only rotations (rotationssplit).

GlobalSE3equivariance over2048queries/262144points: RMS0.001177650mm,max0.005778516mm,p990.003082298mm. IID/SE3EMD4.418514/4.418495mm. Axis-alignedvoxelcoverage can differ underrotation; use pointwisedrift/Euclideandistances for equivariance.

Reference E174last18k IIDEMD7.766mm/centroid5.999mm/alignedCD3.446mm/coverage62.83%; E17518k4.451mm/2.133mm/2.770mm/91.94%. Encoderpretraining/trainability,architecture andLRhorizon differ, so no isolatedcausal claim about equivariance alone.

Independentfullaudit passed: three checkpoints,21504canonical+6144SE3queryrecords; allrawGT/arrayfiniteness/metricaggregates,checkpoint/SAM3/graphhashes. Recomputed15360HungarianEMD/centroidqueries and336visible-surfacecoverage spotchecks; all pointwiseSE3drifts independently verified. Existing12kpreview predictions exactly matchfinalbest (maxdrift0mm). Sourcecommit552a2a3,auditcommitca81a10.

[Allcheckpointresults](e175_results.md) · [Best3Dscene](http://localhost:17070/e175_scene3d/) · [BestvsGT](http://localhost:17070/e175.html) · [Auditsummary](/workspace/robotics_assets/experiments/e175_equiv/test_full/audit.json)

E172/E173 remain explicitlypaused. No furthertraining launched by this test.
