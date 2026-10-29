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

## Shape/extent readout from saved best12k predictions

Analysis of existing results only; no retraining or new model inference. Each split has128scenes ×4tasks. Robust extent is Q95−Q05 on each robot-frame axis; error is per-query absolute prediction−GT128 extent, averaged over512queries (not the difference of split means). These are extents of the observed/sampled target surface, not complete simulator-object local dimensions. Rg is RMS distance to each cloud centroid; R² computed over all512queries.

| Split | Robust extent MAE X/Y/Z mm | Rg MAE mm | Rg R² | Unit-shape EMD |
|---|---:|---:|---:|---:|
| iid | 2.264 / 1.793 / 1.944 | 0.767 | 0.9536 | 0.2012 |
| sizes | 3.116 / 2.862 / 2.655 | 1.142 | 0.9540 | 0.2728 |
| rotations | 4.332 / 3.921 / 5.203 | 1.559 | 0.8079 | 0.3066 |

Existing tests cover changed cuboid dimensions and physical rotations, but not novel shape families/topology or an explicitly held-out extreme aspect-ratio split. E157 independently applies local per-axis scale U(0.6,1.4) to all splits and rerenders consistent RGB-D/GT. Original base sides: train24–50mm, sizes14–22mm or54–68mm. Hence possible per-axis physical lengths overlap: train14.4–70mm, size split8.4–30.8mm or32.4–95.2mm (before acceptance rejection). Calling every size-test sample strictly outside the training extent support would be incorrect. This is a shifted size distribution within the same cuboid family, with some dimensions extrapolating.

Rotation split remains weaker: robust extent underprediction biases X/Y/Z −3.728/−1.678/−4.336mm, coverage69.81%, vs84.27% on sizes and91.75% IID. This is physical pose generalization, separate from exact global-frame SE(3) equivariance. A new explicit shape/aspect-ratio holdout with consistent rerendering is still needed.
