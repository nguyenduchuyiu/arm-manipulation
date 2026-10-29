# E177 — Frozen DINOv3 + E175 equivariant point grounding

Status: complete. UserstoppedGVP andrequested oldE175architecture/SAM3→DINOv3. SameE1752localscalar-vectorblocks/UV16graph,2layer128queryGroundedDecoder andequivariantcoordinatehead. Onlyvisualinputdimension256→1024. Alltrainablemodulesfromscratch,onlyDINOv3frozen. Selected block6 normalizedpatchtokens fromfacebook/dinov3-vitl16-pretrain-lvd1689m,24blocks/1024channels;CLS+4registersremoved. RGB320×240resized1152square→72×72patchgrid,bilinearsampledonto4096fullsceneXYZ.

Layerselection: fixed64train/32validationbalancedlinear-ridge5classforegroundprobe,no heldouttestuse,no backboneupdates;probeweightsnotmodelinput. Selectedhighest4colorIoU72.54%. This is bestamongtestedcandidates forsimplecolorgrounding,notuniversalrichesthiddenstate.

| Candidate | Validation foreground mIoU % |
|---|---:|
| block6 | 72.54 |
| block12 | 71.10 |
| block18 | 70.68 |
| block24 | 70.09 |
| mean_last4 | 71.45 |

DataE1571024train/128val/5×128heldoutscenes,4taskIDs/colors,4096fullsceneXYZ→128visibleGTpoints,originalcam_front/robot/chair/rug/table/floor. GTmasks/center/pose/size/objecttokensnotgrounderinputs. ProbeGTmasksonlysupervisionforlayerchoice. Originaldatasetphysicalcuboidresizealreadyrendered;no addedXYZscale/SE3augmentation.

Train30000steps,B8scenes×4tasks,EMDonlyexactunsquaredHungarianinmetres,AdamW3e-4→3e-5cosine30k,warm100,wd.01,clip1,seeds170. BF16autocastscalarMLPs/transformers,FP32geometryvectors/XYZreadout/loss/masterweights/optimizer;primaryvalidation/testFP32. Totaltrainable871858,encoder341034. Bestcanonicalvalstep11000, elapsedminutes36.72. No testbasedcheckpointselection.

| Model/split | EMD mm | Centroid mm | Aligned CD mm | Coverage5mm % | Selection % |
|---|---:|---:|---:|---:|---:|
| E177_best/train | 3.865 | 1.314 | 2.619 | 91.56 | 100.00 |
| E177_best/val | 4.606 | 2.219 | 2.866 | 89.50 | 100.00 |
| E177_best/iid | 4.656 | 2.259 | 2.888 | 89.70 | 100.00 |
| E177_best/positions | 5.544 | 2.910 | 3.312 | 84.86 | 100.00 |
| E177_best/rotations | 7.124 | 3.290 | 4.450 | 62.00 | 100.00 |
| E177_best/sizes | 6.158 | 3.656 | 3.421 | 82.45 | 99.80 |
| E177_best/combinations | 4.912 | 2.333 | 3.023 | 88.00 | 100.00 |
| E177_best/SE3 | 4.656 | 2.259 | 2.888 | 89.93 | 100.00 |
| E177_18k/train | 3.231 | 0.727 | 2.360 | 94.03 | 100.00 |
| E177_18k/val | 4.656 | 2.401 | 2.840 | 90.50 | 100.00 |
| E177_18k/iid | 4.737 | 2.457 | 2.880 | 90.40 | 100.00 |
| E177_18k/positions | 5.660 | 3.186 | 3.279 | 86.44 | 99.80 |
| E177_18k/rotations | 7.416 | 3.646 | 4.514 | 61.39 | 100.00 |
| E177_18k/sizes | 6.270 | 3.930 | 3.321 | 83.36 | 100.00 |
| E177_18k/combinations | 4.986 | 2.523 | 3.015 | 88.91 | 100.00 |
| E177_18k/SE3 | 4.737 | 2.457 | 2.880 | 90.67 | 100.00 |
| E177_last/train | 2.834 | 0.357 | 2.193 | 94.69 | 100.00 |
| E177_last/val | 4.739 | 2.467 | 2.854 | 90.28 | 100.00 |
| E177_last/iid | 4.863 | 2.583 | 2.914 | 90.37 | 100.00 |
| E177_last/positions | 5.867 | 3.384 | 3.339 | 85.36 | 99.80 |
| E177_last/rotations | 7.620 | 3.767 | 4.601 | 59.87 | 100.00 |
| E177_last/sizes | 6.575 | 4.222 | 3.424 | 82.80 | 99.80 |
| E177_last/combinations | 5.209 | 2.750 | 3.092 | 88.18 | 100.00 |
| E177_last/SE3 | 4.863 | 2.583 | 2.914 | 90.60 | 100.00 |

## Best vs E175SAM3best12k

| Split | DINO EMDmm | SAM3 EMDmm | DINO coverage5% | SAM3 coverage5% |
|---|---:|---:|---:|---:|
| iid | 4.656 | 4.419 | 89.70 | 91.75 |
| positions | 5.544 | 4.962 | 84.86 | 89.00 |
| rotations | 7.124 | 6.320 | 62.00 | 69.81 |
| sizes | 6.158 | 5.624 | 82.45 | 84.27 |
| combinations | 4.912 | 4.597 | 88.00 | 91.14 |
| SE3 | 4.656 | 4.418 | 89.93 | 91.87 |

Backbone/featurewidth/image resize/precision/LRhorizon change vs E175,notstrictbackbone-onlycausalablation. Samegeometryarchitecture/UVgraph/dataset/loss. CoordinateSE3 keepsRGB/DINO/UVassociationfixed; not physicalcamerarobustness.

| Model | Pointwise SE3 RMSmm | Maxmm |
|---|---:|---:|
| E177_18k | 0.001339 | 0.011921 |
| E177_best | 0.001135 | 0.011244 |
| E177_last | 0.001320 | 0.011555 |

## Physical OOD, frozen best only

SameE175immutableRGB-D/GTobservations, freshchosenDINOfeatures andUVgraphforeachobservation.128scenes×4tasks/group. Shapes:fourunseentargetfamiliesinsideoriginalcuboidbounds,not equal-volume/density. Camera:originalresampledcontrol,orbit±15/±30/elevation+15,XYZoriginalrobotbase/newvisibleGT. Allqueryscoresprimary; densitysubsetsdiagnostic.

| Domain/condition | DINO EMDmm | SAM3 EMDmm | DINO coverage5% | SAM3 coverage5% |
|---|---:|---:|---:|---:|
| shapes/sphere | 6.143 | 5.948 | 82.46 | 80.97 |
| shapes/ellipsoid | 5.274 | 5.248 | 84.20 | 81.73 |
| shapes/cylinder | 4.293 | 4.220 | 90.33 | 90.22 |
| shapes/capsule | 9.743 | 8.427 | 79.04 | 79.82 |
| camera/original | 4.620 | 4.422 | 89.86 | 91.49 |
| camera/yaw_m15 | 5.090 | 4.796 | 86.05 | 89.78 |
| camera/yaw_p15 | 4.939 | 4.851 | 86.30 | 88.40 |
| camera/yaw_m30 | 5.506 | 10.550 | 80.05 | 74.38 |
| camera/yaw_p30 | 5.488 | 19.802 | 79.90 | 68.01 |
| camera/up15 | 6.363 | 5.084 | 72.12 | 84.19 |

## Correspondence controls

| Featurecondition | EMDmm | Coverage5% | Selection% |
|---|---:|---:|---:|
| baseline | 4.656 | 89.70 | 100.00 |
| other_index | 368.689 | 0.04 | 22.27 |
| other_uv | 183.828 | 0.91 | 31.84 |
| own_shuffled | 356.810 | 0.02 | 23.05 |

Same128IID×4tasks,donorBxor1,seed17562,indexswap/nearestUVswap/ownscenepermutation,recipientXYZ/UVgraph/taskfixed. Conflicting-inputOOD;failure provesdependenceoncorrespondence,notcompletegeometry/semanticdisentanglement.

## Metrics and limits

EMD/directionalCD areunsquaredEuclidean,mm. AlignedCD subtractsindividualcentroids only; otherXYZraw. RobustextentQ95−Q05/RgMAE/R² inrawNPZ/summaries. CoveragevisibleGT4mmvoxels/5mmtolerance. SelectionforcednearestGTcentroidamong4,not absolutegeometrycorrectness. No hidden-surfacecompletion/realworld/noisydepth/newclutter/multi-seedclaim. Newshapevolume/visibility andcamerabackgroundchange simultaneously.

DINOweightsrevision `ea8dc2863c51be0a264bab82070e3e8836b02d51`,sha256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`;featureselectionsource `cff3081f64cc3148cd531221b6b9ec322e613f31`;train/testsource `02b6078df115c69d2a46f02bffe9b8731150e8a0`. Frozenweightsunchanged,checkpoint/source/data/calibrationhashes inconfigs. Retaintrain/valfeaturecache~9GiB;temporaryheldoutsdeletedaftertestbutSHAmanifest retained.
