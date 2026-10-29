# E178 — DINOv3 block 24 versus E177 block 6

Status: complete. Controlled layer comparison: only frozen DINOv3 selected hidden changes from block6 to block24. Same pretrained weights, final LayerNorm, RGB resize1152 square, 72×72 patch grid, 1024 channels, bilinear point sampling, E175 architecture/UV16 graph, task IDs, seeds, data, BF16/FP32 policy, EMD loss, LR schedule and 30k budget. All trainable modules initialized from scratch. No E177 trained weights loaded.

Block24 fixed by user request, not selected by a new probe or heldout scores. Final pretrained model.norm is applied in FP32; extractor asserts exact equality to official last_hidden_state before removing CLS+4 registers. No GT used for layer selection in E178.


DataE1571024train/128val/5×128heldoutscenes,4taskIDs/colors,4096fullsceneXYZ→128visibleGTpoints,originalcam_front/robot/chair/rug/table/floor. GTmasks/center/pose/size/objecttokensnotgrounderinputs. No GT layer-selection probe in this run. Originaldatasetphysicalcuboidresizealreadyrendered;no addedXYZscale/SE3augmentation.

Train30000steps,B8scenes×4tasks,EMDonlyexactunsquaredHungarianinmetres,AdamW3e-4→3e-5cosine30k,warm100,wd.01,clip1,seeds170. BF16autocastscalarMLPs/transformers,FP32geometryvectors/XYZreadout/loss/masterweights/optimizer;primaryvalidation/testFP32. Totaltrainable871858,encoder341034. Bestcanonicalvalstep12000, elapsedminutes35.85. No testbasedcheckpointselection.

| Model/split | EMD mm | Centroid mm | Aligned CD mm | Coverage5mm % | Selection % |
|---|---:|---:|---:|---:|---:|
| E178_best/train | 3.481 | 0.973 | 2.447 | 93.16 | 100.00 |
| E178_best/val | 4.445 | 2.123 | 2.769 | 90.45 | 100.00 |
| E178_best/iid | 4.574 | 2.264 | 2.834 | 90.36 | 100.00 |
| E178_best/positions | 5.562 | 3.063 | 3.257 | 86.12 | 100.00 |
| E178_best/rotations | 6.735 | 3.061 | 4.271 | 64.91 | 100.00 |
| E178_best/sizes | 5.858 | 3.444 | 3.201 | 82.60 | 100.00 |
| E178_best/combinations | 4.728 | 2.350 | 2.893 | 89.98 | 100.00 |
| E178_best/SE3 | 4.574 | 2.264 | 2.834 | 90.58 | 100.00 |
| E178_18k/train | 3.070 | 0.559 | 2.287 | 94.47 | 100.00 |
| E178_18k/val | 4.455 | 2.225 | 2.758 | 91.30 | 100.00 |
| E178_18k/iid | 4.555 | 2.281 | 2.807 | 91.35 | 100.00 |
| E178_18k/positions | 5.741 | 3.282 | 3.260 | 86.32 | 99.80 |
| E178_18k/rotations | 6.767 | 3.162 | 4.247 | 65.67 | 100.00 |
| E178_18k/sizes | 5.995 | 3.622 | 3.224 | 83.86 | 99.80 |
| E178_18k/combinations | 4.772 | 2.449 | 2.888 | 90.73 | 100.00 |
| E178_18k/SE3 | 4.555 | 2.281 | 2.807 | 91.55 | 100.00 |
| E178_last/train | 2.744 | 0.309 | 2.148 | 94.94 | 100.00 |
| E178_last/val | 4.601 | 2.371 | 2.798 | 91.19 | 100.00 |
| E178_last/iid | 4.672 | 2.389 | 2.828 | 91.18 | 100.00 |
| E178_last/positions | 5.906 | 3.416 | 3.344 | 85.57 | 99.80 |
| E178_last/rotations | 6.999 | 3.398 | 4.314 | 65.53 | 100.00 |
| E178_last/sizes | 6.121 | 3.713 | 3.298 | 83.47 | 100.00 |
| E178_last/combinations | 4.981 | 2.635 | 2.950 | 90.10 | 100.00 |
| E178_last/SE3 | 4.672 | 2.389 | 2.828 | 91.39 | 100.00 |

## Controlled comparison: block24 versus block6

| Split | Block24 EMD mm | Block6 EMD mm | Block24 coverage5 % | Block6 coverage5 % |
|---|---:|---:|---:|---:|
| train | 3.481 | 3.865 | 93.16 | 91.56 |
| val | 4.445 | 4.606 | 90.45 | 89.50 |
| iid | 4.574 | 4.656 | 90.36 | 89.70 |
| positions | 5.562 | 5.544 | 86.12 | 84.86 |
| rotations | 6.735 | 7.124 | 64.91 | 62.00 |
| sizes | 5.858 | 6.158 | 82.60 | 82.45 |
| combinations | 4.728 | 4.912 | 89.98 | 88.00 |
| SE3 | 4.574 | 4.656 | 90.58 | 89.93 |

Best checkpoint independently selected on the same validation split in each run; single shared seed, no multi-seed statistical claim. Full reference summaries and source hashes are saved in evaluation/block6_comparison.json.

| OOD condition | Block24 EMD mm | Block6 EMD mm | Block24 coverage5 % | Block6 coverage5 % |
|---|---:|---:|---:|---:|
| shapes/sphere | 9.045 | 6.143 | 70.57 | 82.46 |
| shapes/ellipsoid | 7.252 | 5.274 | 66.39 | 84.20 |
| shapes/cylinder | 9.173 | 4.293 | 65.41 | 90.33 |
| shapes/capsule | 15.092 | 9.743 | 59.72 | 79.04 |
| camera/original | 4.516 | 4.620 | 90.52 | 89.86 |
| camera/yaw_m15 | 5.029 | 5.090 | 86.92 | 86.05 |
| camera/yaw_p15 | 5.158 | 4.939 | 84.66 | 86.30 |
| camera/yaw_m30 | 6.024 | 5.506 | 75.88 | 80.05 |
| camera/yaw_p30 | 6.145 | 5.488 | 74.57 | 79.90 |
| camera/up15 | 5.483 | 6.363 | 78.13 | 72.12 |

## Best vs E175SAM3best12k

| Split | DINO EMDmm | SAM3 EMDmm | DINO coverage5% | SAM3 coverage5% |
|---|---:|---:|---:|---:|
| iid | 4.574 | 4.419 | 90.36 | 91.75 |
| positions | 5.562 | 4.962 | 86.12 | 89.00 |
| rotations | 6.735 | 6.320 | 64.91 | 69.81 |
| sizes | 5.858 | 5.624 | 82.60 | 84.27 |
| combinations | 4.728 | 4.597 | 89.98 | 91.14 |
| SE3 | 4.574 | 4.418 | 90.58 | 91.87 |

Backbone/featurewidth/image resize/precision/LRhorizon change vs E175,notstrictbackbone-onlycausalablation. Samegeometryarchitecture/UVgraph/dataset/loss. CoordinateSE3 keepsRGB/DINO/UVassociationfixed; not physicalcamerarobustness.

| Model | Pointwise SE3 RMSmm | Maxmm |
|---|---:|---:|
| E178_18k | 0.001092 | 0.006008 |
| E178_best | 0.001020 | 0.006115 |
| E178_last | 0.001042 | 0.005798 |

## Physical OOD, frozen best only

SameE175immutableRGB-D/GTobservations, freshchosenDINOfeatures andUVgraphforeachobservation.128scenes×4tasks/group. Shapes:fourunseentargetfamiliesinsideoriginalcuboidbounds,not equal-volume/density. Camera:originalresampledcontrol,orbit±15/±30/elevation+15,XYZoriginalrobotbase/newvisibleGT. Allqueryscoresprimary; densitysubsetsdiagnostic.

| Domain/condition | DINO EMDmm | SAM3 EMDmm | DINO coverage5% | SAM3 coverage5% |
|---|---:|---:|---:|---:|
| shapes/sphere | 9.045 | 5.948 | 70.57 | 80.97 |
| shapes/ellipsoid | 7.252 | 5.248 | 66.39 | 81.73 |
| shapes/cylinder | 9.173 | 4.220 | 65.41 | 90.22 |
| shapes/capsule | 15.092 | 8.427 | 59.72 | 79.82 |
| camera/original | 4.516 | 4.422 | 90.52 | 91.49 |
| camera/yaw_m15 | 5.029 | 4.796 | 86.92 | 89.78 |
| camera/yaw_p15 | 5.158 | 4.851 | 84.66 | 88.40 |
| camera/yaw_m30 | 6.024 | 10.550 | 75.88 | 74.38 |
| camera/yaw_p30 | 6.145 | 19.802 | 74.57 | 68.01 |
| camera/up15 | 5.483 | 5.084 | 78.13 | 84.19 |

## Correspondence controls

| Featurecondition | EMDmm | Coverage5% | Selection% |
|---|---:|---:|---:|
| baseline | 4.574 | 90.36 | 100.00 |
| other_index | 878.976 | 0.00 | 26.37 |
| other_uv | 182.618 | 0.74 | 31.45 |
| own_shuffled | 865.314 | 0.00 | 25.00 |

Same128IID×4tasks,donorBxor1,seed17562,indexswap/nearestUVswap/ownscenepermutation,recipientXYZ/UVgraph/taskfixed. Conflicting-inputOOD;failure provesdependenceoncorrespondence,notcompletegeometry/semanticdisentanglement.

## Metrics and limits

EMD/directionalCD areunsquaredEuclidean,mm. AlignedCD subtractsindividualcentroids only; otherXYZraw. RobustextentQ95−Q05/RgMAE/R² inrawNPZ/summaries. CoveragevisibleGT4mmvoxels/5mmtolerance. SelectionforcednearestGTcentroidamong4,not absolutegeometrycorrectness. No hidden-surfacecompletion/realworld/noisydepth/newclutter/multi-seedclaim. Newshapevolume/visibility andcamerabackgroundchange simultaneously.

DINOweightsrevision `ea8dc2863c51be0a264bab82070e3e8836b02d51`,sha256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`;extractor origin source `02b6078df115c69d2a46f02bffe9b8731150e8a0`;train/testsource `ed39d076aa330262ed0df6a91ce2ec5c60262886`. Frozenweightsunchanged,checkpoint/source/data/calibrationhashes inconfigs. Retaintrain/valfeaturecache~9GiB;temporaryheldoutsdeletedaftertestbutSHAmanifest retained.
