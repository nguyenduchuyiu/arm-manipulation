# E175 — Unseen target shape families

Status: complete. Frozen best step12,000. No retraining, adaptation or new checkpoint selection. E175 grounder trained only on cuboid targets; frozen pretrained SAM3 may already know these shapes, and background robot geometry contains curved surfaces. This tests unseen target families for the trained grounder.

## Protocol

All128 IID scenes ×4color tasks, paired original cuboid control plus sphere/ellipsoid/cylinder/capsule. All4objects use the same family in each condition. Total512newrenderedscenes/2048newqueries +512controlqueries. Preserve worldcenters/orientations/colors, originalcam_front320×240, robot/chair/rug/table/floor/lighting. Change visual+collision geometry, static rerender RGB/depth/GTmasks, no physics step. No GTcenter/pose/size/mask/family supplied to model.

For original local half-dimensions h: sphere r=min(h); ellipsoid radii=h; cylinder r=min(hx,hy),halfheight=hz; capsule r=min(hx,hy,hz/2),halfsegment=hz−r. Every shape fits original bounds, preserving placement/separation. Volume, Rg and visible area differ; not equal-volume or equal-density testing.

Keep the exact original4096pixelrays per scene, backproject newdepth to actualXYZ; retain the original UVgraph. Recompute SAM3 on every newimage, same exactweights/revision and fpn_hidden_states[-2], BF16vision/FP16sampledfeatures, batch4. GrounderFP32/TF32off,128outputs, allfrozen. No permanent pointfeaturecache. GT128visiblepoints sampled with seed17563000+scene, replacement only when necessary. Include every query, even zero sampled targetpoints.

Parent architecture/training unchanged:772018trainableequivariantencoder/groundeddecoderparameters,1024canonicalE157cuboidscenes,4tasks,4KXYZ,50kEMDupdates,B8,AdamW3e-4→3e-5,noSE3augmentation; bestcanonicalval at12k.

## All-query metrics

| Target family | EMD mm ↓ | Centroid mm ↓ | Aligned CD mm ↓ | Coverage@5mm % ↑ | Forced selection % | Unit-shape EMD ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Cuboid control | 4.419 | 2.047 | 2.779 | 91.75 | 100.00 | 0.2012 |
| Sphere | 5.948 | 4.092 | 2.753 | 80.97 | 99.80 | 0.3590 |
| Ellipsoid | 5.248 | 3.128 | 2.858 | 81.73 | 99.80 | 0.2969 |
| Cylinder | 4.220 | 2.060 | 2.599 | 90.22 | 100.00 | 0.2284 |
| Capsule | 8.427 | 6.408 | 3.242 | 79.82 | 99.22 | 0.3905 |

Each row512queries. Raw XYZ unless explicitly aligned/unit metric. Unitshape removes each cloud centroid/Rg and retains orientation. Coverage uses visibleGT4mmvoxels,5mmtolerance. Selection is forced nearest predictedcentroid to one of4GTcentroids, not absolute reconstruction correctness.

| Target family | Robust extent MAE X/Y/Z mm ↓ | Object-local robust extent MAE X/Y/Z mm ↓ | Rg MAE mm ↓ | Rg R² ↑ |
|---|---:|---:|---:|---:|
| Cuboid control | 2.26 / 1.79 / 1.94 | 2.02 / 1.74 / 1.84 | 0.767 | 0.9536 |
| Sphere | 3.58 / 4.38 / 3.60 | 3.54 / 4.31 / 3.45 | 1.249 | -0.5865 |
| Ellipsoid | 3.34 / 3.43 / 4.07 | 3.29 / 3.33 / 3.98 | 1.051 | 0.3306 |
| Cylinder | 2.63 / 1.99 / 2.99 | 2.43 / 2.18 / 2.48 | 0.920 | 0.9190 |
| Capsule | 4.04 / 8.03 / 5.73 | 4.07 / 8.00 / 5.69 | 2.340 | -4.4426 |

Extent is Q95−Q05, per-query absolute errors then averaged. These are visible sampled surface extents, not fullobject dimensions. GTorientation rotates both clouds only for local-frame metrics, never as modelinput.

## Density diagnostic

| New family | Median input points/target | Zero-target-point queries /512 | EMD with ≥1targetpoint mm | Paired cuboid EMD on same subset mm | Subset queries |
|---|---:|---:|---:|---:|---:|
| Sphere | 7 | 8 | 4.431 | 4.418 | 504 |
| Ellipsoid | 11 | 2 | 4.680 | 4.418 | 510 |
| Cylinder | 14 | 0 | 4.220 | 4.419 | 512 |
| Capsule | 5 | 18 | 4.796 | 4.438 | 494 |

Full-query table remains primary. Subsets select different object sizes/visible areas and do not causally isolate shape effects. Largest sphere/capsule errors cluster at zero targetinputpoints; an ellipsoid outlier has just1point. Excluding zero-point queries changes sphere meanEMD5.948→4.431mm and capsule8.427→4.796mm. Radius R² changes sphere−0.5865→0.7223,ellipsoid0.3306→0.6405,capsule−4.4426→0.7141: large outliers dominate squared size error.

Both newshape and pairedcuboid with≥8inputpoints: sphere245queries EMD3.985vs4.988mm,ellipsoid380queries4.380vs4.668,cylinder424queries4.211vs4.555,capsule150queries3.882vs5.131. Counts/volume still differ, so these do not establish a shape-only gain.

Cylinder generalizes well under this protocol (EMD4.22mm/coverage90.22%). Other curved families partly generalize, but coverage/extent are weaker and4Ksampling has severe outliers on small surfaces. Visualscene0 shows predictions following curved visible surfaces. Smaller objects can have lower rawEMD simply because they are smaller; inspect unitshape/extent/coverage too. No claim about arbitrary nonconvex/hollow shapes, full hidden-surface reconstruction, newcameras/backgrounds, extremeaspectratioholdout, or mixed-family scenes.

## Verification and artifacts

Source branch diag/e175-shapes, executed commit `79720c7ce61690f8634a700aa8ce52b031a60107`. Owned isolatedcode at `/workspace/robotics_assets/experiments/e175_equiv/test_shapes/code`; selective source hashes in code/sources.json, actualcode hashes in results/config.json. No imports from another experiment package.
Checkpoint `/workspace/robotics_assets/experiments/e175_equiv/test_best/results/checkpoint.pt`, SHA256 `aebc05fc6e3afff34d28fd283b12a65e1ca97f82666c6fbcfe5d56affd206a52`.

Four cuboidcontrol renders exactly reproduce savedRGB/depth/segmentation/XYZ. FreshSAM3features exact vs historicalcache; freshgrounder maxbatchFP32drift 0.00014975480735301971mm. Projectionmaxerror 1.0143316217181564e-05pixels. SAM3weighthash unchanged.

All2048erodedobjectsurface checks pass one-pixel-footprint tolerance; worst rawp95<0.922mm and normalizedp95<0.396pixel. Independent audit passed all2560queries: recomputed HungarianEMD androbustextents/summaryaggregates, EMDdifference0mm. Saved independent_check.json. Allpredictions finite.

Pipeline beforeplotting 139.3seconds. Attachedtmux ssh_tmux:e175-shapes, logs run.log, jobcompleted/GPUidle. Run from testroot `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py` creates freshresults; do not overwrite completedresults.

Artifacts /workspace/robotics_assets/experiments/e175_equiv/test_shapes/results: fullconfiguration,summary,stratifiedmetrics,per-familyrawpredictionNPZ/per-querymetricsJSON,render_manifest,pairedchecks,allnewRGB-D/masks/points,RGBstrip and3Dplots. No duplicatedweights or permanentfeaturecache.

Viewer [new shapes](http://localhost:17070/e175_shapes.html), [staticplot](/workspace/robotics_assets/experiments/e175_equiv/test_shapes/results/shapes.png). ChromiumsoftwareWebGL verifiedHTTP200,16scenes/16canvases,no pageerrors; benignreadbackhint/favicon404. RawGTgreen/predictionorange,no GTcorrection. Existingviewersunchanged.

### Precheck note

Initial analytic precheck used an overly strict0.5mm absolute tolerance and stopped before generating newshape data. A direct diagnostic showed original cuboid RGB,depth,segmentation and all4096XYZ exactly reproduce savedE157, while original erodedGTsurface residual already reaches0.665mm p95 for scene0/task0. Keep historical calibration/renderer unchanged; validate new analytic surfaces within one pixel footprint(depth/focal-length), record rawmm and normalizedresiduals. Initial incomplete output contained no predictions/cache and was removed before rerun.

