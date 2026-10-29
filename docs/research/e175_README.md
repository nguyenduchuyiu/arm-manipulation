# E175 — SE(3)-equivariant encoder and direct coordinate decoder

Status: training/evaluationcomplete;see RESULTS.md.

User request: remove FM, constrain both3Dencoder and direct128pointdecoder to be equivariant, train50000updates on original scenes with no SE3augmentation. Only pretrainedSAM3 remainsfrozen. Old E168/E170non-equivariant3Dmemory is not used. Fresh scalar/vector3Dencoder and taskdecoder trained jointly; actual trainable count in verify.json/train/config.json. No GT masks,centroid,size,objecttokens or cameraextrinsics as modelinput.

Inputs: original E157fullscene4096XYZ(metres), matching raw frozenSAM3features256, taskID0red/1blue/2green/3yellow, observation-derived pixel-neighbor graph. SAM3features frozen facebook/sam3 revision3c879f39826c281e95690f02c7821c4de09afae7,vision_features FPN[-2]256×72×72, originally sampled bilinearly onto points;454038784frozenvisionparameters,cacheFP16. Source features/config.json hash recorded; no new image inference/featurecache. Original RGB320×240 camera and robot/chair/rug remain. Dataset1024train×4tasks,128val and128each IID/positions/rotations/sizes/combinations; GT128visiblepoints perobject.

Encoder: width128,8vectorchannels,2local scalar/vector message blocks. Trainable SAM3scalarMLP gives h; initial V=scalar-gated (P−mean_sceneP)/0.1m. Local messages use invariant3Ddistance RBFs,scalarfeatures andvectornorms. Vector update is bias-free channelmixing plus scalar-weighted relativeXYZ, with scalar gates shared acrossXYZ. Geometry is scaled uniformly by0.1m; no per-axis normalization. Localgraph is16nearestpixelUVneighbors,canonically UV-sorted beforeKDtree to make tie-breaking point-order independent. PixelUV already belongs to the RGB-pointfeature correspondence and staysfixed under3Dcoordinate changes; graph is not computed fromGT masks or objectcrops. Cache only these graphindices (~224MiB), rawfeatures/datasets unchanged.

Decoder:128learnedscalarqueries+taskembedding4×128,2TransformerDecoderlayers,width128,4heads,FF384 on invariant h. Coordinateattention weights softmax(q(latent)k(h)^T) across all4096points. Output = scene_mean + sum_i a_ki(P_i−scene_mean) +0.1m sum_c scalar_coefficient_kc sum_i a_ki V_ic. Scalars are invariant; vectors rotate; affinepointweights sum1. No ordinary MLP outputs fixed-axis XYZ; the MLP emits only scalar vector-channelcoefficients. Signedvectoroffsets allow output beyond convex pointaverages. One deterministic forward, no noise/timestep/integration/GTcentering.

Contract/proof: when P'=RP+t and sameRGB/SAM3/pixelgraph/task, distances,norms and h remaininvariant; V'=RV; scalarattention/coefficients unchanged; output'=Routput+t. This is equivariance for a change of3Dcoordinateframe with point-observation associations preserved. It does not automatically guarantee new physicaltarget-only rotations, unseen shapes or good reconstruction. FP32 numerical errors measured beforetraining/afterupdates/every1000validation/finalheldoutSE3tests. Source concepts: EGNN (https://proceedings.mlr.press/v139/satorras21a.html), scalar/vector representations (https://arxiv.org/abs/2104.12229); implementation is custom and all source is experiment-local.

Train scratch50000steps, B8scenes×4tasks=32queries,full4096points/128outputs,FP32 encoder/decoder/attention/EMD/AdamW,TF32off; rawSAM3cacheFP16castFP32. Loss only exactunsquaredEuclideanHungarianEMD inmetres. AdamW3e-4→3e-5cosine across50k,100warmup,wd.01,clip1;seed170 model/data. OriginalcanonicalP/G only, no rotation/translation/scaling augmentation. Consume original familyviewRNGdraws then discard for matching scene-ID stream; no transformedtraining examples. Benchmark fullB8 before launch; actual speed/peak in verify.json. Save initial,best,last,resume and18kmilestone only. E172/E173 remain paused, completedexperiments untouched.

Validation: canonical128×4 every1000 selects best byEMD; step0eligible. FixedSE3val is diagnostic only, no gradients/checkpointselection. End evaluate E17518k/best/last50k on train/val/all5heldoutcanonicalsets and4IIDglobalSE3transforms fromE137/E170, sameSAM3/graph. Report EMD,centroid,alignedCD,directionalCD,unitshapeEMD,Rgerror,robustextents,surfacevoxelcoverage and targetselection; additionally exactquery-corresponding pointwiseSE3RMS/max/p99 and per-query transformedGT. Axis-alignedvoxelcoverage itself is not rotation-invariant; use pointwise drift andEuclideandistances to judge equivariance. Compare E17418k as contextual baseline: encoder pretraining/trainability,architecture andbudget differ;18kmilestone uses50kLRhorizon, not a fullycontrolledablation. Single seed.

Own worktree branch exp/e175-equiv from65c7f27. Selective E174geometry/EMD utilities, hashes in sources.json; no cross-experiment code imports. Commit before run, hash recorded in artifacts. Run `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py cache`,then verify,then train; fresh graph/train/evaluation directories created by programs. Longstages in attachedtmux ssh_tmux:e175-log,stdout+stderr mirrored to run.log. Train autoevaluates. Standalone best-vs-GT will use existing forwarded localhost:17070/e175.html. Runtime estimate only after fullGPUbenchmark. No reconstruction result claimed yet.

Artifact status: training/evaluationcomplete;see RESULTS.md. Executing source commit: `552a2a353aaae14f3b2e8faa90fb6a20d5781a41`.

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
