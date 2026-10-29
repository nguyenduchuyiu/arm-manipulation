# E177 — E175 architecture with frozen DINOv3

Status: **complete —30kandalltestsfinished;best11k**. See [finalresults](e177_full.md). The implementation/training progress snapshots below are historical. Updated 2026-10-28T14:36:11.335722+00:00. E176GVP explicitlystoppedbyuser at14.6klogged/14kresumesaved, best9kval4.3885mm; no30k/fulltestclaim. Oldcheckpoints/logs preserved,deadwindowremoved.

## Data and architecture

Same E1571024train/128val/5×128heldoutscene/4task data,RGB320×240/full4096XYZrobotbase→128visibleGTpoints;originalcam_front/robot/chair/rug/table/floor retained. No GTmask/center/size/pose/objecttoken asgrounderinput. OldE1752localscalar-vectorblocks,width128/eightvectorchannels/pixelUV16graph,2layer128queryGroundedDecoder andequivariantcoordinatehead. No GVP/no fullpointselfattention. Onlyvisualfeatureinputwidth256→1024 changed;trainablecount871858,encoder341034. ReusesoriginalE175UVgraphcachereadonly.

FrozenDINOv3ViT-L/16(300M,24blocks,1024features),officialHF facebook/dinov3-vitl16-pretrain-lvd1689m,revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`,weightsSHA256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`. Use normalizedpatchtokensonly,discardCLS+4registers. RGBstretchedto1152square/no crop→72×72patchgrid tomatchoriginalSAM3FPNgriddensity. Projectionviaoriginalcalibration/bilinearsamplingalign_cornersFalse. BackboneBF16,learnedfinalnormappliedFP32,selectedsampledfeaturesFP16. Noprompt/masks/fine-tuning. FullfrozenViTforward isexecuted whenextractingcachedintermediatefeatures; no earlyexitoptimization.

## Selected hidden state

**block6** selected using64train/32validationfrozenfeaturebalanced5classlinearridgeprobe,seed177. Foreground4colorIoUcriterion;notestsetselection. GTmasksonlyprobe-labelsupervision,notbackbone/grounderinput. Probeweights discarded,notusedbyfullmodel. This quantifieslinearlyreadablecolorgrounding,notuniversalmaximalinformation or a directmetricshapeprobe.

| Candidate | Validation foreground mIoU % |
|---|---:|
| block6 | 72.54 |
| block12 | 71.10 |
| block18 | 70.68 |
| block24 | 70.09 |
| mean_last4 | 71.45 |

Eachlayernormalizedwithfrozenpretrainednorm;mean_last4averagesnormalizedblocks21–24. Selectionsourcecommit `cff3081f64cc3148cd531221b6b9ec322e613f31`. Rawprobeconfusions/projectioncheck/dataindices infeature_selection.json andprobe/summary.json.

## Training / precision

Inheriteduserbudget **30000updates**, B8scenes×4tasks=32queries. Alltrainablemodulesfromscratch,onlyDINOv3frozen. EMDonlyexactunsquaredHungarian128pointdistanceinmetres,reportmm. AdamW3e-4→3e-5cosine30k,warm100,wd.01,clip1,seeds170. NoSE3/rotation/translation/XYZscaleaugmentation;E157hasphysicallyresizedcuboidswithconsistentRGB-D. Canonicalvalevery1000selectsbest,step0eligible;SE3diagnosticnotselection. Savebest,last,fulloptimizer/RNGresume+18kmilestoneonly.

BF16autocastforheavy scalarMLPs/transformers;masterweights/optimizer,vectorchannelmix,metricXYZreadoutandHungarianEMD remainFP32. Primaryvalidation/testFP32. This avoidsquantizingabsolutemetricXYZduringBF16bmm. Fixed-observationSE3proof isevaluatedinFP32,not a claim BF16arithmeticisexact. Backbone/visualwidth/resize/precision/LRhorizon differ fromE175,notstrictbackbone-onlycausalablation.

FullB8preflightpassed6BF16forward/backwardEMDupdates,allparametergradients/encoder/decoder/composedSE3/task/permutationchecks: 0.16s/step,peakallocated11.219GiB. No preflightweights saved. RealOODsmoke sphere+camera+30/freshDINOfourtaskspassed.

Latestrecordedstep4450,meanlast50trainEMD5.126mm,peak11.22GiB. Validationstep4000: EMD5.419mm,centroid2.451mm;SE3EMD5.419mm,pointwiseRMS0.000169mm. Earlymetrics, not finalquality. Currentstate in train/progress.json/validation_progress.json.

## Automatic tests after30k

Best/18k/last fulltrain/val/IID/positions/rotations/sizes/combinations plus4globalSO3+translationviews ofall128IID×4tasks,RGB/DINO/UVassociationfixed. ReportrawEMD/centroid,directionalCD,alignedCD,robustextent/RgMAE,R²,visibleGT4mmvoxelcoverage2/5mm,forcednearestGTcentroidselection andpointwiseSE3drift. No GTcorrectionexceptexplicitlabeledaligned/unitdiagnostics.

Bestonly: sameE175physicalshapesphere/ellipsoid/cylinder/capsule128×4querieseach; samephysicalcameraoriginalresampled/yaw±15/±30/elevation+15,128×4each. FreshselectedDINOfeatures/UVgraphs foreachobservation, no GTinputs/adaptation. Then128IID×4tasks baseline/othersceneindexswap/UVswap/ownscenepermutation,donorBxor1/seed17562,recipientgeometry/task/UVgraphfixed. Allqueriesprimary;densitysubsetsdiagnostic. Camera/shapeoutcomesnotusedforcheckpointorlayerselection.

PartialvisibleGTonly, fixedclutter/idealdepth/singletrainseed;newcameraviewpointnotcoveredbycoordinateSE3guarantee. Shapesdifferentvolumes/visibility, sizesdistributionhasoverlappedsupportafterphysicalanisotropicscale. No realworld/hiddencompletion/multi-seedclaim. Tests pending until completion.json.

## Source, cache and run

Code branch exp/e177-dino, currenttrain/testsource `02b6078df115c69d2a46f02bffe9b8731150e8a0`. SelectiveownE175/E176utilities copiedwithhashes,sameexperimentimports only,noGVPvendoredcode. Actualconfig /workspace/robotics_assets/experiments/e177_dino/train/config.json. Layerfeatureconfigimmutable;sourcecachecommitcb03467 andSHAmanifest preserveextractionidentity. No completedE175code/checkpoints changed.

DINOtrain/valfeaturecache~9GiB,persistent. Heldouts1GiB/splittemporary,deletedafterevaluation;IIDkeptuntilfeaturecontrols thenremoved. SourceSHA/calibrationrecordsremaininfeatures/manifest.json. No per-imagebackboneforwardduringtraining;cachedfeaturesused. Frozenweightsextractedseparately,notall300Mbackboneactivationskeptduringgroundertraining.

Run sequence:`code/probe.py`→`code/cache.py`→`code/pipeline.py` with `/workspace/arm-manipulation/.venv-concerto/bin/python` fromE177root. Freshstageoutputsrequired; explicitpipeline.py resume onlyforinterruptedtraining/samecode/30khorizon. Currentattachedtmux **ssh_tmux:e177-dino**,stdout/stderrmirroredrun.log;probe.log preserved. AutomaticRESULTS.md,docs/research/e177_results.md,completion.json andviewers localhost:17070/e177.html,e177_shapes.html,e177_camera.html afterfulltests.

[Config](/workspace/robotics_assets/experiments/e177_dino/train/config.json) · [Featureselection](/workspace/robotics_assets/experiments/e177_dino/feature_selection.json) · [Model](/workspace/robotics_assets/experiments/e177_dino/code/model.py) · [Log](/workspace/robotics_assets/experiments/e177_dino/run.log).

## Preparation notes (historical)

# E177 — E175 architecture with frozen DINOv3

Status: preparing featureselection/train30kBF16. E176stoppedbyuser,14.6klogged/14kresume/best9kval4.3885mm. DINOv3ViT-L16officialHFweightsdownloaded,pinned ea8dc2863c51be0a264bab82070e3e8836b02d51. Comparehiddenblocks6/12/18/24/meanlast4 on64train/32valsmalllinearprobe,notestselection. First feature precheck caught dtype mismatch: DINO finalnorm underautocastreturnsFP32 while directnorm(BF16hidden)outsideautocast returnsBF16. Use explicitFP32learnedLayerNorm for allcandidates, assertfinalblock matchesofficiallast_hidden_state exactly. IncompleteprobeoutputonlyREADMEremovedbeforerun;no trainedweights/cachelost.

Integrationpreflightpassed871858trainableparameters,B8BF16/FP32EMD0.157s,11.219GiBpeak. OODsmokecaughtdynamicUVgraphuint16Torchindexdtypebeforetraining;explicitcasttolonginOODcaller. No trainingstartedbeforefix;fullcommittedpipeline rerun withupdatedprovenance.

