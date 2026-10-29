# E176 — Vendored GVP core and SAM3/XYZ adapter

Status: fullmodel integrated; authorized30ktraining currentlyrunning. The module-only preparation below is historical; see current training section. User asked to take an actual portion of the authors code after discussion of replacing E175UVgraph/customencoder with a published local equivariant architecture.

## Reused code

Official repository https://github.com/drorlab/gvp-pytorch, pinned upstream commit `316eea7cd6446dd928e5975634152ec3b28e65f5`. Vendored gvp/__init__.py (355lines) byte-for-byte and original MITLICENSE; authors copyright/permission notice retained. Modules include GVP,GVPConv,GVPConvLayer,vector-aware normalization/dropout and tuple utilities. No protein task models/data/weights downloaded. Files/URLs/SHA256 in code/vendor/gvp/manifest.json. This is actual upstream code, not a reimplementation under the same name.

Owncodebranch exp/e176-gvp, sourcecommit `7aa46bb60f1c59f0cb9fe5e5a6cbf254fa89abdd`; isolatedroot `/workspace/robotics_assets/experiments/e176_gvp`. Original E175code/checkpoints untouched.

## Our adapter

code/encoder.py is our wrapper, separate from unchanged upstream core. Inputs: fullsceneXYZ[B,N,3],frozenSAM3features[B,N,256],16neighborindices[B,N,16] from actual3DkNN. FeatureLN256 + upstreamGVP((256,1),(128,8)); vectorinput=(P−sceneMean)/0.1m. Two upstreamGVPConvLayers((128,8),(9,1)),3GVPmessage/2GVPfeedforward,vector_gating=True,dropout0. Edges:scalarlogdistance+8distanceRBFs,vectorunitrelativeXYZ. Outputscalar h[B,N,128],vector V[B,N,8,3]. No taskID here; planned existingGroundedDecoder would add taskconditioning.

No UVgraph or fullN×Npointselfattention. UV remains necessary before thisencoder toproject/sampleSAM3features. Graph built from rawXYZ/FP64 withSciPycKDTree, no per-axis geometrynormalization; features remainattached to samepoints under globalframechanges. Neighborhoodties/floatingpoint can affectkNNindices; reported rather than assuming exact numerical graphidentity. Training with graphcache could avoid per-forward CPUneighborsearch, but no trainer is implemented at this stage.

Encoderadapter trainableparametercount: **490626**; this is encoderonly, not a completegroundingmodel parametercount. Initialweightsrandom seed176. No checkpointload oroptimizer/weightupdate. No claimed reconstructionquality/convergence/camerarobustness.

## Verification

Actualfirst2E157IIDscenes,4096points/scene and matchingSAM3feature256. FP32/TF32off,RTX3090. Preservedgraph underfullSO3+translation, rebuilt3Dgraphdiagnostic, pointpermutationwithgraphremapping, batchisolation, outputshape/finiteness, one realrandomprojectionbackward withoutoptimizerstep. Upstreamfiles matchpinnedSHA256. Allchecks passed.

| Check | Result |
|---|---:|
| Scalar invariant max absolute error | 1.669e-06 |
| Vector equivariant max absolute error | 7.749e-07 |
| Rebuilt kNN scalar/vector max error | 1.907e-06 / 7.749e-07 |
| Rebuilt kNN changed ordered slots | 170 / 131072 |
| Permutation scalar/vector max error | 1.431e-06 / 3.576e-07 |
| Batch-isolation scalar/vector max error | 1.192e-06 / 4.768e-07 |
| Input/layer0/layer1 gradients | finite, allpositive norms |
| Encoder GPU forward, B2×4096, median | 15.96 ms |
| Graph CPU build,2scenes | 28.22 ms |
| Peak allocated during B2 backward | 1.977 GiB |

Benchmark is encoderonly, excludes SAM3/groundeddecoder/readout/HungarianEMD. Ordered-neighbor slot differences can include reorderings of the same neighbor set; no claim all170slots changed membership. Errors are rawfeature-unit errors, not millimeters or reconstructionmetrics.

## Dependencies and usage

Installed torch-geometric==2.7.0 withuvpip into existing .venv-concerto plus its17missingpackages; torch stayed2.7.0+cu128. Usesexistingtorch_scatter2.1.2+pt27cu128,scipy1.17.1,numpy2.2.6. No torch_cluster,proteinextras or globalinstallation required. Exactversions in verification.json.

From E176root: `/workspace/arm-manipulation/.venv-concerto/bin/python code/verify.py` creates freshverification.json; existingresult is preserved, so the script refuses to overwrite it. Live tmuxwindowssh_tmux:e176-gvp andrun.log retained. No checkpoints/featurecaches generated. Root/codeREADMEs contain actualconfiguration and status.

[Upstream core](/workspace/robotics_assets/experiments/e176_gvp/code/vendor/gvp/__init__.py) · [License](/workspace/robotics_assets/experiments/e176_gvp/code/vendor/gvp/LICENSE) · [Adapter](/workspace/robotics_assets/experiments/e176_gvp/code/encoder.py) · [Checks](/workspace/robotics_assets/experiments/e176_gvp/verification.json).

Next fullmodel would consume h/V using taskconditionedgroundeddecoder andcoordinatehead, then need its own training/budget/metrics. This step establishes usable importedmodules/interface/equivariance and gradients, not a trained successor to E175.

## Current authorized30k training run

Updated 2026-10-28T12:57:52.445225+00:00. User requested trainandtest30000steps. Sourcecommit `5178f29f03c9fe33a6189e7444b0f603e03299c3`. ScratchGVPencoder + copiedE175GroundedDecoder/coordinatehead, total1021450trainableparams (encoder490626). OnlySAM3frozen; no pretrainedE175/proteinweights. Fullscene4096,128outputs,1024train/128val/5×128heldout,4tasks;originalE157camera/robot/chair/rug andphysicalcuboidresize data. 3DkNN16, noUVgraph/fullpointselfattention.

Training:30000updates,B8scenes×4tasks=32queries,FP32/TF32off,onlyunsquaredHungarianEMD,inmetres. AdamW3e-4→3e-5cosine30k,warm100,wd.01,clip1,seeds170. No SE3/rotation/translation/scaleaugmentation. Canonicalvalidationevery1000selectsbest;SE3diagnosticnotselection. Saveoverwrittenbest/last/resume plus18kmilestone. Fulloptimizer/RNGresumecheckpoint. Config/workspace/robotics_assets/experiments/e176_gvp/train/config.json.

Newfullmodelpreflight passed:3Dgraphhashes/permutation,scalar/vector+coordinatehead+composedSE3,6realB8FP32EMDupdates withfinitegradients; time0.336s/step,peakallocated10.403GiB. These are throwawaypreflightweights, not partofactualtraining. OODsmoke withfreshSAM3sphere/camera+30and4tasks passed. Sourcepreflightproof /workspace/robotics_assets/experiments/e176_gvp/preflight.json.

At recordedstep750, meanlast50trainingEMD16.755mm,elapsed249.7s,peakallocated10.507GiB. Earlytrainingmetrics,not finalresults; readtrain/progress.json/validation_progress.json forcurrentstate. At~0.34s/step trainingbudget roughly2h50min plusvalidation/tests; estimateonly.

Automaticpipeline after30k: best/18k/last allcanonicaltrain/val/IID/positions/rotations/sizes/combinations and4globalSE3views with3DkNNrebuilt; thenbest-onlysame2048newshapequeries and3072cameraprotocolqueries asE175 withfreshSAM3/new3Dgraph; then2048featurecontrolqueriesbaseline/indexswap/UVswap/ownshuffle. ReportEMD,centroid,directionalCD,alignedCD,visibleGTvoxelcoverage,robustextent/Rg andpointwiseSE3drift. No testselection/GTinputs. UseE175baselinesforcontext,notisolatedgraphablation becauseencoder/paramcount/LRhorizon change.

Livewindow ssh_tmux:e176-train,stdout/stderrmirrored /workspace/robotics_assets/experiments/e176_gvp/run.log. Originalmodulechecklogpreservedasmodule_check.log. Entrypoint `/workspace/arm-manipulation/.venv-concerto/bin/python code/pipeline.py`; explicit`code/pipeline.py resume` for interruptedtrainingonly,samesource/30khorizon,fullRNG+optimizer. No outputoverwrite. AutoRESULTS.md,docs/research/e176_results.md,completion.json andforwardedviewers/e176.html,/e176_shapes.html,/e176_camera.html onlyaftertestscomplete. Trainingandtestsnotyetcomplete atthisnote.

First realvalidation saved atstep1000: canonicalEMD10.6071mm,centroid3.3489mm;SE3EMD10.6071mm,fixedgraphmonitorRMS0.0008906mm/max0.0031779mm. Best and fullstate resumecheckpoint saved atstep1000. Atlastcheckstep1300,meanlast50trainEMD6.7176mm. Training remainsrunning; fulltestpending.

Stopped by user. Lastloggedstep14600,fullresumestep14000,beststep9000/valEMD4.3885mm. No30kcompletion or fulltests; switching to E175+DINOv3. Checkpoints/logs preserved.
