# E173 — FM plus global translation-velocity penalty

Status: training18k Gaussian FM, Z+memory FP32.

User-confirmed objective: Lnew=LFM+Lglobal=2Ltranslation+Lrelative-deformation. This supersedes the earlier1:1/log-only choice. For error e=vpred−vtarget and per-query particle mean m=mean128(e), code computes FM=mean(e²), translation=mean(m²), relative=mean((e−m)²), total=FM+translation. Mean includes batch and3coordinates for every term, preserving the scale of E171's original FM component (the vector-norm formulation is divided by3). Effective translation:relative weights2:1. This is instantaneous velocity supervision, not an endpoint centroid-position loss; no GT centroid/Rg enters the model. Log total,originalFM,translation andrelative separately every50steps.

Fresh18000steps from same exact E171 initial parameters, not a resume. Same580483trainable FM/adapters, frozen E168best16500 grounding862080parameters produces Z128×128/task + shared scene memory4096×128.128Gaussian particles,2layers,width128,4heads,self-attention andcondition cross-attention; no particle-index embedding. No architecture/conditioning/output normalization changes.

Data E157 unchanged fullscene RGB320×240/4096XYZ withrobot/chair/rug and4colored cubes;1024train scenes×4tasks,128val and128each IID/positions/rotations/sizes/combinations. GT128visible targetpoints. Canonicalview0 only fortraining (noSE3);4IIDSE3views are OODtest. Frozen source encoder originally SE3-trained. E170cache reused read-only,FP16storage castFP32; no new feature caches. Source E168best16500 SHA25665527d6657e57aef6552c01bf1d33c2c9014c88a08a4c332693342f71e776207.

Train FP32/TF32off,B8scenes×4tasks,GaussianN(0,I)metres→targetXYZ. Same exact unsquared Hungarian source↔GT pairing/linearCFM/timestep1−(Beta(1.5,1)×.999+.001). AdamW3e-4→3e-5cosine,100warmup,wd.01,clip1. Same model/data seed170,noise167; consume original view draws then force0 to preserve E171scene/noise/timestep stream. Only lossweight changes. E17250k continuation remains paused at user request and is not touched.

Validation: canonicalval128×4,20Eulersteps,fixedGaussianseed167,every500,minimum worldEMD selectsbest,initialstep0 eligible. SE3val diagnostic only. Primary compare E173last18000 vsE171last18000;best-vs-best also use same canonicalcriterion. Autoevaluate train/val/5heldoutcanonicalsets+4IIDSE3views. Report EMD,centroid,alignedCD,directionalCD,unitshapeEMD,Rgerror,robustXYZextents,surfacevoxel4mmcoverage@2/5mm,targetselection. Generate losscomponent curve andbest-vs-GT viewer. One matched seed; no guaranteed endpoint centroid improvement from velocity reweighting.

Own code on branch exp/e173-trans2 from65c7f27, selective E171 source hashes in sources.json,no imports from other experiment code. Exact initialstate/cache/source/fullGPU and loss-decomposition/gradient checks before training. Commit recorded beside artifacts. Run `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py verify` then `... code/run.py train` from root; trainer creates fresh train directory and autoevaluates. Attachedtmux ssh_tmux:e173-log,stdout+stderr mirrored to run.log. Estimated40min+evaluation; existingforwarded viewer localhost:17070/e173.html oncompletion. No result claimed yet.

Artifact status: training18k Gaussian FM, Z+memory FP32. Executing source commit: `d06e77d9874480ec8d898bd6451ee3231c988ebb`.

Launch verified: objective values and gradients equal2translation+1deformation, new global gradient is uniform across particles; initialstate matchesE171, cache/source hashes/alignment/permutation/KV/fullGPU updates passed.580483trainable,862080frozen; fullB8median0.11835s/update,peak7.445GiB. Train running18k, no result yet.

Paused explicitly by user after job stopped around9.1k; lastfullstatecheckpoint9000. No18kresults/evaluation. Keep stopped.
