# E169 — Oracle centroid/Rg + frozen all-scene Z + flow-only reconstruction

Status: complete18k, oracle evaluation and diagnostics complete.

Purpose: remove centroid/Rg prediction error, competing regression losses and changing upstream conditions. Use **GT128 centroid and Rg**, cache task-conditioned Z for every scene, freeze grounding, train only a fresh flow head to reconstruct oriented unit shape. This is an oracle diagnostic, **not GT-free inference**.

Source/data: E168 best16500 checkpoint, SHA65527d6657e57aef6552c01bf1d33c2c9014c88a08a4c332693342f71e776207; frozen original4096-point/SAM3/task grounding862080parameters. Original frozen SAM3 vision_features remain cached in E157; no rerender/re-encode images. Same robot/chair/rug, original cam_front RGB320×240, physical cube size/pose variation. E157 train1024,val128,five heldout splits128scenes each (1792total),4color tasks,GT128visiblepoints. No GT information in Z extraction; full scene/point-feature/task inputs only.

Cache: all1792scenes×5views×4tasks=35840fixed Z matrices[128,128]. View0 original, views1–4 fullHaarSO3+translation[-.5,.5]m applied to scene and GT with unchanged SAM-feature association. Per-split CPU seed169000+splitindex; valview1 and allIIDSE3views are exact E137 transforms. Cache each transformed forward separately; do not rotate latentZ. Roughly2.2GiB for float32Z plus labels. BF16 extraction matches original E168 forward; store its output exactly as FP32. All splits remain disjoint: caching heldout features is read-only and does not train on them. Finite cached SE3 views replace fresh online draws because encoder is frozen.

Oracle labels: cGT=mean(GT128), rGT=Rg(GT128), YGT=(G-cGT)/rGT. Rg is visible-cloud RMS spread, not cube side/bounding radius. Model condition is fixed cachedZ; cGT/rGT are only for normalization and oracle world conversion. Initialization in world is cGT+rGT*U. U has64random unit directions plus antipodes, giving128points/centroid0/Rg1. Target orientation remains in robot axes; no canonical rotation.

Flow: unchanged E168 width128/two-layer/four-head point self-attention +cross-attention to128 Ztokens, FF384/dropout0; shared XYZ/time projections, **no learned particle-index embedding**. Fresh513923parameter head, seed169, zero velocity readout, no trained flow weights loaded. **Only velocity MSE**, exact unsquared Euclidean EMD/Hungarian U↔YGT, t=1-(Beta(1.5,1)*.999+.001). No center/Rg/pair/reconstruction auxiliary loss. No encoder/centroid/size head or upstream optimizer exists in the FM trainer.

Train:18000updates, batch8cached scene views×4tasks=32queries, trainviews sampled uniformly. 20480train scene-view/task examples, about28.125passes over that pool. AdamW3e-4→3e-5 cosine/warmup100, wd.01/clip1. Data/beta seed169, CUDAunit-direction seed167. **FP32 flow attention/MLPs/parameters/optimizer/MSE/Euler**, TF32off; avoid prior BF16 particle-order numerical sensitivity. Original cache never modified. Validate every500 on cachedvalview1 (E137SE3), select minimum **unit-shape EMD**, report canonical too. Initial sphere/step0 is eligible; report last18k separately if best remains0.

Inference:20positive Eulersteps in unit coordinates, conditionZ computed/cached once. Project only endpoint to centroid0/Rg1; worldoutput=cGT+rGT*Yunit. Do not renormalize intermediate states. Fixed per-task sphere bank167 rotated with each cachedR. Output centroid/Rg must equal suppliedGT parameters; near-zero errors are expected by construction, not grounding success.

Evaluation planned: fulltrain canonical, valcanonical, five canonical heldout128scene splits,4IIDSE3views. Compare oracle sphere, E169best, E169last18k, **old E168flow using the same cachedZ/FP32/oracle parameters**, plus E169best with Z rolled between scenes within each8scene batch, preserving taskID and oracle parameters. Report unit-shapeEMD/CD, oracle worldEMD/directionalCD, robustextent, fullvisible-surface voxelcoverage2/5mm, rawendpointRg; no target-selection accuracy because oracle centroid makes that uninformative. Best-vs-GT viewer includes fixed train/IID examples, sphere toggle.

Limitations: oracle GTstatistics invalidate a direct end-to-end comparison with E168's predicted-head numbers. Frozen E168Z could lack detailed geometry/orientation, so failure does not isolate FM capacity. Finite cached views and one source-noise bank; visible surfaces only. Source/hash/frozen-state/cache consistency, realgradients and FP32particle permutation must pass before training.

Ownership: isolated branch exp/e169-oracle based65c7f27, all variant code in own e169 package; selective actual source hashes in sources.json, no cross-experiment imports. Commit code before execution; exact executing commit recorded in manifest/config/README. Artifacts outside Git.

Commands:
```
/workspace/arm-manipulation/.venv-concerto/bin/python /workspace/robotics_assets/experiments/e169_oracle/code/run.py cache
/workspace/arm-manipulation/.venv-concerto/bin/python /workspace/robotics_assets/experiments/e169_oracle/code/run.py verify
/workspace/arm-manipulation/.venv-concerto/bin/python /workspace/robotics_assets/experiments/e169_oracle/code/run.py train
```
Live log `ssh_tmux:e169-log`, mirrored `/workspace/robotics_assets/experiments/e169_oracle/run.log`. Trainer automatically evaluates/reports and serves `http://localhost:17070/e169.html` after completion. Actual status in cache/manifest.json, verify.json and train/config.json.

Artifact status: training/evaluation complete; see RESULTS.md. Executing source commit: `907e501265502fac07d7d9554453ca8d8cdc182d`.

Actual results and limitations: /workspace/robotics_assets/experiments/e169_oracle/DIAGNOSIS.md

# E169 — oracle frozen-feature FM: result and failure localization

Complete18k, best14500. GT128centroid/Rg supplied; E168best16500 context frozen, all1792scenes×5SE3views×4tasks cachedFP32. Only fresh513923parameter flow trained, FP32 velocityMSE, batch8sceneviews/32tasks,18kcos3e-4→3e-5. No encoder/center/size update. See code/README.md, cache/manifest.json and train/config.json.

| IID protocol | Sphere unit EMD | Best FM unit EMD | Best + scene-shuffledZ |
|---|---:|---:|---:|
| Original frame |0.49079|0.37865|0.38025|
| Four SE3 views |0.49079|0.49721|0.50851|

Canonical IID oracle worldEMD9.936→7.642mm, coverage5mm38.58→56.90%. SE3worldEMD9.936→10.060mm, coverage38.63→40.57%. Pairedscene10kbootstrapFM-minus-sphere EMD95CI canonical[-2.362,-2.227]mm, SE3[+0.059,+0.190]mm. Not GT-free results. Main128surface statistics and cache/checkpoint hashes audited.

FM can deform the sphere into a useful canonical shape prior; it has not recovered the target-conditioned orientation under arbitrary coordinate rotations. Canonical scene-shuffledZ performs nearly the same, indicating limited scene-specific grounding despite clear geometry improvement.

Last18k velocity diagnostic (first32valscenes×4tasks, cachedSE3view1, same-task sphere bank identical across scenes): t0 correctZ MSE0.09650 vszero-velocity0.10353; t0.1 correct0.01387/shuffled0.01398; t0.5 correct0.01202/shuffled0.01210. Training states containing even10%GT shape allow strong denoising with little benefit from correctZ; pure-sphere initial velocity remains weak. This supports weak conditional orientation decoding; it is not proof of a defect in flow matching generally.

GTconstant-velocity teacher sampler control gives unitEMD~1.5e-6: matching/positiveEuler/label/output normalization reproduce the target numerically. No source or FM weights changed by diagnostics.

FrozenZgeometry probes (train-only standardization, val-selected ridge/MLP, scene-disjointIID): mean/std affine IIDcovarianceR²0.035, directional-extentR²0.052. Nonlinear256→256→128→19 readout101139parameters/5000steps: IIDcovarianceR²0.053, extentR²0.077; train0.107/0.128. Size readout quality does not establish oriented-shape information. These probes pool128tokens: they do not prove geometry absent from the full128×128 token set or distinguish information loss from a difficult readout.

Likely bottleneck localized to using the condition for geometry/orientation: currentFM receives only task-latentZ, no explicit per-point relativeXYZ. A next falsifiable test is direct normalizedXYZ+point-feature conditioning, preserving robot-axis orientation; do not claim this is already tested.

Viewer: http://localhost:17070/e169.html
Chart: http://localhost:17070/e169-why.png
Detailed diagnostic: test_cause/summary.json and test_cause/mlp/summary.json.
