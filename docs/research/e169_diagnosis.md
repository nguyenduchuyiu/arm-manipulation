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
