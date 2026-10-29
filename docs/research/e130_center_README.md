# E130 — oracle centroid alignment of saved predicted clouds

Status: COMPLETE.

Purpose: separate centroid displacement from residual surface-cloud mismatch
using the unchanged E130 predicted/GT clouds. This is an output-only oracle
diagnostic, not an inference solution, model intervention or coordinate-frame
change. For every sample, compute pred_center=mean(pred), gt_center=mean(GT),
then aligned_pred=pred-pred_center+gt_center. No rotation, scaling, ICP,
point resampling, clipping or shape deformation is applied.

Parent checkpoint: E122 final23520; original training used 2048 E117 demos,
32 layouts, stride1, batch128, 23520 updates (~10.217epochs), frozen ViT-B/16,
FM initialized from E118 reach120000. Full original training configuration
is retained in ../config.json. No weights are loaded, no optimizer/model is
constructed, and no training/epochs/batch/LR applies to this diagnostic.

Input: ../attention.npz and ../attention_records.json, 75 saved snapshots.
Primary comparison: frame0 of all8 physical inside/outside pairs from the
X-boundary pilot, four colors x two edges. Original traces used H25/K15,
Euler10/seed0/300frame budget; original cloud inference was CUDA FP32 with
TF32 disabled. None of these action settings is executed in this offline
analysis. Later snapshots are reported separately, because their trajectories,
occlusion and sample counts differ. Both ID and OOD clouds are aligned for
the fair residual-shape comparison, in addition to their original scores.

Metric: unchanged E130 symmetric, UNSQUARED Euclidean Chamfer in mm:
0.5*(mean nearest pred-to-GT distance + mean nearest GT-to-pred distance).
GT is the observed front-camera target surface, not a complete object mesh;
its mean is a visible-surface centroid, not the physical object center.
Residual Chamfer includes orientation, scale, sampling and coverage mismatch.
The raw-to-aligned Chamfer difference is not an additive variance/error
budget, and centroid alignment need not minimize Chamfer under unequal
surface sampling.

Run `.venv/bin/python -u -m OATFlow.top_pick.center_probe` from the repository.
Output: /workspace/robotics_assets/experiments/e130_probe/test_center.
Artifacts: config.json, summary.json, records.json, aligned_clouds.npz,
chamfer_comparison.png, aligned_cloud_xy.png, aligned_cloud_xz.png,
verification.json. Verify original Chamfer parity, equal aligned centroids,
pairwise-distance preservation, synthetic pure-translation behavior and
unchanged source/checkpoint hashes. Results below.


## Results

| Group | N | Raw Chamfer | Centroid-aligned Chamfer |
|---|---:|---:|---:|
| initial inside | 8 | 3.0580 mm | 3.3823 mm |
| initial outside | 8 | 13.5539 mm | 3.5996 mm |
| later inside | 27 | 25.9887 mm | 3.7996 mm |
| later outside | 32 | 39.4175 mm | 3.6013 mm |

Initial paired mean aligned OOD minus ID: 0.2174 mm.

All eight initial OOD clouds have aligned Chamfer below 4 mm. This supports
prioritizing localization over additional shape-decoder work for the tested
fixed-size cube task. It does not establish reconstruction of arbitrary
object geometry, full hidden surfaces or correct physical object centers.
The slight increase in ID Chamfer after centroid alignment is compatible
with unequal point sampling/coverage: centroid matching is not a Chamfer
optimizer. Source/checkpoint hashes, raw-score parity, equal centroids and
pairwise-distance preservation all pass; exact values in verification.json.

This test also supersedes the earlier interpretation of E130's extrinsic
translation probe: that probe changed the entire cloud coordinate mapping
without transforming the other inputs consistently. It measures response
to a counterfactual input, not a clean target translation or an equivariance
requirement for the physical pipeline. The paired rendered scenes and this
output-only oracle alignment give the direct localization evidence.

Figure-only revision: XY/XZ axes span +/-30 mm to include every saved point;
metrics are unchanged. The original metric-source hash and updated figure
source hash are retained separately in config.json.
