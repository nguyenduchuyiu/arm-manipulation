# Reusable robotics assets

Status: complete; all moved assets and post-move checks passed.

User requested discarding e94 and moving the selected reusable datasets/models
out of experiment-number directories. Canonical root: /workspace/robotics_assets.
No training; preserve tensor/data contents, original metadata, RNG/optimizer
snapshots and evaluations. Original paths become symlinks for saved configs.
Only e94_full artifacts are deleted; unrelated old experiments remain outside.

Selected assets:
- datasets/reach_voxel_grid: e86 complete feasible grid, trajectories and H25 EE
  chunks, including joint_states/ consolidated from e88.
- datasets/shuffle_pick_multiview: e96 complete shared-context shuffle/pick RGB3,
  masks, centers, labels, full simulation state and layout/permutation splits.
- models/reach_joint_delta_ee: e88_joint/resume120k, best115000 plus resumable
  latest120000 and corrected physical evaluations.
- models/object_localization_topview: e99_top, best27000; frozen ImageNet ViT
  required. Original e55/e91 training data are outside this selected catalog.
- models/point_tracking: e100_point, best1250; point baseline, opening val23.099mm.
- models/point_tracking_local: e101_local, best1250; frame0 local appearance
  reference, opening val25.411mm. Saved live tracking->FM evaluation lifts one
  of three cases using a separate IK lift probe; tracking remains experimental.
- pretrained/vision: frozen ViT-B/16 and native TCOW initialization/provenance.
- robot/nexarm: visual/physics references, visual meshes, settled home, IK seeds.

Model weights, original config/normalization/split/provenance/test outputs stay
unchanged. catalog.json and per-asset asset.json provide canonical relative
paths, task semantics, measured results and limitations. Keep the paired latest
and optimizer.pt for e101; optimizer is not paired with best1250. e88 latest
contains optimizer/RNG/sampler state; current historical trainer is not claimed
as revalidated for resumption. final/latest are already hardlinks. Identical
reach evaluation checkpoint copies may be deduplicated via hardlinks after SHA
matching, preserving their paths/content.

Before moving, record file sizes/inodes/mtime; verify them immediately after
atomic directory rename. SHA256 manifest for reusable payloads and source
snapshot. Post-move strict model loads and representative inference for FM,
static top localization and both trackers; check old aliases, dataset counts,
joint alignment, split and e96 expert state preservation. No new rollout or
training is needed to validate a storage relocation. Existing corrected videos
and physical replay evidence remain beside each model.

Prepare source snapshot before updating active code paths; retain original
source hashes in historical experiment metadata. Active code uses canonical
locations. Artifact root README describes reuse, no claim of deployment or
universal tracking success. Metadata/migration.json, integrity.jsonl and
verification.json record actual work. Disk usage reported separately afterward.

Actual verification:82516payload files SHA256-manifested; all original rename inodes/sizes/mtimes preserved;4strict selected model loads and representative inference passed; both tracker saved validation predictions reproduced exactly with the original batch4precision setup.24focused tests pass. Exact settled home,reach resumable state,local tracker latest/optimizer pairing,all chunk measured-jaw values and original mesh EGL render passed. e94_full removed explicitly; two identical reach evaluation checkpoints now hardlink best, reclaiming839648572bytes. Curated allocated size 20.295GiB; remaining disk free 5.424GiB. Original metadata remains untouched. See catalog.json and metadata/{migration.json,integrity.jsonl,verification.json,disk_usage.json}.
