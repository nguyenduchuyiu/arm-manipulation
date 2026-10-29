# Point-cloud code cleanup

Status: completed, then curated further at the user’s request.

Current source index, retained implementations, discarded runs and lessons: [pointcloud_retained.md](pointcloud_retained.md).

The initial relocation removed21 untracked experiment packages from OATFlow and made29 local bundles. After review the user requested keeping only useful SAM3/custom-encoder/PTv3/EMD/data implementations and important tests. The final set is6core+4diagnostics;19 redundant/unsuccessful code bundles, their private checkpoints/caches and unfiltered source backups were removed. Failed-run reports and numeric/visual results remain. Full SE(3) augmentation is retained.

Verification/manifests: `_maintenance/code_cleanup_20261027/curation_verification.json`, `curation_deletion.json`, plus each retained `code/migration.json`. Current retained code keeps only required local helper functions/classes. Old failed source backups are intentionally not retained; this supersedes the initial rollback plan.
