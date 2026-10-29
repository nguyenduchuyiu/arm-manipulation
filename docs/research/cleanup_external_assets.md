# Remove unselected checkpoint and dataset payloads

Status: complete; all curated payloads unchanged.

User explicitly requested deleting all checkpoint/dataset artifacts outside
/workspace/robotics_assets. Protect the entire curated root and its payload
manifest. Prune /workspace/checkpoints and /workspace/datasets, preserving only
symlink aliases into the curated root and parent directories needed by aliases.
Remove obsolete generated shuffle pilot/chunk-smoke datasets and NumPy dataset
payloads under workspace_analysis/results. The identical IK seed copy becomes
an alias to robot/nexarm/ik_seeds.npz. Preserve source code, original robot assets,
software environments and canonical research documentation.

No training. Verify curated file inode/size/mtime before and after deletion,
confirm all legacy roots contain only aliases, and scan for remaining generated
checkpoint files outside curated assets (excluding environment/package assets).
Record exact deletion paths and actual reclaimed disk space in
robotics_assets/metadata/cleanup_external.json. Historical experiment metadata
remains immutable; current catalog marks old e55/e91 training data as deleted.
Model inference does not require those old training datasets. Retained tracker
baselines remain experimental; point_tracking is e100 initial-dot TCOW;
point_tracking_local is e101 initial-dot plus frame0 3x3patch appearance reference.

Actual reclaimed 58.397GiB; free 63.819GiB. 82516original curated files unchanged before current-catalog availability annotations. No generated external training checkpoint files remain; legacy data/checkpoint roots contain only aliases.
