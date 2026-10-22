# Reusable dataset tools

All commands run from repository root with `.venv/bin/python`. Choose fresh
outputs; the curated datasets are read-only source assets. No historical
experiment directories or deleted e55/e91 data are required.

Voxel grid ->physical trajectory audit ->H25EE-delta chunks ->joint inputs:

```bash
python -m OATFlow.data_tools.voxel_grid --output /tmp/voxels --workers 8
python -m OATFlow.data_tools.audit_voxels /tmp/voxels --workers 8 --replay 16
python -m OATFlow.data_tools.export_ee /tmp/voxels /tmp/voxels/chunks --workers 8
python -m OATFlow.data_tools.export_joints --chunks /tmp/voxels/chunks
```

Use --limit8 --workers1 for a voxel smoke check. Exact retained user physics
reference and IK initialization seeds are in robotics_assets/robot/nexarm.
Chunks store physical deltaXYZ+relative rotation6D+absolute jaw,tail mask and
measured EE/joint anchors. No normalization is applied on disk.

Shuffle collection ->RGB3/masks ->full audit:

```bash
MUJOCO_GL=egl python -m OATFlow.data_tools.collect_shuffle --output /tmp/shuffle --workers 8
MUJOCO_GL=egl python -m OATFlow.data_tools.render_shuffle --data /tmp/shuffle --workers 8
MUJOCO_GL=egl python -m OATFlow.data_tools.audit_shuffle --data /tmp/shuffle
```

Add --pilot to each shuffle command for one context/four queries. Full recipe:
40layouts,24permutations split16seen/8unseen,928shared contexts/3712queries,
original three cameras,diverse shuffle trajectories/speeds,successful physical
expert pick with no mid-rollout perturbation. Source scene is the retained
shuffle scene. Core expert/environment/controller modules remain shared.
