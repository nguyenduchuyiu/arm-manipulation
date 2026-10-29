# Independent experiment layout validation

Both projects now contain src/configs/assets/tests/outputs, a short README, requirements and run.py. Datasets and HF caches are shared resources configured independently; outputs and Python environments are excluded from source commits.

Validation: 6/6 current-resource regression tests passed. Both project copies also passed their data-free core tests from isolated temporary directories (retained integrations explicitly skipped there). Isolated voxel generation accepted2goals; isolated E178 source/assets rendered RGB-D/XYZ/GT. All61SAM3 and34geometry modules imported from their own src trees. Core model/loss AST definitions are unchanged. Dataset/checkpoint relocation checked inode/size preservation. AGENT.md was not changed.

Current venvs: /workspace/.venvs/sam3_reach_pick and /workspace/.venvs/unified_geometry_semantic.
