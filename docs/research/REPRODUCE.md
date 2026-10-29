# Current independent reproduction pipelines

## Voxel reach → point-cloud pick → SAM3 (E118→E131→E133)

Use `/workspace/arm-manipulation/sam3_reach_pick/README.md`. Code restores the original required symbols from Git and uses paths relative to that bundle. Includes voxel/expert generation, data export, split/normalization reconstruction, direct point-cloud/action cache, E118/E131 training, GT/SAM3 evaluations and replay. No E122 RGB/ViT policy is needed.

Retained data, E118 best120000 checkpoint, labels and XY16 states are under `sam3_reach_pick/runs/assets/`. Official pinned SAM3 weights were downloaded again. E131 final23520 weights were deleted in the earlier user-authorized post-E122 cleanup and remain unavailable; retrain with `.venv/bin/python run.py pick_train` from that bundle. Historical93.32% is not a new result. E178 and its data/weights remain independent and unchanged.

## Unified geometry + semantic (E178)

Use `/workspace/arm-manipulation/unified_geometry_semantic/README.md` for code-based regeneration/train/test. Retained original E178 weights, DINO features and relocated data are in `/workspace/robotics_assets/experiments/e178_dino24/`, with `current_assets.json` and `research.py` for ongoing analysis.

All historical experiment folders before E133 now retain Markdown summaries only. Full cleanup inventories live in robotics_assets/experiments/_maintenance/. The former recipe with old absolute paths is preserved only as historical provenance in sam3_reach_pick/reference/historical/REPRODUCE_legacy.md.
