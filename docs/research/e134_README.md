# E134 — ManiSkill 3 task gallery

Status: COMPLETE.

Install official ManiSkill3.0.1/SAPIEN3.0.3 in isolated `.venv-maniskill`, sharing the existing PyTorch2.7.0+cu128 runtime without modifying the arm-training environment. Assets live under `/workspace/robotics_assets/external/maniskill`. Use the existing NVIDIA Vulkan ICD `/etc/vulkan/icd.d/nvidia_icd.json`.

Export the installed environment registry with task class documentation, default kwargs, supported robots, maximum episode steps and asset dependencies. Render24representative task reset scenes (seed0), including tabletop pick/push/pull/stack/assembly/tool use, dual-arm, drawing, dexterity, quadruped, humanoid manipulation and simple control tasks. Task defaults are retained. Each task runs in a separate process so SAPIEN/Vulkan resources are released between tasks. Download only assets required by those selected tasks; no full demonstration dataset or large room-scene benchmarks.

No training, checkpoints, FM, action chunks or policy evaluation applies. These are initial task-scene renders, not solved rollouts or success-rate tests. CPU physics with one environment, NVIDIA Vulkan rendering, `state` observations, `rgb_array` render mode,640x480human camera override, default minimal raster shader. Capture directly after reset without actions. All screenshots must be nonconstantRGB and all24must finish before marking complete.

Save native PNGs, four categorized contact sheets, a searchable local HTML gallery, installed task catalog JSON/Markdown and exact environment/config/source hashes to `/workspace/robotics_assets/experiments/e134_mani`. Renderings do not test E131 transfer to the different default robot embodiments.

Run from repository root:
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json MS_ASSET_DIR=/workspace/robotics_assets/external/maniskill MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 OPENBLAS_NUM_THREADS=1 .venv-maniskill/bin/python -u -m maniskill_tasks.preview`

An interrupted render resumes completed PNGs under the same configuration; completed galleries are preserved. Run with tmux and tee. Official references: https://maniskill.readthedocs.io/en/latest/tasks/ and https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html .

Asset availability: storage1.ucsd.edu is unreachable (verified12sconnection/TLS timeout; upstream issue https://github.com/mani-skill/ManiSkill/issues/1489). PickClutterYCB,AssemblingKits,TurnFaucet,OpenCabinetDrawer remain listed in the installed catalog but cannot be rendered here with official assets. Four available robot/control variants replace them in the24scene gallery; task definitions are unchanged. The interrupted PickClutter download produced no image; prior2successful renders are retained.

Image validation: nativeRGB640x480uint8,standarddeviation>1 and dynamicrange>10. Cartpole is a valid low-contrast scene (std3.16), so its originalstd>10previewguard was too strict; verified visually before correcting this check. No task camera or geometry was changed. Asset-download socket timeout30s.

Installed registry:74IDs; rendered:24task scenes. Exact configuration and individual robot/action spaces are saved.

Gallery assets:overview.jpg(all24),pick_place.jpg(7),tools.jpg(6),assembly.jpg(4),other.jpg(7),index.html(searchable24cards),task_catalog.md/.json(all74registeredIDs). All24PNGs and5contact sheets decoded successfully; originalMLenvironment still has no mani_skill installation; pipcheck passes in isolatedenv. No training/policy test run.
