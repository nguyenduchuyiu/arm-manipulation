> Historical snapshot before checkpoint cleanup. See [checkpoint_cleanup.md](checkpoint_cleanup.md) for current checkpoint allocation.

# Storage inventory

Measured: 2026-10-22T10:13:03.181037+00:00

Read-only scan; no datasets, checkpoints, caches or source files were deleted.

- Workspace allocated usage (`du -x`): **69.328 GiB**.
- Robotics assets: **61.614 GiB**.
- Scanned regular files: 152,046. Files at least100MiB: 81paths, representing 71distinct inodes.
- Largest-file values below are logical sizes. Directory totals are allocated usage from `du`, with hard-link deduplication across each invocation. Hard-linked test/final/latest checkpoint paths share storage; deleting one link alone may free no blocks. Overlay filesystem base layers also make root `du` differ from `df`; root-visible totals are not additional reclaimable writable usage.
- The current disk status was captured with `df`; workspace scan errors: 0.

## Largest workspace directories

| GiB | Path |
|---:|---|
| 61.614 | `/workspace/robotics_assets` |
| 31.927 | `/workspace/robotics_assets/experiments` |
| 22.822 | `/workspace/robotics_assets/datasets` |
| 7.607 | `/workspace/arm-manipulation` |
| 7.407 | `/workspace/arm-manipulation/.venv` |
| 5.878 | `/workspace/robotics_assets/models` |
| 0.864 | `/workspace/robotics_assets/pretrained` |
| 0.139 | `/workspace/arm-manipulation/docs` |
| 0.107 | `/workspace/.hf_home` |
| 0.107 | `/workspace/.hf_home/hub` |
| 0.074 | `/workspace/robotics_assets/external` |
| 0.036 | `/workspace/robotics_assets/robot` |
| 0.033 | `/workspace/arm-manipulation/assets` |
| 0.023 | `/workspace/arm-manipulation/.git` |
| 0.014 | `/workspace/robotics_assets/metadata` |

## Dataset and experiment directories

| GiB | Path |
|---:|---|
| 9.219 | `/workspace/robotics_assets/datasets/voxel_xyz_two_view` |
| 6.058 | `/workspace/robotics_assets/experiments/e103_center` |
| 5.622 | `/workspace/robotics_assets/experiments/e115_direct` |
| 5.468 | `/workspace/robotics_assets/datasets/pick_wide_rgbd` |
| 4.164 | `/workspace/robotics_assets/datasets/shuffle_pick_rgbd_dynamic` |
| 3.971 | `/workspace/robotics_assets/datasets/reach_voxel_grid` |
| 3.599 | `/workspace/robotics_assets/models/reach_joint_delta_ee` |
| 3.023 | `/workspace/robotics_assets/experiments/e102_top` |
| 2.929 | `/workspace/robotics_assets/experiments/e111_bc` |
| 2.681 | `/workspace/robotics_assets/experiments/e122_full` |
| 2.442 | `/workspace/robotics_assets/experiments/e120_fusion` |
| 2.297 | `/workspace/robotics_assets/experiments/e121_patches` |
| 2.296 | `/workspace/robotics_assets/experiments/e119_pick` |
| 2.046 | `/workspace/robotics_assets/experiments/e118_cloud` |
| 1.939 | `/workspace/robotics_assets/experiments/e114_mask` |
| 1.823 | `/workspace/robotics_assets/models/point_tracking_local` |
| 0.776 | `/workspace/robotics_assets/pretrained/vision` |
| 0.455 | `/workspace/robotics_assets/models/point_tracking` |
| 0.306 | `/workspace/robotics_assets/experiments/e113_voxel` |
| 0.132 | `/workspace/robotics_assets/experiments/e123_mask` |
| 0.125 | `/workspace/robotics_assets/experiments/e112_isee` |
| 0.087 | `/workspace/robotics_assets/pretrained/trackon` |
| 0.074 | `/workspace/robotics_assets/external/MobileSAM` |
| 0.036 | `/workspace/robotics_assets/robot/nexarm` |
| 0.023 | `/workspace/robotics_assets/experiments/e116_dynamic` |

## Largest individual files — one row per distinct inode

| GiB logical | Links | Path |
|---:|---:|---|
| 8.928 | 1 | `/workspace/robotics_assets/datasets/voxel_xyz_two_view/rgb.npy` |
| 1.923 | 1 | `/workspace/robotics_assets/datasets/reach_voxel_grid/ee_delta_25hz_h25_s2/actions.npy` |
| 1.624 | 1 | `/workspace/robotics_assets/experiments/e111_bc/train/latest.pt` |
| 1.529 | 4 | `/workspace/robotics_assets/experiments/e122_full/train/latest.pt` |
| 1.527 | 1 | `/workspace/robotics_assets/experiments/e121_patches/train/latest.pt` |
| 1.526 | 3 | `/workspace/robotics_assets/experiments/e120_fusion/train/latest.pt` |
| 1.523 | 1 | `/workspace/robotics_assets/experiments/e103_center/train_2/latest.pt` |
| 1.523 | 1 | `/workspace/robotics_assets/experiments/e103_center/train/latest.pt` |
| 1.523 | 1 | `/workspace/robotics_assets/experiments/e102_top/train/latest.pt` |
| 1.522 | 3 | `/workspace/robotics_assets/experiments/e119_pick/train/latest.pt` |
| 1.361 | 1 | `/workspace/robotics_assets/experiments/e114_mask/train_3/latest.pt` |
| 1.251 | 1 | `/workspace/arm-manipulation/.venv/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` |
| 1.183 | 2 | `/workspace/robotics_assets/experiments/e118_cloud/train/latest.pt` |
| 1.181 | 1 | `/workspace/robotics_assets/models/reach_joint_delta_ee/final.pt` |
| 1.181 | 1 | `/workspace/robotics_assets/models/reach_joint_delta_ee/latest.pt` |
| 1.172 | 1 | `/workspace/robotics_assets/experiments/e115_direct/train_2/latest.pt` |
| 1.172 | 1 | `/workspace/robotics_assets/experiments/e115_direct/train/latest.pt` |
| 0.908 | 1 | `/workspace/robotics_assets/models/point_tracking_local/optimizer.pt` |
| 0.727 | 1 | `/workspace/arm-manipulation/.venv/lib/python3.11/site-packages/nvidia/cublas/lib/libcublasLt.so.12` |
| 0.722 | 1 | `/workspace/robotics_assets/experiments/e122_full/train/best.pt` |
| 0.722 | 1 | `/workspace/robotics_assets/experiments/e121_patches/train/best.pt` |
| 0.722 | 1 | `/workspace/robotics_assets/experiments/e120_fusion/train/best.pt` |
| 0.721 | 1 | `/workspace/robotics_assets/experiments/e119_pick/train/best.pt` |
| 0.720 | 1 | `/workspace/robotics_assets/experiments/e103_center/train_2/final.pt` |
| 0.720 | 1 | `/workspace/robotics_assets/experiments/e103_center/train_2/best.pt` |
| 0.720 | 1 | `/workspace/robotics_assets/experiments/e103_center/train/final.pt` |
| 0.720 | 1 | `/workspace/robotics_assets/experiments/e103_center/train/best.pt` |
| 0.720 | 1 | `/workspace/robotics_assets/experiments/e102_top/train/final.pt` |
| 0.720 | 1 | `/workspace/robotics_assets/experiments/e102_top/train/best.pt` |
| 0.541 | 1 | `/workspace/robotics_assets/experiments/e111_bc/train/best.pt` |

## Shared checkpoint allocations

One allocation of **1.529 GiB**, 4hard links:

- `/workspace/robotics_assets/experiments/e122_full/train/latest.pt`
- `/workspace/robotics_assets/experiments/e122_full/train/final.pt`
- `/workspace/robotics_assets/experiments/e122_full/test_k10/checkpoint.pt`
- `/workspace/robotics_assets/experiments/e122_full/test_k15/checkpoint.pt`

One allocation of **1.526 GiB**, 3hard links:

- `/workspace/robotics_assets/experiments/e120_fusion/train/latest.pt`
- `/workspace/robotics_assets/experiments/e120_fusion/train/final.pt`
- `/workspace/robotics_assets/experiments/e120_fusion/test_k10/checkpoint.pt`

One allocation of **1.522 GiB**, 3hard links:

- `/workspace/robotics_assets/experiments/e119_pick/train/latest.pt`
- `/workspace/robotics_assets/experiments/e119_pick/train/final.pt`
- `/workspace/robotics_assets/experiments/e119_pick/test_k10/checkpoint.pt`

One allocation of **1.183 GiB**, 2hard links:

- `/workspace/robotics_assets/experiments/e118_cloud/train/latest.pt`
- `/workspace/robotics_assets/experiments/e118_cloud/train/final.pt`

One allocation of **0.392 GiB**, 3hard links:

- `/workspace/robotics_assets/experiments/e118_cloud/train/best.pt`
- `/workspace/robotics_assets/experiments/e118_cloud/test_pick_k10/checkpoint.pt`
- `/workspace/robotics_assets/experiments/e118_cloud/test_size_k15/checkpoint.pt`

## Full inventory

See [large_files.tsv](large_files.tsv) for every >=100MiB file found inside and outside workspace, including allocation sizes and duplicate hard-link names. Dataset arrays such as `rgb.npy`/`actions.npy` are retained data; `latest.pt` checkpoints often contain optimizer state for resuming. Shared-library `.so` files belong to the Python environment. No deletion decisions were made.

## Large files outside workspace — same root filesystem

Filename/size scan only, no file contents read; mounted filesystems and symlinks excluded. These include installed software and local caches, not only research artifacts.

| GiB logical | Links | Path |
|---:|---:|---|
| 21.134 | 1 | `/var/lib/vastai_kaalia/data/core-python.421995.8c05d1e91ca4.1792588622` |
| 1.015 | 1 | `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt_static.a` |
| 0.700 | 1 | `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt.so.12.8.4.1` |
| 0.542 | 1 | `/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9.8.0` |
| 0.438 | 1 | `/usr/lib/x86_64-linux-gnu/libnccl_static.a` |
| 0.393 | 1 | `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusparse_static.a` |
| 0.360 | 1 | `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusparse.so.12.5.8.93` |
| 0.345 | 1 | `/root/.vscode-server/data/CachedExtensionVSIXs/openai.chatgpt-26.51007.21434-linux-x64` |
| 0.281 | 1 | `/root/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome` |
| 0.279 | 1 | `/usr/lib/x86_64-linux-gnu/libnccl.so.2.31.2` |
| 0.276 | 1 | `/opt/nvm/versions/node/v24.20.0/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex` |
| 0.276 | 1 | `/root/.codex/packages/app-server-daemon/releases/0.163.0-x86_64-unknown-linux-musl/bin/codex` |
| 0.274 | 1 | `/root/.vscode-server/extensions/openai.chatgpt-26.51007.21434-linux-x64/bin/linux-x86_64/codex` |
| 0.267 | 1 | `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcufft_static.a` |
| 0.267 | 1 | `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcufft_static_nocallback.a` |
| 0.260 | 1 | `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcufft.so.11.3.3.83` |
| 0.244 | 1 | `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusolver_static.a` |
| 0.238 | 1 | `/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.8.0` |
| 0.225 | 1 | `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusolver.so.11.7.3.90` |
| 0.193 | 1 | `/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell` |

Outside-workspace regular files scanned: 158,677; files >=100MiB: 47; metadata access errors: 0. No files were deleted.

## Largest root-file finding

`/var/lib/vastai_kaalia/data/core-python.421995.8c05d1e91ca4.1792588622` is an ELF Python core dump: **21.134GiB logical**, **3.422GiB allocated**. It is sparse, so deleting it would not reclaim its full logical size. It is a crash-memory artifact, not a model checkpoint. It was identified from its ELF header only; no memory contents were exported. No deletion was performed.
