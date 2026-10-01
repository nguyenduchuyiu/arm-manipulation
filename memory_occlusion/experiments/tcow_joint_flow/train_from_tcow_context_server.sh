#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
base=/mnt/disk1/backup_user/hoang.pm/huy/arm-manipiulation
python=/mnt/disk1/backup_user/hoang.pm/conda/envs/huy/bin/python
: "${CUDA_VISIBLE_DEVICES:?select one GPU}"
: "${RUN_NAME:?set a fresh experiment name}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec "$python" -m memory_occlusion.experiments.tcow_joint_flow.train_from_tcow_context \
  --data "$base/datasets/memory_occlusion_tcow_25hz_240x320_v1" \
  --weights "$base/checkpoints/tcow_rgbd_pretrained_joint1200_20260930.pth" \
  --smolvla-weights "$base/checkpoints/smolvla_base_dense_expert_20261001/expert.safetensors" \
  --config-checkpoint "$base/checkpoints/tcow_upstream_25hz_camera50_20260929/best.pth" \
  --output "$base/checkpoints/$RUN_NAME" "$@"
