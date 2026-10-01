#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source_base=/mnt/disk1/backup_user/hoang.pm/huy/arm-manipiulation
output_base=/mnt/disk1/backup_user/25thanh.tk/memory_occlusion
python=/mnt/disk1/backup_user/hoang.pm/conda/envs/huy/bin/python
: "${CUDA_VISIBLE_DEVICES:?select two or three GPUs with enough free memory}"
: "${RUN_NAME:?set a fresh experiment name}"
IFS=, read -ra selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
gpu_count=${#selected_gpus[@]}
(( gpu_count >= 2 && gpu_count <= 3 ))
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
exec "$python" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$gpu_count" --module memory_occlusion.experiments.tcow_joint_flow.train_distributed \
  --data "$source_base/datasets/memory_occlusion_tcow_25hz_240x320_v1" \
  --weights "$source_base/checkpoints/tcow_rgbd_pretrained_joint1200_20260930.pth" \
  --flow-weights "$source_base/checkpoints/memory_occlusion_action_expert_init_20261001/expert.safetensors" \
  --config-checkpoint "$source_base/checkpoints/tcow_upstream_25hz_camera50_20260929/best.pth" \
  --output "$output_base/checkpoints/$RUN_NAME" "$@"
