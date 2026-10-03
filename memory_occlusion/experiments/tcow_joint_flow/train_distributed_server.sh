#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source_base=/mnt/disk1/backup_user/hoang.pm/huy/arm-manipiulation
output_base=/mnt/disk1/backup_user/25thanh.tk/memory_occlusion
python=/mnt/disk1/backup_user/hoang.pm/conda/envs/huy/bin/python
: "${CUDA_VISIBLE_DEVICES:?select two or three GPUs with enough free memory}"
: "${RUN_NAME:?set a fresh experiment name}"
[[ "$RUN_NAME" =~ ^memory_occlusion_[a-zA-Z0-9_.-]+$ ]]
shared_cache="/dev/shm/$RUN_NAME"
[[ ! -e "$shared_cache" ]]
trainer_pid=
cleanup() {
  if [[ -n "$trainer_pid" ]] && kill -0 "$trainer_pid" 2>/dev/null; then
    kill -TERM "$trainer_pid"
    wait "$trainer_pid" || true
  fi
  rm -rf -- "$shared_cache"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
IFS=, read -ra selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
gpu_count=${#selected_gpus[@]}
(( gpu_count >= 2 && gpu_count <= 3 ))
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
"$python" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$gpu_count" --module memory_occlusion.experiments.tcow_joint_flow.train_distributed \
  --data "$output_base/datasets/memory_occlusion_multiview_25hz_delta_v1" \
  --weights "$source_base/checkpoints/tcow_published_kubric/checkpoint.pth" \
  --flow-weights "$source_base/checkpoints/memory_occlusion_action_expert_init_20261001/expert.safetensors" \
  --config-checkpoint "$source_base/checkpoints/tcow_upstream_25hz_camera50_20260929/best.pth" \
  --output "$output_base/checkpoints/$RUN_NAME" "$@" &
trainer_pid=$!
wait "$trainer_pid"
