#!/usr/bin/env bash
# Run inside tmux after activating the Python environment.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
data="${1:?usage: run.sh DATA OUTPUT VISION_WEIGHTS FLOW_WEIGHTS [expert|context] [PLAN]}"
output="${2:?checkpoint output required}"
vision_weights="${3:?ImageNet ViT weights required}"
flow_weights="${4:?SmolVLA action expert export required}"
mode="${5:-expert}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
log_dir="$(dirname -- "$output")/logs"
mkdir -p "$log_dir"
name="$(basename -- "$output")"
if [[ -e "$data" || -e "$output" ]]; then
    echo 'Use fresh dataset and checkpoint directories.' >&2
    exit 1
fi
case "$mode" in
    expert)
        python -u -m OATFlow.dataset.generate_dataset --mode expert --output "$data" \
            --demos-per-task 50 --workers 4 --start-seed 16000 \
            2>&1 | tee "$log_dir/${name}_collect.log"
        python -u -m OATFlow.prior.audit "$data" --demos-per-task 50 \
            2>&1 | tee "$log_dir/${name}_audit.log"
        ;;
    context)
        plan="${6:?context mode requires a scene plan}"
        python -u -m OATFlow.dataset.generate_dataset --mode context --plan "$plan" \
            --output "$data" --workers 4 2>&1 | tee "$log_dir/${name}_collect.log"
        ;;
    *) echo 'Mode must be expert or context.' >&2; exit 1 ;;
esac
python -u -m OATFlow.prior.train --data "$data" --output "$output" \
    --vision-weights "$vision_weights" --flow-weights "$flow_weights" \
    --epochs 10 --batch-size 256 \
    --lr 1e-4 --min-lr 3e-6 --warmup-steps 500 --seed 0 \
    2>&1 | tee "$log_dir/${name}_train.log"
