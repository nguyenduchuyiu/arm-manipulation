#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
root=memory_occlusion/datasets/memory_occlusion_tcow_25hz_240x320_v1
python=.venv/bin/python
workers="$("$python" -c 'import os; print(min(6, os.cpu_count()))')"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "Preparing full 25 Hz dataset: 800 standard + 80 composition episodes"
test -s "$root/plan.jsonl" || { echo "Missing frozen plan: $root/plan.jsonl" >&2; exit 1; }
test "$(wc -l < "$root/standard_seeds.txt")" -eq 200 || { echo "Expected 200 standard seeds" >&2; exit 1; }
test "$(wc -l < "$root/composition_seeds.txt")" -eq 20 || { echo "Expected 20 composition seeds" >&2; exit 1; }
shasum -a 256 memory_occlusion/dataset/episode.py \
  memory_occlusion/dataset/generate_dataset.py \
  memory_occlusion/dataset/make_tcow_25hz_plan.py \
  memory_occlusion/dataset/validate.py \
  memory_occlusion/dataset/audit_tcow_dataset.py \
  memory_occlusion/environment/env.py controllers/oracle_pick.py \
  assets/robot/robot.xml "$root/plan.jsonl" > "$root/source_sha256.txt"

echo "Generating standard episodes with $workers workers (resume completed episodes)"
"$python" -m memory_occlusion.dataset.generate_dataset \
  --seed-file "$root/standard_seeds.txt" --workers "$workers" --tcow-labels \
  --output "$root/standard"
echo "Generating composition episodes with $workers workers (resume completed episodes)"
"$python" -m memory_occlusion.dataset.generate_dataset \
  --seed-file "$root/composition_seeds.txt" --workers "$workers" --tcow-labels \
  --output "$root/composition"
echo "Auditing all 880 episodes"
"$python" -m memory_occlusion.dataset.audit_tcow_dataset "$root"
echo "Dataset generation and audit complete: $root"
