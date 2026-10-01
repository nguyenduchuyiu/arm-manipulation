#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
root=memory_occlusion/datasets/memory_occlusion_tcow_25hz_240x320_v1
python=.venv/bin/python

test -s "$root/plan.jsonl"
test "$(wc -l < "$root/pilot_seeds.txt")" -eq 8
test "$(wc -l < "$root/standard_seeds.txt")" -eq 200
test "$(wc -l < "$root/composition_seeds.txt")" -eq 20
shasum -a 256 memory_occlusion/dataset/episode.py \
  memory_occlusion/dataset/generate_dataset.py \
  memory_occlusion/dataset/make_tcow_25hz_plan.py \
  memory_occlusion/dataset/validate.py \
  memory_occlusion/dataset/audit_tcow_dataset.py \
  memory_occlusion/environment/env.py controllers/oracle_pick.py \
  assets/robot/robot.xml "$root/plan.jsonl" > "$root/source_sha256.txt"

"$python" -m memory_occlusion.dataset.generate_dataset \
  --seed-file "$root/pilot_seeds.txt" --workers 4 --tcow-labels \
  --output "$root/standard"
"$python" -m memory_occlusion.dataset.audit_tcow_dataset "$root" --pilot-scenes 8

"$python" -m memory_occlusion.dataset.generate_dataset \
  --seed-file "$root/standard_seeds.txt" --workers 4 --tcow-labels \
  --output "$root/standard"
"$python" -m memory_occlusion.dataset.generate_dataset \
  --seed-file "$root/composition_seeds.txt" --workers 4 --tcow-labels \
  --output "$root/composition"
"$python" -m memory_occlusion.dataset.audit_tcow_dataset "$root"
