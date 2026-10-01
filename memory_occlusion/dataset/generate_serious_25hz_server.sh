#!/usr/bin/env bash
set -euo pipefail

cd /home/hoang.pm/duchuy/arm-manipulation
root=/mnt/disk1/backup_user/hoang.pm/huy/arm-manipiulation/datasets/memory_occlusion_tcow_25hz_240x320_v1
plan=/mnt/disk1/backup_user/hoang.pm/huy/arm-manipiulation/scratch/memory_occlusion_tcow_25hz_240x320_v1_plan
python=/mnt/disk1/backup_user/hoang.pm/conda/envs/huy/bin/python
pilot_scenes=8
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if [[ ! -e "$root/source_sha256.txt" ]]; then
  until [[ -s "$plan/plan_summary.json" ]]; do
    if ! tmux list-windows -t huy -F '#W' | grep -qx tcow_plan; then
      echo "expert preflight ended without a complete plan" >&2
      exit 1
    fi
    sleep 30
  done
  "$python" - "$plan/plan_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
if (summary["scene_seeds"], summary["episodes"], summary["expert_preflight"]) != (220, 880, True):
    raise ValueError(f"unexpected preflight summary: {summary}")
PY
  cp "$plan/plan.jsonl" "$plan/plan_summary.json" "$plan/standard_seeds.txt" \
     "$plan/composition_seeds.txt" "$plan/pilot_seeds.txt" "$root/"
fi
test -s "$root/plan.jsonl"
test "$(wc -l < "$root/pilot_seeds.txt")" -eq "$pilot_scenes"
sources=(memory_occlusion/dataset/episode.py memory_occlusion/dataset/generate_dataset.py
         memory_occlusion/dataset/make_tcow_25hz_plan.py
         memory_occlusion/dataset/validate.py memory_occlusion/dataset/audit_tcow_dataset.py
         memory_occlusion/environment/env.py controllers/oracle_pick.py assets/robot/robot.xml)
if [[ -e "$root/source_sha256.txt" ]]; then
  sha256sum -c "$root/source_sha256.txt"
else
  sha256sum "${sources[@]}" "$root/plan.jsonl" > "$root/source_sha256.txt"
fi

"$python" -m memory_occlusion.dataset.generate_dataset \
  --seed-file "$root/pilot_seeds.txt" --workers 4 --tcow-labels \
  --output "$root/standard"
"$python" -m memory_occlusion.dataset.audit_tcow_dataset "$root" --pilot-scenes "$pilot_scenes"

"$python" -m memory_occlusion.dataset.generate_dataset \
  --seed-file "$root/standard_seeds.txt" --workers 4 --tcow-labels \
  --output "$root/standard"
"$python" -m memory_occlusion.dataset.generate_dataset \
  --seed-file "$root/composition_seeds.txt" --workers 4 --tcow-labels \
  --output "$root/composition"
"$python" -m memory_occlusion.dataset.audit_tcow_dataset "$root"
