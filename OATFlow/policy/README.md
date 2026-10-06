# OAT-Flow policy

All policy layers and checkpoint restoration are defined in [model.py](model.py).
The upstream TCOW implementation remains in `OATFlow/third_party/tcow`.

| Component | Class | Current configuration |
| --- | --- | --- |
| Policy | `OATFlowPolicy` | TCOW features + wrist vision + proprioception |
| Wrist vision | `WristVisionEncoder` | Frozen ImageNet ViT-B/16, 400 patches pooled to 64 |
| Context | `ContextDecoder` | Four random-initialized transformer blocks, 365 tokens, width 960 |
| Actions | `FlowMatchingHead` | Pretrained action expert, eight self/cross-attention pairs, H25 |

Frozen TCOW produces 300 current-frame spatial tokens from 30 overview frames
and the first-frame target query mask. Trainable adapters concatenate those
300 tokens with 64 wrist tokens and one proprio token. The context decoder runs
once per prediction, before the action head's ten-step Euler integration.
Current actions are fixed-anchor world-frame XYZ/rotation-vector deltas plus
absolute gripper (seven coordinates); proprioception remains six joint values.
New training is RGB-only. TCOW and wrist stay frozen with the flags below;
context decoder, adapters and action head train together.

Parameter keys and checkpoint metadata retain their existing format. Joint-delta,
overview-only and original dense-head checkpoints can still be restored for
comparison; `LegacyFlowHead` is used only for that original checkpoint format.
Live simulator rollout requires an RGB-only checkpoint.

## Source layout

| File | Responsibility |
| --- | --- |
| `model.py` | Complete policy architecture, normalization and checkpoint restoration |
| `tracking.py` | Import upstream TCOW and compute its original mask loss |
| `data.py` | Episodes, demonstration expansion, history selection and action labels |
| `loader.py` | LeRobot/TorchCodec decoding + native PyTorch DataLoader |
| `train.py` | Single-GPU training and epoch checkpoints |
| `train_distributed.py` | Distributed training |
| `rollout.py` | Closed-loop simulator evaluation |
| `visualization.py` | Tracking-mask video panels |

The single-GPU loader copies compressed train MP4s once into a dedicated
`/dev/shm/oatflow_*` directory. Two spawned workers share those files and decode
only requested frames through LeRobot's eight-entry decoder cache. Native
DataLoader controls batching, collation, pinning and prefetching (factor 2 per
worker, four batches total). History is `round(linspace(0,t,30))`; sampling order,
H25 labels and short batches are preserved. Distributed training uses this same
DataLoader/LeRobot path, with two workers per rank and one compressed video cache
shared across ranks. Global batches are split into explicit rank batch sizes;
an empty rank gets a zero-weight dummy at a short tail. No manual episode
prefetch queue, decoder thread pool or shared decoded clusters remain.

## Train on 3090

Run from the repository root in the `huy` tmux session. Use a fresh run name.
The current server environment uses PyTorch 2.7.0+cu128, LeRobot 0.6.1 and
TorchCodec 0.3.0. Source weights must already exist at the paths below.

```bash
cd /workspace/arm-manipulation
source /workspace/memory_occlusion/runtime/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run_name=e14
set -o pipefail
taskset -c 0-7 python -u -m OATFlow.policy.train \
  --data /workspace/datasets/e09_ee \
  --weights /workspace/checkpoints/e09_init/tcow.pt \
  --flow-weights /workspace/checkpoints/memory_occlusion_action_expert_init/expert.safetensors \
  --config-checkpoint /workspace/checkpoints/e05_ctx64_3/config.pt \
  --wrist-weights /workspace/checkpoints/e09_init/vit_b_16-c867db91.pth \
  --flow-only --wrist-camera --freeze-wrist-encoder --contextualize \
  --epochs 1 --batch-size 16 --cluster-size 2 --flow-lr 1e-4 --seed 0 \
  --output "/workspace/checkpoints/${run_name}" \
  2>&1 | tee "/workspace/memory_occlusion/logs/${run_name}.log"
```

Both trainers skip validation and automatic rollout. They save `latest.pt` at
epoch end and link `final.pt` when complete; there is no `best.pt` selection.
Distributed training uses `python -m torch.distributed.run --standalone
--nproc_per_node=<count> --module OATFlow.policy.train_distributed` with explicit
GPU selection and the same required weight/data flags. Its rank batch sizes are
set with `--rank-batch-sizes`; keep total CPU affinity within eight cores.

For end-to-end joint-delta training, omit `--flow-only` and
`--freeze-wrist-encoder`. `--contextualize` also supports trainable backbones.
Use `--base-demonstrations-only` to exclude attached perturbation demos.
Separate peak learning rates and a schedule are selected with
`--tcow-lr 1e-6 --wrist-lr 1e-6 --flow-lr 1e-5
--lr-schedule cosine --warmup-fraction .05 --min-lr-ratio .1`.
Warmup and cosine share the exact optimizer-update count across all epochs;
the last update uses 10% of each peak LR. Default runs retain constant LR.

```bash
cd /mnt/disk1/backup_user/25thanh.tk/arm-manipulation
base=/mnt/disk1/backup_user/hoang.pm/huy/arm-manipiulation
runs=/mnt/disk1/backup_user/25thanh.tk/memory_occlusion
export CUDA_VISIBLE_DEVICES=0,1,6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -o pipefail
taskset -c 56-63 /mnt/disk1/backup_user/hoang.pm/conda/envs/huy/bin/python \
  -u -m torch.distributed.run --standalone --nproc_per_node=3 \
  --module OATFlow.policy.train_distributed \
  --data "$runs/datasets/memory_occlusion_multiview_25hz_delta_v1" \
  --base-demonstrations-only \
  --weights "$base/checkpoints/tcow_published_kubric/checkpoint.pth" \
  --config-checkpoint "$base/checkpoints/tcow_upstream_25hz_camera50_20260929/best.pth" \
  --flow-weights "$base/checkpoints/memory_occlusion_action_expert_init_20261001/expert.safetensors" \
  --wrist-weights "$runs/checkpoints/memory_occlusion_wrist_vit_base_imagenet1k_v1/vit_b_16-c867db91.pth" \
  --wrist-camera --contextualize --rank-batch-sizes 10 10 10 \
  --epochs 5 --cluster-size 2 --tcow-lr 1e-6 --wrist-lr 1e-6 --flow-lr 1e-5 \
  --lr-schedule cosine --warmup-fraction .05 --min-lr-ratio .1 --seed 0 \
  --output "$runs/checkpoints/e17_e2e" \
  2>&1 | tee "$runs/logs/e17_e2e.log"
```

Run in `huy:e17_e2e` after checking GPU availability and fresh output/cache
directories. No validation or separate smoke run; progress reports ETA and LR.

## Closed loop

Run separately after a checkpoint is complete. The frame budget below includes
the reveal/shuffle context; increase it if the episode's manipulation needs more.
`--execute-chunk` supports K1–K25; H remains 25, Euler steps 10, noise seed 0.

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -u -m OATFlow.policy.rollout \
  /workspace/datasets/e09_ee/standard/episode_000217_Milk \
  --checkpoint /workspace/checkpoints/e13_dl/final.pt \
  --config-checkpoint /workspace/checkpoints/e05_ctx64_3/config.pt \
  --execute-chunk 10 --max-total-frames 2500 \
  --output /workspace/memory_occlusion/videos/e13_dl/test_k10 \
  2>&1 | tee /workspace/memory_occlusion/logs/e13_k10.log
```

This is a command example, not a completed evaluation. Keep a configuration
README beside each actual run/test output.

## Cleanup and verification

One-time geometry probes, rollout diagnostics, checkpoint extraction, depth
cleanup, obsolete joint-only evaluation and server launch wrappers were removed.
The unused early decision reader and metric helper module were also retired.
Temporary data tests were run and removed after passing. Collection and
augmentation remain part of data generation (`--extra-demos 2/3`). One-time EE
conversion/replay scripts were retired after use; `--action-mode ee` now generates
EE labels directly, including extras and train-only normalization. EE kinematics remain necessary
for executing predicted EE actions in closed loop.

The consolidation was checked on CPU: 12 data/EE/augmentation tests passed;
seeded policy parameter hashes, H25 predictions, contextualized tokens and Euler
outputs match before/after; strict checkpoint restoration preserves frozen flags.
The architecture comparison uses a synthetic tracker fixture, not a new simulator
success evaluation. Native two-worker loading and CLI imports are also checked.
The running `e13_dl` server copy uses the pre-consolidation source and is left
running; this cleanup changes the local source only.

Historical commands and source manifests under `outputs/` and `docs/research/`
retain their original paths as experiment provenance. Some referenced diagnostic
scripts have been retired; use the current entry points documented here.
