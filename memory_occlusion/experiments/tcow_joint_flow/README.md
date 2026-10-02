# Memory occlusion: joint tracking and action policy

TCOW processes RGB-D history and the visible target mask on the first frame.
Its final frame supplies 300 spatial tokens of width 768. A learned visual
adapter maps them through width 256 to 960. Proprio supplies one state token.
The current policy also adds wrist tokens below; previous overview-only
checkpoints used 301 dense context tokens. No predicted mask enters the policy.

New training enables `--wrist-camera` by default. FM also reads the **current**, full 320x320 wrist RGB
image. A separate ViT encoder copies TCOW's RGB patch convolution, spatial
attention/MLP weights and positional embeddings at initialization. It keeps
all 400 spatial tokens (20x20), projects 768 directly to 960, and adds a
learned wrist view embedding. Its weights are independent of TCOW and train
with action loss, including when TCOW and its mask head are frozen.

```text
overview RGB-D history + first-frame query -> original TCOW -> 300 x 768 -> 300 x 960
current wrist RGB                         -> spatial ViT   -> 400 x 768 -> 400 x 960
current proprio                           -> state MLP                  ->   1 x 960
                                                                          |
                                                    701 dense context tokens -> FM -> 25 x 6
```

Each of the eight action blocks reads all 701 tokens. The wrist encoder runs
once per predicted chunk, outside the Euler integration loop. Context-only
mask minibatches forward TCOW alone; they skip the wrist encoder and FM.
Closed loop supplies the latest live wrist image at each K10 replan. Videos
append the wrist view beside RGB and the three predicted masks.

Dataset generation saves synchronized overview RGB-D, wrist RGB and three TCOW
mask channels at 25 Hz. Collect a fresh dataset with the command in
[memory_occlusion/README.md](../../README.md). Missing or misaligned wrist views
are rejected by training and validation. Use `--no-wrist-camera` only to restore
previous overview-only experiments. Checkpoints record their architecture.

`flow_matching.py` loads the 16 action layers, gated MLPs, RMSNorms, action/time
projections, state projection and eight context K/V projections from
`lerobot/smolvla_base`, revision `d9f33c94a60fb382c90dea2164c96845bd955e28`.
Pairs of pretrained layers form eight blocks:

```text
action self-attention -> MLP -> cross-attention to all context tokens -> MLP
```

The expert width is 720. Action self-attention is bidirectional, matching this
experiment's dense interaction. The context is projected directly from TCOW/wrist/proprio;
SmolVLM's evolving prefix hidden states are replaced. These changes require
fine-tuning even though all action expert weights are imported.

The policy predicts 25 actions with six joints, padding noisy actions to the
checkpoint's 32 dimensions internally. Beta time sampling, sinusoidal time
embedding and ten-step Euler inference follow SmolVLA. The public time
convention is noise at zero; both the pretrained time and velocity sign are
reversed together. Mean/std statistics use only valid action frames in the
train split, and are stored in the joint checkpoint. External actions remain
normalized absolute joint targets, with arm joints in [-1,1] and gripper in
[0,1].

At 25 Hz, sample history endpoints every ten frames. Context minibatches
forward TCOW alone and train masks; action minibatches train action and mask
loss together. Closed loop executes ten of the 25 predicted actions and
recomputes TCOW context from the first query frame through the current frame.

## Prepare weights

TCOW's unchanged vendor checkout must exist at `memory_occlusion/third_party/tcow`,
including its TimeSformer dependency. It is excluded from this repository.
The tested TCOW revision is `a72e3e13a45e4156137328e5290f9e848d360367`.

```bash
.venv/bin/python -m memory_occlusion.experiments.tcow_joint_flow.download_action_expert \
  --output memory_occlusion/checkpoints/memory_occlusion_action_expert_init_20261001
```

This validates the pinned Hub checkpoint checksum and exports only the needed
weights to `expert.safetensors`, with `import_report.json`. It retains no unused
VLM checkpoint. A failed network transfer can be resumed with `--resume-download`.

## Train

```bash
.venv/bin/python -m memory_occlusion.experiments.tcow_joint_flow.train_from_tcow_context \
  --data memory_occlusion/datasets/memory_occlusion_multiview_25hz_v1 \
  --weights memory_occlusion/checkpoints/tcow_rgbd_pretrained_joint1200_20260930.pth \
  --config-checkpoint /path/to/tcow_upstream_config_checkpoint.pth \
  --flow-weights memory_occlusion/checkpoints/memory_occlusion_action_expert_init_20261001/expert.safetensors \
  --output memory_occlusion/checkpoints/tcow_dense_action_expert \
  --wrist-camera --device mps --batch-size 1 --cluster-size 2 --epochs 1
```

On the server, `train_from_tcow_context_server.sh` supplies the dataset and
checkpoint paths. Select one GPU, a fresh `RUN_NAME`, and an explicit epoch
budget. Stream the command through `tee` in the `huy` tmux session.

## Smoke and evaluate

For the additional server-2 run, `train_distributed_server.sh` uses `torchrun`
on two or three selected GPUs. `--rank-batch-sizes` specifies each GPU's batch;
their sum is the global batch. It preserves the full sample plan and uses
zero-weight dummy samples only when a final partial batch leaves a rank empty.
Action loss is weighted by the global count of valid action steps. Each rank
uses the original mask loss on its local examples, weighted by its sample count.
Rank zero writes checkpoints. Validation episodes are divided between ranks;
error sums and valid counts are reduced across all ranks.

Validation uses every valid action start at stride ten, including short final
chunks with padding excluded. `action_mae` covers all valid entries of the
25-action chunks; `action_mae_first`, `action_mae_first10`, per-offset and
per-joint errors expose where the policy fails. Cover/object close phases
report gripper MAE and binary error at threshold 0.5. Counts include repeated
frames in overlapping predicted chunks. Metrics remain in joint-limit units
after reversing the flow's mean/std normalization. This protocol supersedes
the old three-chunk-per-episode validation, so their aggregate MAEs are not
directly comparable.

Add `--closed-loop-every-evals 4` to either training command to also run one
fixed validation episode per target every fourth evaluation. The default zero
disables simulation. Rollouts use K10 and the existing 1,000-frame total budget
(including context), save mask videos and summaries under
`validation/step_<step>/`, and report correct-cover selection/removal, target
grasp and task-success rates. Cover removal requires the correct cover in its
drop zone. This is success within that frame budget, which may be shorter
than an expert demonstration. Test scenes are excluded. `best.pt` continues to
use the expanded offline `action_mae`; rollout scores are reported separately.

```bash
CUDA_VISIBLE_DEVICES=2,5 RUN_NAME=memory_occlusion_ddp_smoke_20261001 \
  bash memory_occlusion/experiments/tcow_joint_flow/train_distributed_server.sh \
  --rank-batch-sizes 8 4 --max-clusters 1 --smoke-steps 4
```

After that smoke passes, omit `--max-clusters` and `--smoke-steps`, select a new
`RUN_NAME`, and set `--epochs 5`. Run through `tee` inside `huy:memory_occlusion`.
Rank zero decodes each eight-episode cluster once into temporary shared RAM
buffers under `/dev/shm/<RUN_NAME>` and prefetches the next cluster. All ranks
map those buffers; completed clusters are released after synchronization.
Shared buffers are capped at 48 GiB, leaving RAM for the model and decoder.
CPU affinity must cap the entire job to eight cores. Do not start if total RAM
or any selected GPU's memory exceeds the experiment limits.

`rollout.py`, `benchmark.py` and the diagnostics restore the architecture and
normalization from the joint checkpoint. The previous dense decoder remains
available solely to restore checkpoints needed for comparison.

## Relative-joint experiment

The data generator accepts `--action-mode absolute|delta`. Delta mode adds an
in-place postprocessing stage after collection and audit: each stride-ten H25
chunk is `expert_command[t:t+25,:5] - observed_proprio[t,:5]`, with absolute
gripper. It retains the original absolute labels and writes `relative_actions.npz`
inside each episode. No linked source dataset is needed. Normalization uses only
valid standard TRAIN samples; `normalization.json` and `dataset.json` record the
convention. Absolute mode retains absolute labels and fits their TRAIN statistics.

Pass the new dataset to either trainer with `--flow-only --epochs 1`. A joint
checkpoint may be supplied as `--weights` in this frozen mode: only TCOW and
the visual adapter initialize from it; the action expert initializes from
`--flow-weights`. TCOW including its mask head has no trainable parameters or
mask loss. Frozen training skips loading dense mask labels. Relative targets
come from the stored chunks; flow inference returns absolute commands by
adding back the observed chunk anchor after undoing mean/std. Checkpoint
`action_representation` restores this decoder, including in closed loop.
Validation converts labels back to absolute joint units for comparison.
