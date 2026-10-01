# Memory occlusion: joint tracking and action policy

TCOW processes RGB-D history and the visible target mask on the first frame.
Its final frame supplies 300 spatial tokens of width 768. A learned visual
adapter maps them through width 256 to 960. Proprio supplies one state token,
giving 301 dense context tokens. No predicted mask enters the policy.

`flow_matching.py` loads the 16 action layers, gated MLPs, RMSNorms, action/time
projections, state projection and eight context K/V projections from
`lerobot/smolvla_base`, revision `d9f33c94a60fb382c90dea2164c96845bd955e28`.
Pairs of pretrained layers form eight blocks:

```text
action self-attention -> MLP -> cross-attention to all 301 tokens -> MLP
```

The expert width is 720. Action self-attention is bidirectional, matching this
experiment's dense interaction. The context is projected directly from TCOW;
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
  --data memory_occlusion/datasets/memory_occlusion_tcow_25hz_240x320_v1 \
  --weights memory_occlusion/checkpoints/tcow_rgbd_pretrained_joint1200_20260930.pth \
  --config-checkpoint /path/to/tcow_upstream_config_checkpoint.pth \
  --flow-weights memory_occlusion/checkpoints/memory_occlusion_action_expert_init_20261001/expert.safetensors \
  --output memory_occlusion/checkpoints/tcow_dense_action_expert \
  --device mps --batch-size 1 --cluster-size 2 --epochs 1
```

On the server, `train_from_tcow_context_server.sh` supplies the dataset and
checkpoint paths. Select one GPU, a fresh `RUN_NAME`, and an explicit epoch
budget. Stream the command through `tee` in the `huy` tmux session.

## Smoke and evaluate

`smoke_policy.py` verifies pretrained import, dense spatial conditioning,
inference and backward. Supply `--tcow-checkpoint` and `--seeker-config` (JSON
`seeker_args` from that TCOW checkpoint's config) to also check action loss
reaches the TCOW patch embedding and context mask loss never calls FM.
This synthetic test checks connectivity, not task performance.

`rollout.py`, `benchmark.py` and the diagnostics restore the architecture and
normalization from the joint checkpoint. The previous dense decoder remains
available solely to restore checkpoints needed for comparison.
