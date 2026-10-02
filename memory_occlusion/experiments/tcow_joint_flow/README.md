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

Dataset generation now saves synchronized, uncropped `wrist_rgb.mp4` at
25 Hz plus camera metadata. Old overview-only data cannot train a wrist
policy; backfill wrist views or collect a fresh dataset before training. Both trainers and
validation require exact wrist/overview frame alignment and reject missing
wrist data before creating a run. Relative-joint conversion links this extra video as well.
Use `--no-wrist-camera` only to reproduce the previous overview-only experiment.

### Backfill existing episodes

`backfill_wrist` restores each frame from recorded observed joint positions
and object/cover poses, then runs only forward kinematics and rendering. It
preserves the split, absolute/relative action chunks, masks and normalization.
Original large files are linked read-only; only wrist videos and metadata are
new. It does not rerun the expert, dynamics, IK, depth or mask generation.

Old records omitted the left jaw/pinion, so their poses are reconstructed from
the geometric joint constraints. Clipped/float32 proprio and old post-step
render states may also introduce small differences. Each episode is gated on
reconstructed overview RGB and visible-mask agreement at reveal, decisions,
approach, closure, grasp, release and final frames. These checks bound visible
alignment; they do not establish bit-identical wrist images. New collection
saves full `qpos` and renders after `mj_forward` to avoid this approximation.

```bash
.venv/bin/python -m memory_occlusion.dataset.backfill_wrist \
  --source /path/to/memory_occlusion_relative_joint_25hz_h25_k10_v1 \
  --output /path/to/memory_occlusion_multiview_relative_joint_25hz_v1 \
  --workers 4
```

For the gate, use `--max-episodes 1` with a separate smoke output. Add `--resume`
only when resuming the same source/output configuration after interruption.
Per-frame/per-episode tqdm progress and reconstruction metrics appear on stdout.
Run long server commands through `tee` in `huy` tmux. `wrist_backfill.json` per
episode and `backfill_summary.json` record checks and timings. Existing raw
data remain unchanged. A missing saved pose or failed gate stops the job.
Checkpoints store `wrist_camera` and `context_tokens`; restoration constructs
the recorded architecture and loads strictly. Existing comparison
checkpoints still restore with 301 tokens.

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

Run the padding/phase and distributed-aggregation regression checks locally:

```bash
.venv/bin/python -m unittest memory_occlusion.experiments.tcow_joint_flow.test_validation -v
.venv/bin/python -m memory_occlusion.experiments.tcow_joint_flow.test_wrist -v
```

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

`smoke_policy.py` verifies pretrained import, dense spatial conditioning,
inference and backward. Supply `--tcow-checkpoint` and `--seeker-config` (JSON
`seeker_args` from that TCOW checkpoint's config) to also check action loss
reaches the TCOW patch embedding and context mask loss never calls FM.
This synthetic test checks connectivity, not task performance.

`rollout.py`, `benchmark.py` and the diagnostics restore the architecture and
normalization from the joint checkpoint. The previous dense decoder remains
available solely to restore checkpoints needed for comparison.

## Relative-joint experiment

`memory_occlusion.dataset.prepare_relative_actions` accepts `--source` and a
fresh `--output` dataset directory. It writes each stride-ten H25 chunk as
`expert_command[t:t+25,:5] - observed_proprio[t,:5]`, keeps gripper absolute,
and fits action mean/std on valid TRAIN chunk entries only. State statistics
use valid TRAIN frames. Each episode has its own `relative_actions.npz`; RGB-D,
absolute source labels and metadata are linked read-only to the source. The
new dataset depends on that retained source. Train/val/test manifests are
preserved. `normalization.json` and `dataset.json` record the new convention.

`memory_occlusion.dataset.replay_relative_actions --data DATA --output OUTPUT`
decodes standardized labels and executes ten actions per chunk, fixing the
live observed anchor for the entire chunk. It selects ten distinct train
scenes spanning targets and one/two/three swaps, writes 25 Hz videos and
traces, and exits with an error unless all ten complete cover removal,
target grasp and placement. This gate uses the full expert action duration.

Pass the new dataset to either trainer with `--flow-only --epochs 1`. A joint
checkpoint may be supplied as `--weights` in this frozen mode: only TCOW and
the visual adapter initialize from it; the action expert initializes from
`--flow-weights`. TCOW including its mask head has no trainable parameters or
mask loss. Frozen training skips loading dense mask labels. Relative targets
come from the stored chunks; flow inference returns absolute commands by
adding back the observed chunk anchor after undoing mean/std. Checkpoint
`action_representation` restores this decoder, including in closed loop.
Validation converts labels back to absolute joint units for comparison.
