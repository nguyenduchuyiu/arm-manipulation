# TCOW + flow matching

TCOW processes 30 past/current RGB frames and the first-frame query mask. Its current 300 spatial tokens, 64 wrist tokens and one proprio token pass through four context decoder blocks, then condition the H25 FM action expert. Actions, noise and velocity outputs have six coordinates throughout; there is no action padding. SmolVLA initialization selects six input columns/output rows from its32-coordinate export; a prior checkpoint must already use the six-coordinate head. The wrist encoder uses the prior's trained vision state and can be frozen; TCOW is trainable by default.

```bash
python -m OATFlow.policy.train \
  --data "$CONTEXT_DATA" --weights "$TCOW_INIT" --config-checkpoint "$TCOW_CONFIG" \
  --prior-checkpoint "$PRIOR_CHECKPOINTS/final.pt" --freeze-wrist-encoder \
  --epochs 10 --batch-size 256 --micro-batch-size 8 \
  --tcow-lr 1e-4 --flow-lr 1e-4 --min-lr 3e-6 --warmup-steps 215 \
  --output "$JOINT_CHECKPOINTS"

python -m OATFlow.policy.rollout "$CONTEXT_EPISODE" \
  --checkpoint "$JOINT_CHECKPOINTS/final.pt" --config-checkpoint "$TCOW_CONFIG" \
  --execute-chunk 10 --max-total-frames 3000 --output "$ROLLOUT_OUTPUT"
```

`--prior-checkpoint` transfers the trained FM expert, proprio encoder, state/KV projections, four context decoder blocks, both visual adapters and view embeddings. It also maps the prior's torchvision wrist ViT into the main encoder. **Keep the prior's action/state normalization** so its learned output calibration is preserved. The overview backbone is pretrained TCOW, and oracle task/target embeddings are excluded. Prior/main context token counts differ (131/365); decoder parameters have the same shapes. Transfer checks key sets and loads each module strictly.

Alternatively, `--flow-weights` initializes a fresh main FM from the prepared SmolVLA action expert export; supply `--wrist-weights` for ImageNet wrist initialization. These two FM initialization options are mutually exclusive. `--flow-only` explicitly freezes TCOW; context-decoder use does not force that flag. The default joint loss is FM velocity loss plus `0.2 * TCOW mask loss`. Context-only samples contribute mask loss; expert samples contribute both. Context observations and valid expert chunk starts both use stride 1. Action loss remains differentiable through the TCOW feature hook.

The TCOW initializer contains `net_seeker` weights (or `model` weights prefixed with `tcow.`); the configuration checkpoint supplies `seeker_args` and `train_args`. The supported geometry is 30 frames, 240x320, ViT-B/16, three output masks. Convert upstream RGB-D+query patch weights once to RGB+query by retaining RGB and query kernels. Runtime uses RGB only.

For two or three GPUs, use `torchrun --nproc_per_node=2 -m OATFlow.policy.train_distributed` with the same initialization/data arguments and `--rank-batch-sizes 1 1`. Each rank handles part of the global batch; short batches use zero-weight dummy samples. The distributed trainer supports `--resume` for model weights with a fresh optimizer, as stated in its CLI help. No standalone profiling/probe modes or legacy dense-head restoration remain.

Training saves `config.json`, `flow_initialization.json`, per-epoch `latest.pt`, final checkpoint and history. No validation-based checkpoint selection is implemented; use held-out closed-loop rollouts to assess behavior. Run long training inside tmux, with stdout/stderr streamed through `tee`.

`MemoryOcclusionEnv.step` advances one40ms policy interval using50Hz linear command interpolation through the shared `environment/control.py` helper. The command sequence is previous waypoint@0ms, midpoint@20ms, new waypoint@40ms; elapsed-step/success accounting remains25Hz. Expert trajectory recording and model training data stay25Hz.

Single-GPU training streams `--micro-batch-size` examples through the unchanged logical `--batch-size` cluster plan. Accumulate gradients, clip once and step the optimizer/scheduler once per logical batch. FM weighting uses the total valid action coordinates across that logical batch, including short terminal chunks; mask losses are evaluated per microbatch and weighted by its example count. Warmup/cosine applies to optimizer updates; TCOW follows the same relative decay as FM. Default TCOW LR remains2e-5 unless explicitly set. Eight compact mask episodes are cached per worker to avoid decompressing full trajectories for every microbatch.

For TCOW mask finetuning followed by frozen encoders:

```bash
python -m OATFlow.policy.train_tcow \
  --data "$CONTEXT_DATA" --weights "$TCOW_INIT" --config-checkpoint "$TCOW_CONFIG" \
  --output "$TCOW_FINETUNE" --epochs 2 \
  --batch-size 16 --micro-batch-size 8 --cluster-size 4 \
  --lr 1e-4 --min-lr 3e-6 --warmup-steps 215

python -m OATFlow.policy.train \
  --data "$CONTEXT_DATA" --weights "$TCOW_FINETUNE/final.pt" --config-checkpoint "$TCOW_CONFIG" \
  --flow-weights "$FLOW_EXPERT" --wrist-weights "$VIT_WEIGHTS" \
  --flow-only --freeze-wrist-encoder --epochs 10 \
  --batch-size 256 --micro-batch-size 16 --cluster-size 4 \
  --flow-lr 1e-4 --min-lr 3e-6 --warmup-steps 215 --output "$FM_CHECKPOINTS"
```

The first stage uses every frame of complete context+expert episodes, including the final frame, and trains only the original three TCOW mask channels. Its checkpoint exposes `net_seeker` for strict `load_training_tracker` loading. The second stage uses only expert action chunks, keeps TCOW and wrist ViT frozen/eval, and optimizes the context decoder, visual/proprio/state/KV adapters and six-coordinate action expert. Every valid expert frame starts an FM chunk. Frozen TCOW still reads the query-to-current history; its large mask outputs are discarded before FM backward. No oracle task/target IDs enter the main model. Execute the second command only after the first succeeds and saves its final checkpoint.

`train_tcow` evaluates all standard/val scenes at initialization, every100optimizer updates (`--eval-every`) and epoch ends. Save `latest.pt` with optimizer/scheduler/RNG, retain best validation weights in `best.pt`, and select the final tracking weights for frozen FM. Validation scores the current frame on visible, each shuffle midpoint, final covered, exposed, grasp and lift clips; the query frame itself is never scored. Foreground IoU excludes empty GT masks. `--early-stop-iou 0.85` stops immediately when all3channel foreground IoUs, occluded-target IoU and both post-shuffle cover IoUs reach the threshold, with absent-mask FP area at most0.5%. Otherwise the requested epochs remain the cap. Detailed per-clip metrics are in `validation/step_NNNNNN.json`. `SIGUSR1` requests an extra save/evaluation at the next completed optimizer update without restarting training. Final metadata distinguishes normal completion and validated early stopping; a stage runner must accept either status rather than requiring an epoch2selected checkpoint.

To inspect any tracker checkpoint independently, use `python -m OATFlow.policy.evaluate_tcow --data "$CONTEXT_DATA" --checkpoint "$TRACKER_CHECKPOINT" --config-checkpoint "$TCOW_CONFIG" --output "$IOU_OUTPUT"`. This evaluates tracking masks on held-out scenes; it does not establish physical grasp success.

For a small TCOW-only finetune, pass `--subset-per-task 10`:120unique standard/train scenes,10per task and5per target identity. Selection is deterministic at the run seed, with no validation/test scenes included. Save the exact scene list in `train_subset.json`; validation still uses all standard/val episodes. This option does not alter the subsequent FM dataset: the frozen-encoder FM trainer continues to use all standard/train expert chunks.

When both encoders are frozen, encode each selected expert start once and train from pre-adapter features:

```bash
python -m OATFlow.policy.features --data "$CONTEXT_DATA" \
  --weights "$TRACKER_CHECKPOINT" --config-checkpoint "$TCOW_CONFIG" \
  --wrist-weights "$VIT_WEIGHTS" --batch-size 16 --ram-gib 20 --output "$FEATURE_CACHE"

python -m OATFlow.policy.train --data "$CONTEXT_DATA" \
  --weights "$TRACKER_CHECKPOINT" --config-checkpoint "$TCOW_CONFIG" \
  --flow-weights "$FLOW_EXPERT" --wrist-weights "$VIT_WEIGHTS" \
  --flow-only --freeze-wrist-encoder --feature-cache "$FEATURE_CACHE" \
  --epochs 10 --batch-size 128 --micro-batch-size 128 --cluster-size 4 \
  --flow-lr 1e-4 --min-lr 3e-6 --warmup-steps 215 --output "$FM_CHECKPOINTS"
```

The cache preserves the same stride-1 action starts, first-frame target query and30past/current frames. Each episode shard contains300TCOW and64pooled wrist tokens in BF16, plus FP32 proprio/actions and exact terminal valid-step masks. Only frozen encoder features are cached; visual adapters, context decoder, proprio/state/KV projections and FM remain trainable. Source checkpoint/config/wrist/manifest hashes are verified before training. `index.json` records the disk and RAM shard paths; RAM shards must remain available until training ends. Cached training keeps both encoders on CPU and saves their original weights in the full policy checkpoint for live inference. Test physical batch VRAM with optimizer states before selecting `--batch-size`; using the same microbatch size removes gradient accumulation.
