# Vision manipulation prior

A frozen shared ViT-B/16 encodes the current overview and wrist images into 64 tokens per view. Task, target and proprio tokens pass through four context decoder blocks. Actions are six absolute joint setpoints: five arm joints and gripper. TCOW is introduced separately in the main policy.

Every valid expert frame starts a chunk (stride 1). Labels and observed proprio anchors are read directly from `supervision.npz` and `observation.npz`. Context and terminal observations are excluded from action supervision; short terminal chunks repeat the last command and mask padded steps. H25 and H50 are supported. No chunk export is needed.

Collect cover-removal data using [the dataset tools](../README.md), or simple cube-lift data using [pick](../pick/README.md). Only standard/train episodes enter training. Pick batches cover all 24 identity permutations; each chunk appears once per epoch.

Cache frozen vision features once, then train either action head:

```bash
python -m OATFlow.prior.features --data "$PRIOR_DATA" \
  --vision-weights "$VIT_WEIGHTS" --output "$VIT_CACHE" --ram-gib 10

python -m OATFlow.prior.train --action-head fm \
  --data "$PRIOR_DATA" --vision-cache "$VIT_CACHE" --output "$PRIOR_CHECKPOINTS" \
  --vision-weights "$VIT_WEIGHTS" --flow-weights "$FLOW_EXPERT" \
  --horizon 50 --epochs 10 --batch-size 256 \
  --lr 1e-4 --min-lr 3e-6 --warmup-steps 500
```

Omit `--vision-cache` to encode images during training. Cache hashes bind features to the manifest and vision weights; frame IDs must match the complete stride-1 training index. Features precede trainable adapters and are independent of action horizon. `config.json` and `index.json` record disk/RAM shard locations; RAM shards must remain available during training.

FM initializes its action expert from the SmolVLA export, selecting six action input columns/output rows while retaining transformer weights. The 32-wide proprio projection is a conditioning feature, not extra actions. All FM action/noise/velocity tensors are `[B,H,6]`. FM uses ten Euler steps at evaluation by default.

Use `--action-head act` for the same conditioning stack with LeRobot ACT modules: a 512-wide four-layer CVAE posterior, 32-dimensional latent and one decoder with H learned queries. Loss is masked normalized L1 plus 10 times KL. Inference uses zero latent and one decoder pass. The ACT head is initialized fresh; only its unchanged proprio projection uses the SmolVLA export. ACT trains without activation checkpointing, while FM retains it. Both heads use the same frozen vision cache.

`--resume` restores model, optimizer, scheduler and noise RNG; `--epochs` counts additional epochs. Extending beyond the original schedule holds its LR floor. `--restart-scheduler` retains model/optimizer/RNG and starts a new warmup/cosine schedule. Resume requires stride-1 data, matching head/horizon, normalization, batch and seed; without a scheduler restart LR settings must also match.

For cover-removal FM evaluation:

```bash
python -m OATFlow.prior.rollout "$EPISODE" \
  --checkpoint "$PRIOR_CHECKPOINTS/final.pt" --execute-chunk 10 \
  --max-frames 1000 --output "$ROLLOUT_OUTPUT"
python -m OATFlow.prior.evaluate --data "$PRIOR_DATA" \
  --checkpoint "$PRIOR_CHECKPOINTS/final.pt" --output "$EVAL_OUTPUT"
```

Cover success requires a released, settled cover supported inside the green zone, then bilateral target lift for ten frames. Pick-object evaluation supports both FM and ACT; see its README. Training loss alone does not establish closed-loop success.

Task IDs are `2 * pair_index + side`, with pair order `(Red,Blue), (Red,Green), (Red,Yellow), (Blue,Green), (Blue,Yellow), (Green,Yellow)` and side 0=left/1=right at post-shuffle expert start. Target identities are Red=0, Blue=1, Green=2, Yellow=3. Cover body names A/B do not encode the current side. IDs are verified against physical containment and remain fixed during cover removal.

`run.sh DATA OUTPUT VISION_WEIGHTS FLOW_WEIGHTS [expert|context] [PLAN]` collects, audits and trains an FM prior for ten epochs at batch 256, stride 1, with warmup/cosine 1e-4 → 3e-6. Context mode requires a scene plan. Activate the environment and run long jobs in tmux; stage logs are stored beside the checkpoint directory.

For ACT-style temporal ensembling during single-scene evaluation, pass `--execute-chunk 1 --temporal-ensemble-coeff 0.01` to `prior.rollout`. The installed LeRobot `ACTTemporalEnsembler` averages overlapping H25 predictions aligned to the same control timestep; weights `exp(-0.01*i)` slightly prefer older plans. All six normalized absolute setpoints are averaged before actuator clipping, including the gripper. The episode resets the ensemble. This changes query cadence to every frame, unlike K5/K10 execution, so comparisons must record both cadence and aggregation. Source: https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py (query frequency and temporal aggregation).

To compare target identities in the same scene, pass `--target RedCube` (or another configured identity) to `prior.rollout`. The physical scene and noise seed remain fixed; both the target embedding and success checks use the requested identity. The task pair and cover side are derived from physical containment at the initial state: swapping within a cover keeps the task ID, while selecting the other cover updates it. The output config records the reference/query targets, effective task ID, pair and cover. Source episode metadata is unchanged. Results identify any wrong object lifted; `object_xyz` traces all four identities in `TARGETS` order.

To use the25Hz policy/TE →50Hz servo pipeline, add `--servo-hz 50` together with `--execute-chunk 1 --temporal-ensemble-coeff 0.01`. `environment/control.py` holds the previous commanded waypoint over0–20ms, sends the linear midpoint at20ms, then the new endpoint at40ms. The endpoint remains as the start of the next segment and is not resent at that boundary. Initial q_k is the initialized actuator command. All six physical actuator coordinates, including gripper, are interpolated after endpoint clipping. Policy observations, video, inference calls and success-counter updates remain25Hz; the trace separately stores actual50Hz command timestamps. The single-scene CLI retains servo25 for baseline comparisons unless the50Hz option is selected. Simulator time advances independently of inference wall time; hardware real-time25Hz requires completing each prediction within40ms.

For asynchronous execution, use:

```bash
python -m OATFlow.prior.rollout "$EPISODE" \
  --checkpoint "$PRIOR_CHECKPOINTS/final.pt" --output "$ROLLOUT_OUTPUT" \
  --execution-mode async --execute-chunk 5 \
  --temporal-ensemble-coeff 0.01 --servo-hz 50 --max-frames 1000
```

The simulator thread copies physics states at25Hz; one background worker immediately renders the latest available snapshot and plans again after each inference finishes. It owns a separate MuJoCo data/renderer and is the sole policy caller. Rendering/input transfer are included in recorded plan latency. Inference cadence depends on latency, rather than waiting for a five-step execution block. Each block boundary freezes five temporally ensembled absolute waypoints, then interpolates and executes them at50Hz while planning continues. New results only affect the next block. `execution.py` aligns each H25 prediction to its observation step: action[j] is the endpoint at(source_step+j+1)/25s. Entries in the past are discarded; valid future entries are weighted oldest-to-newest using ACT's exponential rule. Irregular inference completions require this timestamped buffer instead of the synchronous LeRobot helper. The video is rendered from recorded rollout states after execution to keep camera/encoding work off the servo thread.

The first plan is primed before starting the execution clock. Async commands are wall-clock paced; lateness shifts the schedule forward without catch-up bursts. When no plan covers a committed timestep, hold the previous waypoint and record an underrun (`ensemble_count=0`). H25 provides one second of lookahead; K5 commits200ms at a time, so inference need not finish in40ms or even200ms, but sustained latency must leave valid forecasts for the next blocks. Trace files include prediction source/receipt steps, full plans, committed blocks, latencies, simulated50Hz timestamps and actual wall command times. This Python/MuJoCo runner measures scheduling behavior; it does not guarantee hardware hard real-time deadlines.
