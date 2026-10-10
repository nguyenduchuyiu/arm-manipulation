# OATFlow

TCOW remembers a queried object through cover occlusion and shuffling. Its current features, wrist RGB and proprioception condition a flow-matching action policy. Train a vision/FM manipulation prior first, then initialize the main policy from that prior and train TCOW + FM end-to-end.

The task has two covers and four colored targets: RedCube, BlueCube, GreenCube and YellowCube, two objects per cover. Each target is a homogeneous32×32×32mm cube,30g, with a centered center of mass and identical inertia on all axes; all six resting faces have the same geometry. Free joints retain natural translation, rotation and contact physics. Objects start80mm apart along the cover's long axis, with randomized in-box position and yaw; cover centers are also randomized. The expert puts the selected cover in the broad green drop zone, then lifts one target with both jaws. It stops after the target is lifted >4cm with bilateral contact for ten control frames. All learned actions are normalized **absolute joint setpoints**: five arm joints in `[-1,1]`, gripper in `[0,1]`. Expert speed is always fast; lift/release height is 12cm. No target placement stage.

## Data collection

Both modes use `dataset/episode.py` for recording and `dataset/expert.py` for the physical expert. Store full overview/wrist RGB at 320x320 and 25Hz, joints, executed expert commands, phase labels and simulation states. Stored H25 action chunks and training chunks start at every valid expert frame (stride 1); short terminal chunks have masked padding. Frame `t_occ` marks the expert start; context and the terminal frame have `action_valid=False`.

| Mode | Initial scene and recorded trajectory | Labels |
|---|---|---|
| `expert` | Both covers already closed; randomized covered scene → expert | 12 pair/side task IDs, four target IDs, absolute joints |
| `context` | Reveal → cover occlusion → randomized shuffle → expert | Same task/target IDs and actions, plus query mask and three TCOW masks |

The 12 prior task IDs encode the chosen object pair and its current left/right side; the target ID identifies which object of that pair to lift. Main TCOW uses the initial query mask and video history rather than oracle task/target IDs. After a shuffle, choose the cover drop goal using its **current side**, rather than its body name.

```bash
# Prior: 12 tasks x 50 demos, two targets balanced within each task.
python -m OATFlow.dataset.generate_dataset --mode expert \
  --demos-per-task 50 --start-seed 16000 --workers 4 --output "$PRIOR_DATA"
python -m OATFlow.prior.audit "$PRIOR_DATA" --demos-per-task 50

# Four small context + expert episodes.
python -m OATFlow.dataset.generate_dataset --mode context \
  --episodes 4 --start-seed 0 --workers 1 --output "$CONTEXT_DATA"

# Full dataset: 1,200 independent prior-style context + expert demos.
python -m OATFlow.dataset.make_scene_plan "$PLAN_DIR" --preflight
python -m OATFlow.dataset.generate_dataset --mode context \
  --plan "$PLAN_DIR/plan.jsonl" --workers 4 --output "$CONTEXT_DATA"
```

Use a fresh output directory. The plan preflight uses the same prior setup and physical expert as collection. Regenerate the plan for the current recipe. `dataset/references.py` still creates separate reference PNGs per object, linked from episode metadata. They do not replace the first-frame TCOW query mask.

Both modes call the same `expert.setup_scene`: cover XY±25mm, randomized object placement/yaw, initial arm joints±0.06rad and randomized grasp offsets. Context mode parks the covers after this setup, records reveal/occlude/shuffle, then runs the same expert from the resulting state. An odd number of swaps reverses the initial task's side so the final covered task matches the requested ID. Object identities/positions follow their cover continuously; the scene is not reset at the expert boundary.

The full plan has1,200 independent seeds and one target per seed:840 standard/train demos(70per task),120val(10per task),120test(10per task),120composition. Both targets within each standard task are balanced. Standard splits balance1/2/3shuffles while respecting withheld composition cases. Preflight verifies the selected physical query from that seed; failed seeds are replaced without changing its task/target quota. Seed splits never overlap. No repeated four-query scene expansion is used.

Context labels use overview crop rows `40:280`, shape 240x320, while raw videos remain full 320x320. `policy/data.py` applies that crop when building TCOW clips. At each sample, the clip contains 30 frames from the first query frame through the current frame; it never includes future frames. Context samples supervise masks only; expert samples supervise masks and FM actions. Normalize only standard/train expert frames. The prior trainer can also read context datasets: it uses current full RGB at expert chunk starts and excludes validation/test scenes and all context-only samples.

To continue an already initialized live simulator, call `dataset.expert.execute(env, target, rng, record)`. The callback receives body-prefixed expert phases; `t_obj` marks the target segment without adding a frame. Both CLI collection modes call this same function.

## Training and evaluation

See [prior/README.md](prior/README.md) for manipulation pretraining and [policy/README.md](policy/README.md) for prior transfer and joint TCOW/FM training. The single-GPU and distributed trainers share initialization. Joint training keeps the action-loss gradient through TCOW; the wrist encoder can remain frozen independently.

For online refinement of a pretrained BC prior, see [rl/README.md](rl/README.md):
covered random resets, frozen observation/conditioning stack, and FM-only
ReinFlow/PPO updates with K5 execution.

Dependencies/setup are documented in the [root README](../README.md). Datasets, weights, caches, logs and videos belong outside the source repository.
