# Reusable tracking and reach code

Current modules: [tracking](tracking/README.md), [reach](reach/README.md),
[localization](localization/README.md), [simulation](simulation/README.md),
[data_tools](data_tools/README.md). Selected assets: `/workspace/robotics_assets`.
Trackers accept an initial2Dpoint; reach outputs10D chunk-relative EE pose plus
absolute gripper. The independent prior pipelines below use absolute joint commands.


The [native point-query tracker](tracking/README.md) predicts a queried object's
amodal center in robot-base coordinates from synchronized RGB history. It has
no third-party tracker dependency. [Manipulation priors](prior/README.md) use
current-frame RGB, task/target identities and measured joints with FM or ACT.

The task has two covers and four colored targets: RedCube, BlueCube, GreenCube and YellowCube, two objects per cover. Each target is a homogeneous32×32×32mm cube,30g, with a centered center of mass and identical inertia on all axes; all six resting faces have the same geometry. Free joints retain natural translation, rotation and contact physics. Objects start80mm apart along the cover's long axis, with randomized in-box position and yaw; cover centers are also randomized. The expert puts the selected cover in the broad green drop zone, then lifts one target with both jaws. It stops after the target is lifted >4cm with bilateral contact for ten control frames. All learned actions are normalized **absolute joint setpoints**: five arm joints in `[-1,1]`, gripper in `[0,1]`. Expert speed is always fast; lift/release height is 12cm. No target placement stage.

## Data collection

Both modes use `dataset/episode.py` for recording and `dataset/expert.py` for the physical expert. Store full overview/wrist RGB at 320x320 and 25Hz, joints, executed expert commands, phase labels and simulation states. Stored H25 action chunks and training chunks start at every valid expert frame (stride 1); short terminal chunks have masked padding. Frame `t_occ` marks the expert start; context and the terminal frame have `action_valid=False`.

| Mode | Initial scene and recorded trajectory | Labels |
|---|---|---|
| `expert` | Both covers already closed; randomized covered scene → expert | 12 pair/side task IDs, four target IDs, absolute joints |
| `context` | Reveal → cover occlusion → randomized shuffle → expert | Same task/target IDs and actions, plus query mask and three TCOW masks |

The 12 prior task IDs encode the chosen object pair and its current left/right side; the target ID identifies which object of that pair to lift. The retained tracker uses top-camera RGB history and a2Dpoint prompt at frame0. After a shuffle, choose the cover drop goal using its **current side**, rather than its body name.

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

Use a fresh output directory. The plan preflight uses the same prior setup and physical expert as collection. Regenerate the plan for the current recipe. `dataset/references.py` still creates separate reference PNGs per object, linked from episode metadata. The retained tracker selects its target using the initial2Dpoint.

Both modes call the same `expert.setup_scene`: cover XY±25mm, randomized object placement/yaw, initial arm joints±0.06rad and randomized grasp offsets. Context mode parks the covers after this setup, records reveal/occlude/shuffle, then runs the same expert from the resulting state. An odd number of swaps reverses the initial task's side so the final covered task matches the requested ID. Object identities/positions follow their cover continuously; the scene is not reset at the expert boundary.

The full plan has1,200 independent seeds and one target per seed:840 standard/train demos(70per task),120val(10per task),120test(10per task),120composition. Both targets within each standard task are balanced. Standard splits balance1/2/3shuffles while respecting withheld composition cases. Preflight verifies the selected physical query from that seed; failed seeds are replaced without changing its task/target quota. Seed splits never overlap. No repeated four-query scene expansion is used.

Legacy context data retain mask supervision for inspection. The native tracker
uses the top camera from the shuffle dataset and base-frame center labels; no mask
labels enter the model. Its prefix loader reads only past/current images.

To continue an already initialized live simulator, call `dataset.expert.execute(env, target, rng, record)`. The callback receives body-prefixed expert phases; `t_obj` marks the target segment without adding a frame. Both CLI collection modes call this same function.

## Training and evaluation

See [tracking](tracking/README.md) for native amodal XYZ training/testing and
[prior](prior/README.md) for manipulation pretraining. Shared FM/batching code
remains in policy; the old TCOW trainers and rollout wrappers are removed.

For online refinement of a pretrained BC prior, see [rl/README.md](rl/README.md):
covered random resets, frozen observation/conditioning stack, and FM-only
ReinFlow/PPO updates with K5 execution.

Dependencies/setup are documented in the [root README](../README.md). Datasets, weights, caches, logs and videos belong outside the source repository.
