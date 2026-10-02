# Memory and occlusion task

`MemoryOcclusionEnv` in `environment/env.py` owns the MuJoCo model in
`environment/scene.xml`. The scene
uses the existing five-joint NexArm and gripper in `assets/robot/robot.xml`,
four small HOPE targets (Butter, Popcorn, Milk, Tuna), two visually identical
30 g covers, and separate green cover and blue object drop zones. The fixed
overview camera records aligned RGB and metric depth. The wrist camera is
available in the Gym observation but is not a model input for this task.

`controllers/oracle_pick.py` is the shared physical pick/place controller.
`dataset/episode.py` runs a complete expert episode with MuJoCo contacts, without
attaching an object to the gripper. Scene setup moves covers and their hidden
objects together; the robot starts acting only after the shuffle.

The same environment also exposes Gymnasium `reset`, `occlude`, `query`,
`choose_cover`, and `step`. Its observation includes overview RGB-D, wrist RGB,
the query image, and raw robot joint positions and velocities. `step` returns
`(observation, reward, terminated, truncated, info)`: reward is 1 on complete
success, a wrong cover or object ends the episode, and the time limit truncates.

## Episode

1. Reveal all four objects.
2. Cover two objects with each cover, shuffle the covers, and hold for one second.
3. At `t_occ`, choose the cover containing the queried target (`0` left,
   `1` right at that instant).
4. At `t_cover_grasp`, the gripper has lifted the chosen cover. Place it in the
   green zone. At `t_obj`, start the target pick. At `t_object_grasp`, the
   gripper has lifted the target; place it in the blue zone.

A grasp event requires both jaws touching the body while it is at least 1 cm
above its starting height for three consecutive 25 Hz frames. The timestamps
are saved in `episode.json` as `grasp_frames` and `grasp_times_s`, and a
frame-aligned `supervision.npz/grasp_event` uses 1 for cover and 2 for object.
These are supervision labels and are absent from the model-facing `input.json`.

The query is one fixed close-up RGB image per HOPE object, shared across episodes.
The active policy receives overview RGB-D history, the visible target mask on
the first frame, and six normalized absolute joint positions. The same scene
can use different target queries with different correct cover IDs.

Each episode is one continuous 320 × 320 RGB-D recording at **25 fps**. Context
can be sampled at 8 fps while loading; the saved video stays at 25 fps. Expert
commands are absolute joint positions at 25 Hz. The five arm values are in
`[-1, 1]`; the gripper is in `[0, 1]` (0 closed, 1 open).

| File | Contents |
| --- | --- |
| `rgb.mp4` | Full RGB video |
| `observation.npz` | Metric depth, normalized joint positions, timestamps |
| `supervision.npz` | Dense task-relevance mask and entity, expert actions, action validity, phase |
| `tcow_labels.npz` | Target, occluder and container masks for auxiliary TCOW supervision |
| `input.json` | Model-facing paths and decision frames |
| `episode.json` | Sim-only IDs and assignments, split seed, phase intervals, event frames/times, outcomes |
| `debug_poses.npz` | Frame-aligned privileged object/cover poses, body names, timestamps |
| `depth_preview.mp4`, `mask_preview.mp4` | Optional inspection videos for samples; bulk generation omits them |

`supervision.npz/mask` labels the entity currently relevant to the query in
every 25 Hz frame. ID 2 is the queried object during reveal and partial
occlusion. Once it is fully hidden, ID 1 labels its responsible cover through
shuffle, pick, carry, placement, and gripper release. Only after the cover
release finishes does ID 2 return, even if the object became visible while the
cover was still held. The target remains labeled through its own placement
and release. After that, ID 0 marks the completed task. The two event frames
are in `episode.json/semantic_transition_frames`; `semantic_entity` stores the
active entity ID even when its visible segmentation is temporarily empty.
Preview videos color the object yellow and the cover blue.

`dataset.reader.decision_observation(directory, "t_occ", context_fps=8)` loads
the observation at a decision point with only earlier video frames.
`input.json` has only model-facing paths and decision boundaries; the IDs,
assignments, poses, and event labels stay in `episode.json`, `debug_poses.npz`,
and `supervision.npz` for debugging and split construction.
`evaluation/metrics.py` contains cover selection accuracy and full task success
calculations.

The active memory occlusion policy is documented in
[`experiments/tcow_joint_flow/README.md`](experiments/tcow_joint_flow/README.md).

## Run

Use the project `.venv` from the repository root:

```bash
.venv/bin/python -m memory_occlusion.dataset.generate_samples
.venv/bin/python -m memory_occlusion.dataset.validate
```

These commands write and validate one example per HOPE target under
`outputs/memory_occlusion/samples`. To create a single target example, use
`python -m memory_occlusion.dataset.generate_samples --targets Tuna` in the
same environment. For a larger dataset, run
`python -m memory_occlusion.dataset.generate_dataset --episodes 4 --output outputs/memory_occlusion/semantic_dataset`.
Use a new output directory for dense labels: older generated episodes contain
sparse masks and cannot be mixed with them. It assigns train/val/test by episode seed
(80/10/10), keeping all queries of one layout in the same split. Generation
stops if the expert fails and does not label a failed rollout as success.

The scene uses Newton contact solving, elliptic friction cones, and
`impratio=10` to reduce gradual slip from MuJoCo's compliant contacts. The
seed-0 one- and two-swap and seed-1 one-swap checks succeeded for all four
queries. These are simulation results; the grasp parameters are not calibrated
to physical hardware. New-layout and unseen-object evaluation remain to be run.

## Local 25 Hz TCOW and Flow Matching data

The frozen plan `dataset/scene_plan.jsonl` contains 200 standard and 20
composition scenes, with four target queries per scene (880 episodes).
From the repository root, collect both cameras, depth and TCOW masks, audit
all episodes, then fit TRAIN normalization:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m memory_occlusion.dataset.generate_dataset \
  --plan memory_occlusion/dataset/scene_plan.jsonl \
  --output memory_occlusion/datasets/memory_occlusion_multiview_25hz_delta_v1 \
  --action-mode delta --workers 8
```

For absolute joints, use `--action-mode absolute` and a fresh output directory
such as `memory_occlusion_multiview_25hz_absolute_v1`. Gripper is absolute in
both modes. Delta processing follows collection; chunks use H25, stride10 and
a fixed observed-state anchor. Both datasets are self-contained. All stages
show tqdm progress. Existing output directories are rejected.

To train only Flow Matching on Apple MPS after data generation and the TCOW
checkpoint download finish:

```bash
.venv/bin/python -m memory_occlusion.experiments.tcow_joint_flow.train_from_tcow_context \
  --data memory_occlusion/datasets/memory_occlusion_multiview_25hz_delta_v1 \
  --weights memory_occlusion/checkpoints/tcow_rgbd_pretrained_joint1200_20260930.pth \
  --flow-weights memory_occlusion/checkpoints/memory_occlusion_action_expert_init_20261001/expert.safetensors \
  --output memory_occlusion/checkpoints/flow_mps_25hz \
  --device mps --flow-only --batch-size 1 --cluster-size 2
```

This mode freezes TCOW, recomputes its query-to-current 30-frame context for
each action sample, and updates only the Flow Matching branch on 25-step action
targets every 10 frames. `--config-checkpoint` is needed only when training
TCOW and Flow Matching jointly.

The active action expert and dense TCOW connection are documented
in [tcow_joint_flow](experiments/tcow_joint_flow/README.md).
