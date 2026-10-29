# e82_mm

Status: 64-goal preflight complete; full-workspace collection not launched.
Planned scope: all32,017goals. Multimodal reaching data for exact e08
voxel XYZ; original robot/table, fixed home, no objects/images, orientation free.

Keep original e08 trajectory as mode0 (byte-exact copy). Find up to7additional
trajectories per goal with at most128IK starts (seed82), stopping at8accepted
modes total. Seeds: knownIK,zero,3nearest FK seeds,remaining uniform joint range.
TRF analytic position Jacobian,max150evaluations,start tolerances1e-9. Require
IK error<=0.5mm and endpoint collision-free at9gripper closure settings.
Final measured modes must differ >=15degrees in at least one of first4joints,
using shortest-angle difference; wrist-roll-only variations are excluded.
Collision-free candidate attempts are also deduplicated at15degrees.
Counts are separated solutions, not a count of distinct topological IK branches.

Each new trajectory is physically executed from the same settled home using
the original50Hz absolute actuator commands and500Hz physics, collision-checked
joint planning/quintic timing, Jacobian alignment, close0.6s,hold0.5s. Final and
arrival EE<=1mm, closing/hold drift<=2mm, penetration<=0.2mm; gripper must close.
Audit every saved trajectory and physically replay32new modes independently.

Export action_absolute[::2] at25Hz,H25,chunk-start stride2; mask padded tails and
never cross trajectories. Export mode IDs as metadata, not policy inputs.
Reuse exact e78 train/val/test goal membership (seed78); all modes/chunks of
one goal stay together. No normalization fitted. Original50Hz states are the
chunk anchors; export does not perform a new25Hz physical simulation.

No training/checkpoint/frozen modules/epochs/batch/LR or FM inference H/K/Euler
steps/noise seed applies. CPU30workers. Tmux huy:e82; main log
`/workspace/memory_occlusion/logs/e82_mm.log`.

Output: `/workspace/datasets/e82_mm`. Raw: trajectories/v*/m*.npz,index.jsonl,
goals.jsonl,config.json,summary.json,audit.json. Chunks/splits:
chunks_25hz_h25_s2/. Sampler can use goal+mode metadata to balance sampling.
Some goals can retain one mode under the search budget; no failed trajectories
are substituted or duplicated to meet the8mode cap.

Run into a fresh directory from the repository root:
`.venv/bin/python workspace_analysis/multimodal_dataset.py --output /workspace/datasets/e82_mm`
Preflight: same pipeline with --limit64 at `/workspace/checkpoints/e82_mm/test_64`.

Completed preflight:64goals,437verified trajectories,63multimodal goals,
35,003H25chunks. Minimum within-goal final mode separation15.0065deg;
maximum final EE error0.123983mm. All trajectories audited;32new modes replayed
independently,maximum recorded/replayed EE discrepancy0.0000531mm. Pipeline
elapsed17.61s,including collection13.62s. Exact settings/results:
`/workspace/checkpoints/e82_mm/test_64/config.json` and `summary.json`.

The user questioned whether multiple solutions for the exact same goal are
necessary. The current complete small dataset is retained for a controlled
comparison before expanding collection. Different solutions at different exact
goals establish variation/discontinuity of labels, not multiple conditional
outcomes at the same (goal, initial robot state). Multiple same-goal trajectories
are an optional diagnostic; they are not required for learning a reaching policy.
