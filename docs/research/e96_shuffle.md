# e96_shuffle: context, 24 permutations, three RGB views

Status: complete; full render and audit passed. User-approved scope:40layouts,
16train permutations/8unseen permutations, held-out layouts from the same
randomization range, no perturbation during expert execution. Original robot,
scene, lights and fixed top/left/right320px cameras at25Hz; no calibration export.

Train32layouts×16permutations×4queries=2,048episodes; val4newlayouts×16seen
permutations×4=256; composition32trainlayouts×8unseen×4=1,024;
spatial4testlayouts×16seen×4=256; combined4testlayouts×8unseen×4=128.
Total3,712queries and928shared contexts. Permutation sets are unions of cyclic
orbits: exactly balanced color/slot marginals and closed under cover swapping,
so unseen initial compositions remain unseen after odd shuffle counts.
Swap-count parity is paired across block-swapped permutations, so final
compositions remain a complete balanced16/8partition as well.

Geometry/yaw is fixed per layout independently of color/query. Two covers with
two slots each; centers randomized within original e36 +/-25mm range, object
offsets +/-6mmX/+/-14mmY, yaw[-pi,pi]. Common settled home robot. Each context
has1–3swaps, variable duration, arc, direction, smooth time warp and pauses.
All4queries share identical context and expert-start physical state. Full expert
opens/deposits the correct cover, then physically grasps/lifts the queried cube.
Only successful complete layouts retained; rejected layout seeds logged.

Store physical command joints6 separately from measured joints/velocities,
EE baseXYZ/rotation6D, full qpos/qvel/ctrl, timestamp/phase/action_valid and
all4object+2cover centers/rotations at every frame. Raw units rad/metres, no
normalization; data allow later H25 EE-delta10D export. Context actions invalid.
IDs, assignments, initial/final permutations, exact shuffle plans and seeds saved.

RGB/context masks are stored once per shared context. Each query stores its
expert continuation and references the context. A loader concatenates both
segments. Masks at full320px for all3views: visible instance IDs; query target
amodal, frontmost cover occluder, outermost containing cover, plus e36 relevance
mask. Shared context records amodal masks for all4cubes/2covers, allowing any
query without duplicate rendering. Binary masks bitpacked/lossless compressed.

No training/checkpoint. Pilot checks successful physical expert, all-query shared
state, mask geometry/occlusion, frame alignment, decoding, replay and storage/
runtime estimates. Full generation runs only after these checks pass and fits
available disk. Outputs `/workspace/datasets/e96_shuffle`; pilot under
`workspace_analysis/results/e96_shuffle/pilot`. Exact settings in config.json.

Pilot:1shared context +4successful physical query rollouts, no rejected layout
seed. All4expert command replays are bit-exact. All3views fully hidden during
shuffle/hold; visible masks are subsets of amodal; containment and TCOW labels
checked. Four context+expert three-view mask-overlay videos in pilot/preview.
Pilot data13.09MB; projected full12.15GB before safety allowance. Lossless
time-contiguous NPZ storage preserves all logical shapes/values. Fixed a
near-zero settled-home TRF IK seed issue in controllers/oracle_pick.py;
ordinary nonzero seeds retain the existing behavior. Rendering enum comparison
uses int to avoid slow object comparisons; all mask arrays match bit-exactly.

Run: `bash workspace_analysis/run_shuffle.sh` (fresh output required).
Stages: collect full successful physical layouts → RGB3/masks → full audit.
Live log `workspace_analysis/logs/e96_shuffle.log`, tmux `huy:e96_shuffle`.
Policy inputs reference RGB + target ID + measured observations; full hidden
states, object XYZ, masks, selected cover and slot IDs are supervision/replay
artifacts. No GT scene state in input.json. Raw values are unnormalized;
normalization.json reports train-only statistics and fixed gripper convention.

Completion:3,712episodes/928contexts, all4,640segments rendered,1,546,042unique
frames with3fully decoded camera streams each. Actual dataset12,265,169,984bytes
(12.27GB); renderer elapsed8,716.45s (~2h25m). All160physical expert replay
checks (4queries on1context per layout) have exactly0qpos error. Split counts,
shared contexts, initial/final permutation partitions, centers/timestamps,
occlusion/masks and input-field isolation passed. config.json statuscomplete,
audit.json statuspassed. No training launched by the dataset pipeline.
