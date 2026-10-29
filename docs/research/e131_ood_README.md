# E131 — broader spatial OOD distributions

Status: COMPLETE.

Freeze E131final23520, initialized from full E118PointCloudReach120000 and
finetuned point encoder/goal/state/FM by BC only on2048E117demos,32layouts,
294640stride1chunks,23520updates,B128 (~10.217epochs), FM LR2e-5,point LR1e-4,
warmup100,AdamW(.9,.95),wd1e-4,clip1,BF16. No training/model selection here.
Use the same GT-visible128point target cloud plus joints and unchanged camera.

Expand the earlier8outside-case pilot into fixed new spatial distributions:
4edges (x_low,x_high,y_low,y_high) and4corners; outward offsets uniform in
[2,4],[4,6],[6,8]cm. Training center support is world X[.30,.78],Y[-.42,-.10]m.
For corners, both coordinate offsets lie in the chosen band; distance is
L-infinity distance to the rectangle. This is OOD relative to pick finetuning,
not necessarily reach pretraining. Object geometry,camera and home stay fixed.
Other3objects have fresh random inside placements and all yaws are random.

For each cell(region,band), seek4accepted pairs per target color (16pairs).
Try at most48independent candidates per color. In each pair only target XY
changes: outside target versus coordinate-reflected inside control, with
matched distractors,target yaw,color,home and camera. Require separation>=55mm,
a valid front GT cloud (>=8filtered pixels and16raw pixels), and successful
original physical pick expert in BOTH scenes. This selection happens before
loading the learned checkpoint. Save all rejection reasons and accepted
fractions. Cells with no feasible/visible pairs are unavailable, not policy
failure rates. Evaluate all accepted pairs even when a quota is not filled;
report actual denominators and per-cell/color coverage explicitly.

Parallel independent preparation uses4CPU worker processes, each with the
same physics and camera; no sub-agents. Policy evaluation reuses unchanged
object_evaluate.rollout:H25/K15/Euler10,noise seed0,300frames,25/50/500Hz,
CUDA BF16. Success requires correct-target lift>4cm with2jaw contacts for10
frames; any distractor lift>4cm fails, even without a stable wrong-object grasp.
Missing<8filtered target pixels fails explicitly. No video or feature cache.

Output:e131_object/test_ood. Run `MUJOCO_GL=egl .venv/bin/python -u -m
OATFlow.top_pick.object_ood` in tmux with tee. Save manifest/initial states,
preparation coverage/rejections,all rollout traces/results,per-band/per-region
summaries,paired outcomes and spatial plots. Verify actual offsets/bounds,
expert successes,disjoint state paths,all accepted cases evaluated,and unchanged
checkpoint/source hashes. Results apply to expert-feasible,front-visible scenes
under this sampler; compare inside controls and do not treat rejected scenes
as policy failures. Coverage uncertainty and finite sample sizes remain explicit.
Results pending.


## Results

| Offset band | Inside controls | Outside targets |
|---|---:|---:|
| 2-4 cm | 69/69 (100.00%) | 69/69 (100.00%) |
| 4-6 cm | 63/64 (98.44%) | 64/64 (100.00%) |
| 6-8 cm | 64/64 (100.00%) | 55/64 (85.94%) |

Exact per-cell counts and rejected candidate reasons are in coverage.json and preparation_cell_*.json; summary.json retains all groups.


Completed394episodes/197matched pairs from2830candidates:54.70s physical
preparation,280.27s learned-policy rollouts,337.00s total including output.
OOD188/197=95.431%;inside controls196/197=99.492%. Paired outcomes:187both
success,9inside-only,1outside-only,0both-failure. All10failures are300frame
timeouts;0missing-cloud or non-target-lift failures.

| OOD region | 2-4 cm | 4-6 cm | 6-8 cm | All offsets |
|---|---:|---:|---:|---:|
| x_low | 16/16 | 16/16 | 16/16 | 48/48 |
| x_high | 16/16 | 16/16 | 16/16 | 48/48 |
| y_high | 5/5 | NA | NA | 5/5 |
| x_low + y_high | 16/16 | 16/16 | 8/16 | 40/48 |
| x_high + y_high | 16/16 | 16/16 | 15/16 | 47/48 |

All y_low cells and their corners lack front-visible target clouds under
the current camera. y_high4-8cm cells have no accepted expert-feasible pair
after192candidates/cell. These are NA, not0% learned success. y_high2-4cm
has only5pairs and does not fill the16pair quota; its denominator is explicit.
The six X-edge cells and six upper-corner cells each have16accepted pairs.
Do not extrapolate success to excluded portions of the workspace.

Initial scene audit confirms all197paired controls differ only in target XY:
maximum distractor position difference0m,robot joint difference0,target
quaternion difference3.28e-12,and recorded-pose error0. All394saved traces,
last10bilateral-lift frames of every success,cloud cadence/finite inputs and
checkpoint hash pass completion_audit.json. No new training or videos.

Interpretation: direct GT-visible geometry conditioning is robust on tested
X-edge shifts up to8cm and all accepted2-6cm cells. At the6-8cm upper-left
corner, success falls to8/16 despite successful expert grasps; the model is
not uniformly robust to combined-axis extrapolation. These are pick-training
OOD tests with oracle segmentation, fixed cube geometry/camera and one noise
seed. Reach pretraining covers a broader workspace. Distance-band aggregate
rates mix different regions (69/64/64outside samples), so use the per-region
table to identify the directional limitation.

Figures:xy_results.png,coverage.png,success_by_region.png. The last figure
shows exact success counts and unavailable cells without conflating them
with policy failures. Coverage figure title was shortened after completion;
all numerical data and predictions are unchanged.
