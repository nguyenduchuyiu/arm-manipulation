# E130 — target-cloud location while the gripper occludes the object

Status: COMPLETE.

Use unchanged E122 final23520; no architecture or training changes. Original
training: 2048 E117 expert demos, 32 layouts, stride1, B128, 23520 updates,
frozen ViT-B/16, FM from E118 point-cloud reach120000. Exact original settings
are copied from the checkpoint into config.json. No optimizer is constructed.

Replay saved states only. Cases: first success and first failure in each of
composition/spatial/combined groups of the completed K15 test (6 cases), plus
all16 paired inside/outside X-boundary cases. Selection is fixed by recorded
order/pair membership, not cloud error or measured occlusion. These are
selected diagnostics, not success-rate estimates. Original action protocol:
H25/K15/Euler10/seed0/300frames. No physics steps or action resampling here.
Recompute auxiliary clouds in CUDA FP32, TF32 disabled, at all K15 replans
and every3frames from15frames before through30frames after the first closing
command (right jaw command<-5mm). Include exact onset and onset+1. Report
fresh clouds and clouds held from the last K15 replan separately.

GT is diagnostic only. Render the same target pose from the same front camera
with all other geometry hidden: an oracle unoccluded front-visible surface.
The target's original complete-scene mask divided by this oracle projected
mask gives visible fraction. Classify >=75% as mostly visible,25-75% partial,
<25% heavy; separately count <8actual visible pixels. This is not a measure
of full3D surface completeness. Generate oracle GT depth with identical2mm
quantization/calibration, and use the true simulator body pose/oriented box
for geometry checks. No GT information enters prediction.

Metrics: centroid error against oracle unoccluded surface, Chamfer against
that surface (no centroid alignment), mean unsigned distance to the true
oriented box surface, fraction of predicted points within5mm of that surface,
cloud-centroid distance to actual body center and EE. A partial-surface cloud
centroid is not intrinsically the physical object center. The head always
outputs128points and has no visibility/confidence output; the policy has no
temporal visual tracker. Display persistence alone cannot show tracking.

Output: /workspace/robotics_assets/experiments/e130_probe/test_occlusion.
Run `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.top_pick.occlusion_probe`
in tmux with tee. Save records/config/clouds, visibility/error timelines and
RGB/GT/prediction sample sheets. Verify unchanged checkpoint/model/trace
hashes, no model gradients, segmentation subset, GT box consistency and
held-cloud cadence. Results below.


## Results

22 selected traces and 714 replayed samples completed in 25.60 s. No new
physics or actions, and no model changes. All source/checkpoint/model hashes,
visible-mask subset and held-cloud cadence checks pass. Oracle GT surface
distance to the true object box averages 0.275 mm, validating the reference.

For the first 15 frames after the closing command in the six held-out clips:

| Recorded outcome | Cases / samples | Median target visible | Fresh centroid error | Held centroid error |
|---|---:|---:|---:|---:|
| Success | 3 / 22 | 93.50% | 6.85 mm | 8.12 mm |
| Failure | 3 / 24 | 77.84% | 42.86 mm | 43.74 mm |

Errors above compare predicted centroid with the oracle unoccluded
front-visible surface centroid at the actual target pose. They are not
physical-body-center errors. Correctness is additionally checked against
the complete simulator oriented-box surface. These outcome-selected groups
differ in pose, visibility and trajectory; the contrast is descriptive,
not a causal effect of occlusion or a population failure estimate.

At the most occluded sampled time within onset..onset+30, the three held-out
success clips retain 90.1-93.2% of target projected pixels; fresh errors are
4.37/6.73/6.22 mm. This does not demonstrate successful severe-occlusion
tracking. Two held-out failure examples retain 51.7%/40.6% of target pixels
and have fresh centroid errors 45.10/46.44 mm. Their mean predicted-point
distances to the real object-box surface are 28.64/32.42 mm, respectively.

Strong occlusion example: xy_inside x_high/t0/inside, frame190, one frame
after closing starts. Only12.5% of projected target pixels are visible.
Fresh centroid error55.51mm; held error43.22mm (age10frames). Fresh mean
distance to actual box surface39.41mm. The held centroid error vector is
approximately[-9.82,-33.68,+25.25]mm in base XYZ, illustrating that projected
2D overlap can hide substantial3D error. See sample_15.png/timeline_15.png.

Across all sampled times, 6samples from2inside-boundary cases have <25%
visibility; fresh median centroid error52.56mm, held43.10mm. None of the
714samples has fewer8visible target pixels, so complete-occlusion tracking
is not tested. Remaining geometry priors can produce a plausible cloud
without reliable localization; the head always emits128points.

Display staleness example: xy_inside x_low/t2/inside, frame155. Fresh error
10.87mm versus held46.20mm (last replan150,age5frames=0.20s). The actual
body moved[2.16,-3.51,+51.39]mm since that replan. The visible overlay can
therefore be stale even when a fresh prediction is substantially better.
At K15/25Hz, a displayed cloud can be held for up to14frames=0.56s.

Artifacts: sample_00.png (small error, successful grasp), sample_03.png and
sample_05.png (partial-occlusion failures), sample_15.png (strong occlusion),
timeline_11.png (fresh versus stale while lifting). Orange points are oracle
GT rendered without occluders for diagnostics only; cyan is the held cloud,
projected through geometry as an x-ray overlay. Cases are described fully
in cases.json; per-sample metrics and explicit per-case selection are in
records.json and analysis.json. Full cloud/GT/body positions are in clouds.npz.

Timeline-only revision shortens vertical labels; numeric inference/metrics
are unchanged. Original inference-source hash and figure-source hash are
recorded separately in config.json.
