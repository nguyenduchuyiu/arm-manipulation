# e100_point

Status: complete; 1500 updates, best step1250.

Budget clarification: pilot subset only,512/2048training queries and8/32training
layouts. Training samples prefix endpoints with replacement, not complete epochs.
Best1250has5000clip presentations/4703distinct query-endpoint clips out of92234
eligible endpoints in this subset; final1500has6000presentations/5566distinct
endpoints. Sampler replay matches saved checkpoint NumPy RNG exactly for e100 and
e101. Endpoint coverage is not unique-image coverage: each clip contains30frames.
Only129sampled endpoints at best1250fall in target-cube lift/hold phases.
No full-data/long-budget tracker control establishes the cause of poor tracking.

Pilot: adapt the official TCOW causal one-camera RGB+query-channel backbone to
predict the queried object's amodal XYZ center. Query mask becomes one small
filled disk (radius3pixels,29pixels away from borders), at frame0 only. It marks
the target's projected center at initialization. All later query frames are zero.
No object/color ID, initial3D XYZ, joints, masks or camera parameters enter the
model. XYZ labels are loss/evaluation supervision only.

Native PyTorch implementation in OATFlow/tracking/point.py; no third-party runtime.
Backbone matches official TCOW: width768,12blocks,12heads,patch16,T30, divided
space/time attention, causal_attention=1, first-frame-only CLS, norm_embeddings=False,
drop-path linearly0to0.1. RGB normalized(.45,.225), query channel unchanged.
Input top RGB320square center-cropped to240x320(rows40:280), following TCOW's
4:3 input convention, without distorting geometry. Temporal embeddings are sliced
for shorter prefixes so extending a prefix does not change earlier embeddings;
ordinary training/evaluation always uses30frames. Dense mask/flag heads are replaced
by a small learned spatial pooling/XYZ head: patch scoring, pooled768D feature plus
pooled2D patch position, LayerNorm,Linear770->256,GELU,Linear256->3. No target
embedding or cross-camera fusion remains.121751048parameters, all trainable.

Initialization: official TCOW pretrained backbone, including its learned fourth
query-channel convolution. RGB tensors exactly match the retained compact
tracking_backbone.pt; query extras recovered from the official archive in RAM,
without saving the1.35GBarchive or1.47GBtraining checkpoint. Only794784bytes
were added as tracking_query.pt. Original provenance in tcow_provenance.json.
Head random,seed100. Backbone eval equivalence against upstream Block implementation
verified to maximum7.16e-7 on a controlled multi-frame/multi-batch test.

Data:/workspace/datasets/e96_shuffle. Same approved e97 subset:512train,
32val,64unseen-permutation,32unseen-layout,64both-unseen query episodes.
Train layouts0..7 and16seen permutations; val layouts32/33 and4seen permutations;
test layouts36/37 and held-out compositions. All splits preserve shared contexts
and four queries. Only top RGB decoded/cached in host RAM, no disk feature cache.
Initial UV annotation is generated offline from context frame0 geometry and the
saved camera rig, then rasterized. Only this2D click enters the network; neither
future centers nor future images define the prompt. Train loads no mask files;
top-view occlusion fractions are evaluation-only hidden/visible labels.

1500full-finetune updates, logicalbatch4/microbatch2,30frames each. Shared context
and common endpoint for four distinct target dots during reveal/shuffle/hold;
separate expert suffix endpoints otherwise. Phase probabilities.15/.35/.2/.3.
No frozen or head-only stage. AdamW,backboneLR1e-5/headLR3e-4,100warmup,
cosine0.1floor,WD1e-4,clip1,BF16,TF32,seed100. Train XYZ mean/std only,10mm std floor.
Loss0.5normalized endpoint MSE+0.5normalized prefix MSE. No mask loss.

Validation every250updates at cover-open t_obj before grasp; choose minimum mean
XYZ error, including initialization as a candidate. Final validation and all three
test groups also report reveal/shuffle/post-shuffle/opening/lift, hidden/visible,
XYZ axis errors and1/2/5/10/20/50mm rates. Trained-checkpoint future RGB gradient,
future value intervention and prefix-extension checks use FP32 with TF32 disabled.
Two causal top-video overlays: first val episode and first combined held-out
episode, every cached frame (stride3plus final), nominal25/3FPS. No robot policy,
IK,action H/K,flow steps or rollout noise applies.

Checkpoint storage: weights-only best/latest; one final optimizer.pt contains
AdamW and RNG states for latest, rather than duplicating optimizer every save.
Expected total around1.95GB. Root artifacts:/workspace/checkpoints/e100_point,
config/subset/normalization/queries/cache/preflight/provenance/progress/metrics/
causality/test JSONs, predicted coordinates, weights and two videos. Full results are complete. The point prompt changes the selected target, but
precise3D localization/tracking has not been achieved in this pilot.

Run from /workspace/arm-manipulation:

```bash
PYTHONPYCACHEPREFIX=/tmp/tracking_xyz_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 .venv/bin/python -u -m OATFlow.tracking.train_point --steps 1500 \
 --output /workspace/checkpoints/e100_point
```

tmux huy:e100_point; log workspace_analysis/logs/e100_point.log.
Tests: `.venv/bin/python -m unittest OATFlow.tests.test_point_xyz OATFlow.tests.test_tracking_xyz -v`.
Source:point.py,point_data.py,train_point.py,point_video.py and shared causal blocks/data.

Actual validation opening mean 23.099mm. Results in test.json; videos in test_videos/.

Final opening mean XYZ error: val23.099mm, unseen-permutation19.254mm,
unseen-layout29.336mm, both-unseen28.748mm. Both-unseen within5/10/20mm:
1.56/7.81/25.00%. Hidden-phase mean31.544/35.754/36.516mm across the three
held-out groups. Lift-phase mean103.988/102.291/100.679mm: substantial failure
when the target is lifted; opening accuracy must not be reported as full tracking
success. Best selected at1250, not the final1500step.

Same-RGB four-query diagnostic on one validation context: initial mean error
13.262mm, prediction pairwise separation177.007mm vs GT178.511mm; after shuffle
error29.788mm, separation139.222mm vs GT178.511mm. Thus the dot affects target
selection; this diagnostic does not establish reliable identity tracking in all scenes.

Trained FP32 causal check: future RGB gradient0, future-value effect0, shorter
prefix prediction difference1.67e-6 normalized units. All23focused tests passed.
Saved predictions exactly reconstruct every reported metric; source hashes and
validation best-step selection verified. evaluation.png/PDF show opening learning
curve and phase breakdown. Videos fully decoded200/202frames. The downloaded
archive and temporary preparation/execution scripts were not retained.
