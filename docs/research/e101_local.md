# e101_local

Status: complete; 1500 updates, best step1250.

Budget clarification: same512/2048training query subset as e100,8/32train layouts,
and128shared contexts. No complete-epoch sweep: randomized phase-weighted prefix
endpoints with replacement. At best1250:5000presentations,4703distinct endpoints
of92234eligible; at final1500:6000presentations,5566distinct endpoints. These
counts replay exactly to saved checkpoint NumPy RNG for both models. Each clip
supervises30past/current frames, so endpoint coverage differs from image coverage.
Only129target-cube lift/hold endpoints were sampled by best1250. This is a matched
pilot ablation, not evidence that full-data TCOW training would also fail.

One-camera TCOW point-query ablation of e100_point. Keep the radius3 binary dot
at frame0, RGB/query backbone, XYZ head and full-finetune recipe. Add a fixed
appearance reference: select the patch with most initial dot pixels, average its
3x3 patch neighborhood from the final backbone features at frame0, project both
reference/current patch features with shared LayerNorm768 and Linear768->128,
L2-normalize, and add 5*cosine similarity to the learned spatial pooling scores.
The neighborhood is 48x48pixels except clipping at borders. Its feature covers
local patch positions but also inherits spatial context from the transformer.
Only frame0 defines this appearance reference; no future frames or GT masks do.
No object ID, initial XYZ, joints or camera parameters enter the model. Only XYZ
labels supervise learning. Query annotation is an offline initial 2D click.

Fresh official TCOW initialization, including original fourth-channel weights;
random head/projection, seed100 (same XYZ head and backbone initialization as
e100). Full finetune all parameters from update1, no frozen stage. Top camera
320square center-cropped to240x320(rows40:280); patch16,T30,width768,depth12,
heads12,causal divided space/time attention,first-frame CLS,drop-path0..0.1.
Descriptor adds99,840 trainable parameters; total121,850,888. RGB(.45,.225),
query unchanged. Train-only XYZ mean/std,10mm floor. No calibration input.

Same e96 subset:512train query episodes/128shared shuffle contexts,32val,
64unseen permutation,32unseen layout,64both unseen. Layout/permutation splits,
30frame prefix sampling,phase probabilities(.15,.35,.2,.3),source stride3,
batch4/micro2,AdamW(backbone1e-5,head3e-4),100warmup,cosine0.1floor,WD1e-4,
clip1,BF16/TF32,1500updates match e100. Every prefix starts atframe0 and ends at
its sampled past/current endpoint. Sparse history and excluded final cached
training endpoint remain unchanged, to isolate the query-reference change.
Loss0.5normalized endpoint XYZ MSE+0.5prefix MSE, no mask loss.

Validate every250updates at cover-open t_obj before grasp; choose lowest mean
opening XYZ error. Final evaluate all phases, hidden/visible and held-out
permutation/layout groups. Run future-value/gradient/prefix checks, same-RGB
four-query binding and two videos. No policy,IK,H/K,flow/noise settings apply.
Comparison against e100 best1250 must include hidden and lift failure, not only
opening. Test status: complete. Artifacts/config/subset/metrics/test/predictions
and videos in /workspace/checkpoints/e101_local. Weights best/latest plus one
final optimizer snapshot. Official weights retained in vision_init/.

Run from /workspace/arm-manipulation:

```bash
PYTHONPYCACHEPREFIX=/tmp/tracking_xyz_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 .venv/bin/python -u -m OATFlow.tracking.train_point --local-query --steps 1500 \
 --output /workspace/checkpoints/e101_local
```

tmux huy:e101_local; log workspace_analysis/logs/e101_local.log.
Tests: `.venv/bin/python -m unittest OATFlow.tests.test_point_xyz OATFlow.tests.test_tracking_xyz -v`.

Actual validation opening mean 25.411mm. Results in test.json; videos in test_videos/.

Matched e100/e101 comparison (same GT, episode order, normalization and seed):
val: opening 23.099->25.411mm; hidden 36.335->36.835mm; lift 99.142->96.902mm.
composition: opening 19.254->24.799mm; hidden 31.544->29.196mm; lift 103.988->101.463mm.
spatial: opening 29.336->34.260mm; hidden 35.754->32.748mm; lift 102.291->95.859mm.
combined: opening 28.748->33.839mm; hidden 36.516->32.333mm; lift 100.679->96.033mm.
Exact metrics/comparison in test.json and comparison.json; comparison.png/PDF. All24focused tests, actualGPUgradient preflight, trained causal checks, prediction reconstruction/source hashes/best-step checks passed. Two videos fully decoded. No claim of reliable tracking follows from opening error alone.
