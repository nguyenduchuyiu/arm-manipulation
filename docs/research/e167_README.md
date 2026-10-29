# E167 — Scratch center/Rg-conditioned spherical Flow Matching

Status: training18k in progress; evaluation pending.

Purpose: learn target center and scalar size, initialize128 points on the predicted sphere, then refine geometry with FM. **User explicitly requested training from scratch.** All point encoder, task TransformerDecoder, center/Rg heads and flow weights initialize randomly (seed135). No E158/E159/E165 checkpoint or probe weights are loaded for training. Only pretrained SAM3 is frozen/cached. E159's positive size probe motivates the hypothesis; it is not a guarantee for this scratch model. E16640k continues independently.

Data unchanged E157: train1024, val128, five heldout splits128 scenes each; original fixed cam_front RGB320×240, robot/chair/rug, four independently sized/posed red-blue-green-yellow cubes. Full-scene4096 XYZ plus matching frozen per-point256D SAM3 vision_features, task ID0–3, target128 visible points. Shared feature/point association preserved. Full Haar SO(3)+translation[-.5,.5]m on scene and GT before global XYZ*2; no coordinate scale augmentation, physical sizes already rerendered. No GT mask/center/size/pose as model input.

Architecture: unchanged128-wide two-layer point TransformerEncoder + two-layer128-query task-conditioned TransformerDecoder yields Z[B*4,128,128]. New center MLP128→128→3 applied per query then mean pooled; positive radius head Linear128→1 on mean Z, radius=.02*softplus(logit)+1e-6m, initially20.001mm. New heads are trainable from scratch. Point flow unchanged E165: two layers of full128-point self-attention +cross-attention to128 Z, heads4/FF384/dropout0, FP32 velocity readout initialized zero. Shared grounding/flow initial parameters match E165 exactly before the extra heads.

Prior:128 unit-sphere surface directions,64 random unit vectors plus their antipodes. Thus empirical mean is0 and RMS radius is1. P0=predicted_center+predicted_Rg*directions, giving exact predicted center and Rg. These are points on a sphere surface, not uniform ball volume. Fresh CUDA directions each train batch, seed167. Fixed per-task CPU direction bank167 for validation/test; rotate bank by R in coordinate-SE3 tests. Radius here means GT128 radius of gyration, not cube edge length or circumscribed radius.

Supervision: GT128 center=mean(G), Rg=sqrt(mean(||G-center||²)). Only labels; never supplied to forward/prior. Exact unsquared Euclidean Hungarian assignment between predicted P0 and GT, in fixed XYZ*2 units; xt=(1-t)P0+tGmatched, t=1-(Beta(1.5,1)*.999+.001), velocity target=Gmatched-P0. Detach P0 during path/target construction to avoid moving-label gradients into center/Rg heads. FM still updates shared Z through context. Geometry heads train explicitly with MSE.

Loss: velocity_MSE + center_MSE + Rg_MSE, weights1/1/1. Center/radius errors multiplied by2 before squaring, consistent with normalized FM coordinates. All terms have squared-distance units; no EMD/pair auxiliary reconstruction loss.

Training:18,000 updates, batch8 scenes/32 tasks, all non-SAM parameters trainable, fresh AdamW3e-4→3e-5 cosine/warmup100, wd.01/clip1; data seed135, SE3 seed137, direction seed167. About140.625 scene passes. BF16 attention/MLPs, FP32 parameters/optimizer/velocity/MSE/center and radius heads/Euler, TF32 off. Every500 updates select minimum20-step sampled EMD on fixed128 E137 SE3 validation scenes; canonical validation diagnostic. Log geometry-head center error/Rg MAE/R² and prior EMD as well as output EMD. No test feedback.

Evaluation planned: best checkpoint, five canonical128-scene heldout splits (512 task queries each), four IID SE3 views (2048 queries),20 Euler steps. Compare E167 output, its own predicted sphere before FM, E16518k Gaussian FM (original fixed Gaussian seed165), E16118k direct XYZ. Same scratch18k budget but prior/supervision/extra parameters change together; not a single-factor architecture comparison. Report EMD, directional CD, aligned CD, centroid error, robust extents, visible-surface coverage, Rg R² and selection. No new physical target-only scale interventions or multiple-noise-bank evaluation in this initial run. GT describes visible surfaces only.

Ownership: own e167 source package in isolated branch exp/e167-sphere based on65c7f27; selective copies from E165, no cross-experiment imports. Actual source SHA256 and exact executing Git commit in config.json. Checkpoints/caches/logs outside Git. No weights reused for training except cached frozen SAM3.

Commands:
```
/workspace/arm-manipulation/.venv-concerto/bin/python /workspace/robotics_assets/experiments/e167_sphere/code/run.py verify
/workspace/arm-manipulation/.venv-concerto/bin/python /workspace/robotics_assets/experiments/e167_sphere/code/run.py train
```
Live log `ssh_tmux:e167-log`, mirrored at `/workspace/robotics_assets/experiments/e167_sphere/run.log`. Automatic evaluation/report/best-vs-GT viewer at `http://localhost:17070/e167.html` after training. See verify.json and train/config.json for the executing status.

Executing Git commit: `f09089542910679b89adef3029951ce2cad49459`. Exact hashes/configuration: `config.json`; verification: `../verify.json`.

Verification passed: scratch shared grounding/flow init bit-identical E165; no checkpoint loading in train/model/objective; correct sphere centroid/Rg and SE3 invariance; exact Hungarian matching; detached FM path/target; supervised geometry-head gradients;6fullGPUupdates/20stepinference finite. Actual trainer running, first250 updates finite. Current code commit f09089542910679b89adef3029951ce2cad49459. Runtime train/config.json and progress.json are authoritative; evaluation pending.
