# E166 — Flow Matching resume to40k

Status: training18000→40000 in progress; evaluation pending.

Purpose: continue E165 from its **last step18000 full-state checkpoint** to **total40000** (+22000 updates), to assess continued convergence. Preserve the completed18k experiment. Constant learning rate3e-5 for the continuation; no schedule restart or new warmup.

Data/architecture/objective unchanged: E157 train1024/val128, five heldout splits128 each, original cam_front RGB320×240 with robot/chair/rug; full-scene4096 XYZ and frozen cached SAM3 vision_features256×72×72, four task IDs,128 visible target points/task. Original width128 point encoder +128-query task TransformerDecoder produce Z. Two-layer128-wide4-head point flow has full128-point self-attention and cross-attention to128 Z tokens. Trainable1,376,003; only SAM3 frozen. No GT masks/centers/size/pose as model input.

Flow path: x0~N(0,I), x1=2*GT_m, exact **unsquared Euclidean Hungarian** noise↔GT coupling (W1 matching); xt=(1-t)x0+t*x1_matched, t=1-(Beta(1.5,1)*.999+.001). Only velocity MSE. Fixed XYZ*2 normalization, no GT centering/per-axis stats. Same Haar SO(3)+translation[-.5,.5]m shared by scene/targets; features stay attached. No coordinate scaling; physical size variation is already rendered in E157.

Resume: source `/workspace/robotics_assets/experiments/e165_fm/train/resume.pt` at18000, full model+AdamW moments/counters+NumPy data RNG+CUDA SO3/translation generator+CUDA noise generator+CPU/CUDA global RNG (including timestep sampling). Carry forward best validation checkpoint17000/EMD16.60543086566031mm. Source checkpoint SHA256 and exact source hashes in `train/config.json`; verify exact restoration and matching next batch/path/loss before continuing. Parent best17k is retained as candidate if extra training never improves validation.

Training: additional22000 steps, batch8 scenes/32 tasks, AdamW3e-5 constant, wd.01, clip1; BF16 attention/MLPs with FP32 parameters/optimizer/readout/velocity MSE/Euler, TF32 off. Approximately171.875 additional scene passes,312.5 total. No architecture/loss/data/augmentation changes from E165.

Selection/test: fixed128 E137 SE3 validation every500 updates, minimum sampled EMD; canonical validation diagnostic. Inference20 Euler steps, common CPU Gaussian seed165, Z computed once. Five canonical128-scene tests (512 task queries/split), four SE3 views of128 IID scenes (2048 queries). Rotate common noise by R for SE3 tests, no translation of prior. Compare E16640k best against unchanged E16518k best and E16118k best, identical data/GT/noise/transforms; report EMD, directional CD, aligned CD/centroid, robust extents, visible-surface voxel coverage and selection. Fixed one noise bank only; no new physical target-only resize intervention. Evaluation/figure pending.

Code ownership: own package `e166`, selective copies of E165 source; no cross-experiment imports. Dedicated branch `exp/e166-fm40k`, base65c7f27. E165 source commit51a4a493b9fc14833b2e09d61a581ba61ea68078. Exact executing commit is recorded by the trainer. Datasets/features/checkpoints/logs live outside Git.

Run with the existing uv environment:
```
/workspace/arm-manipulation/.venv-concerto/bin/python /workspace/robotics_assets/experiments/e166_fm40k/code/run.py verify_resume
/workspace/arm-manipulation/.venv-concerto/bin/python /workspace/robotics_assets/experiments/e166_fm40k/code/run.py train
```
Trainer automatically evaluates the selected best checkpoint and writes best-vs-GT viewer at `http://localhost:17070/e166.html` after completion. Live logs: `ssh_tmux:e166-log`, mirrored in `/workspace/robotics_assets/experiments/e166_fm40k/run.log`.

Executing Git commit: `3475bc210d97d0cd21674895d37215476b159dc8`. Exact hashes/configuration: `config.json`; resume verification: `../verify_resume.json`.

Resume audit passed: exact model/AdamW/used RNGs; repeated next real batch/path identical; actual step18001 loss matched preflight0.004753523506224155 and data IDs. Copied architecture/flow/data/transforms/metrics bytes unchanged from E165. Training is active; metrics in train/progress.json, logs in ssh_tmux:e166-log. Evaluation pending.

Observed during E167 preparation: E166 process/window absent; last progress31150, last full-state checkpoint31000; log ends without traceback. Total40k/evaluation not completed. Cause/intent not established. Artifacts preserved.
