# E122 — K8 action execution

Status: INTERRUPTED —1120/1408episodes,partial results retained.

Change only the executed action prefix from K10toK8: predict H25actions,execute
the first8,then observe/replan. This is not a change to flow integration steps
or temporal ensembling. Same final checkpoint step23520 and original1408held-out
episodes as the completed K10test. Output:`e122_full/test_k8`; keep K10intact.

Training: full E117pick_wide_rgbd,2048expert demos/32layouts,294640stride1chunks,
23520updates/10.217214effective epochs,B128. Entire FM/state initialized from
E118reachbest120k;ImageNet ViT-B/16 frozen/encoded online.400RGB+aligned XYZpatches
fuse before target-query decoder; its hidden conditions FM and128point cloud
auxiliary head. Train geometry/fusion/query/projections/cloud head/FM jointly.
FM+1.0normalized squared Chamfer,AdamW(.9,.95),wd1e-4,clip1,BF16,
FM LR2e-5/other1e-4,warmup100. Exact config:`../train/config.json`.

Test: final23520,H25/K8,Euler10,noise seed0,300frame budget at25Hz,
servo50Hz/physics500Hz. Original composition1024/spatial256/combined128cases,
same order/batch32/source initial physics states. Native front RGB-D/calibration,
target color ID and measured joints; no GT target mask/XYZ input.
Actions:chunk-anchor EE deltaXYZ/relative rotation6D/absolute jaw via same IK.
Success:correct target lift>4cm with bilateral contact10consecutive frames;
wrong-object lift fails. Save traces/results; no videos in the full evaluation.

K10baseline:1028/1408=73.0114%,composition699/1024,spatial218/256,combined111/128.
Use paired episode outcomes and verify checkpoint/manifest/H/flow steps/seed/frame
budget equal before writing `comparison_k10.json`. Same seed means the same RNG
protocol,not identical frame-by-frame noise across different replan cadences.
Do not generalize early ordered partial results to the whole test distribution.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.top_pick.full_evaluate --k 8
```

Sources:`config.json`, `results.jsonl`, `summary.json`, `comparison_k10.json`;
baseline:`../test_k10/config.json`, `../test_k10/summary.json`.
Log:`/workspace/robotics_assets/metadata/e122_k8.log`;tmux`ssh_tmux:2`.

LAUNCH VERIFIED/RUNNING: first32episodes saved. Across all32cases, first8commands and resulting qpos are bit-exact to K10; action index8changes in all32cases, confirming the new replan cadence. Same checkpoint/manifest/H25/Euler10/seed0/300frame budget verified. First ordered composition-only batch:K8=15/32,K10=18/32; not a full-test estimate. Full1408case run continues,approximately30minutes. Verification:launch_verification.json. Initial parameter-shadowing bug fixed before any episode was recorded.

Interrupted before K15 request:651/1120successes,58.125% on ordered partial test; no full summary. Composition complete588/1024=57.4219%,K10same group699/1024=68.2617%. Preserve all original outcomes/traces. Actual partial counts:partial_summary.json.
