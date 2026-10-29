# E122 — K15 action execution

Status: COMPLETE — all1408K15episodes evaluated.

Evaluate the same E122final23520checkpoint on the same original1408held-out
episodes,changing action execution prefix from K10toK15. Predict H25,execute
actions0–14,then observe/replan. Keep flow Euler integration at10steps.

Training: full E117pick_wide_rgbd,2048expert demos/32layouts,294640stride1chunks,
23520updates/10.217214effective epochs,B128. FM/state initialized from reach-trained
E118best120k;ImageNet ViT-B/16 frozen/online.400RGB+aligned XYZpatches fuse before
target-query decoder; same hidden conditions FM and128point auxiliary cloud head.
Train geometry/fusion/query/projections/cloud head/FM jointly,FM+1.0normalized
squared Chamfer. AdamW(.9,.95),wd1e-4,clip1,BF16,FM LR2e-5/other1e-4,warmup100.
Exact training:`../train/config.json`; no new training or weights selected on test.

Test: checkpoint final23520,H25/K15/Euler10,noise0,300frames at25Hz,
servo50Hz/physics500Hz. Original composition1024/spatial256/combined128cases,
same manifest/order/batch32/initial physics states as K10. Native front RGB-D,
calibration,target color ID and measured joints; no GT target mask/XYZ input.
Output:chunk-anchor EE deltaXYZ,relative rotation6D,absolute jaw via same IK.
Success:correct-target lift>4cm,bilateral contacts10consecutive frames;
wrong-object lift fails. Save physical traces/results,no videos in full test.

Baseline K10:1028/1408=73.0114%,composition699/1024,spatial218/256,combined111/128.
K8was interrupted at1120/1408cases,651successes; not a completed global benchmark.
Its composition group is complete:588/1024=57.4219%,versus K10 699/1024=68.2617%.

Verify first10commands/physical states against the same K10first32cases; the
first plan should agree until K10's earlier replan at frame10. At completion,
verify checkpoint/manifest/H/flow steps/seed/frame budget match before writing
paired1408case `comparison_k10.json`. Same seed is the same RNG protocol,
not identical wall-frame noise across different replan cadences.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.top_pick.full_evaluate --k 15
```

Outputs:`e122_full/test_k15/config.json`, `progress.json`, `results.jsonl`,
`summary.json`, `comparison_k10.json`; preserve K10and partial K8artifacts.
Log:`/workspace/robotics_assets/metadata/e122_k15.log`;tmux`ssh_tmux:3`.

LAUNCH VERIFIED: completed64ordered cases,K15successes57,same-caseK10successes49;not an overall test estimate. First10commands/qpos bit-exact for first32cases,then commands differ in32cases at K10replan frame10. Same checkpoint/manifest/H/Euler/seed/budget verified. Full test continues in ssh_tmux:3; exact verification:launch_verification.json.

COMPLETE: K15 1144/1408 (81.250%) versus K10 1028/1408 (73.011%),delta+8.239percentage points. Same checkpoint/manifest/protocol verified. Paired outcomes:{'both_success': 979, 'K15_only': 165, 'both_failure': 215, 'K10_only': 49}. Exact comparison:test_k15/comparison_k10.json.

E122 K15 final verified:1144/1408=81.25%,K10 1028/1408=73.0114%,+116successes/+8.2386percentage points. Composition811/1024=79.1992% versus699/1024=68.2617%;spatial220/256=85.9375% versus218/256=85.1563%;combined113/128=88.2813% versus111/128=86.7188%. All264failures timeout,no wrong-object lift. Paired outcomes:979both-success,165K15-only,49K10-only,215both-failure. K15test1207.22s(20.12min). Same final23520/H25/Euler10/noise0/300frames/original1408paths verified; K8remains interrupted. Exact source:e122_full/test_k15/comparison_k10.json.
