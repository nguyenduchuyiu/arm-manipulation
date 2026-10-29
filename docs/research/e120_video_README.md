# E120 — Success and failure videos

Status: COMPLETE — six verified clips; no aggregate test conclusion.

Replay saved physical K10 test trajectories from the E120 final checkpoint
step23,520, exact80epochs. Training: E117 part1/8,256 demonstrations,
37,536stride1chunks; batch128, FM LR2e-5/other LR1e-4,100step warmup, loss FM+
weight1 target-cloud Chamfer. Frozen ImageNet ViT; train XYZ patch/fusion,
query decoder/projections/cloud head and FM initialized from E118reach120k.
Inputs are full front RGB, depth, calibration, task ID and measured joints;
GT target mask is only auxiliary supervision, not a policy input.

Test: H25/K10, Euler10,noise0,300frame maximum,25Hz policy/50Hz servo/500Hz physics.
Correct target lift>4cm with bilateral jaw contact for10frames is success;
wrong-object lift fails. Full3200episode evaluation remains independent/running.

Select the first completed success and first completed failure from each group
in a fixed results snapshot: composition(unseen permutation),spatial(unseen
layout),combined(both). Outcome-selected clips illustrate behavior and must not
be used to estimate success rate. Exact selected cases and checkpoint hash are
recorded in `config.json` before rendering.

Each25FPS clip has the native front RGB view enlarged2x and an oblique view of
the same saved robot/object configurations. Orange target annotation and green
measured EE path are diagnostics only. No policy inference or physics rerun;
no auxiliary predictions. On-screen time, target lift and jaw contacts follow
the saved trajectory. Video resolution1344×700. Contact sheet: `contact_sheet.png`.

Verify complete video decode, frame count and unchanged physical trace hashes;
record results in `verification.json`. Source: `OATFlow/top_pick/fusion_video.py`.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.top_pick.fusion_video
```

Rendered from final23520 (80epochs). Selection snapshot:256completed cases,13successes; this ordered partial set is not the final benchmark.

- `success_composition.mp4`: g000/p01/t0, RedCube, success, 5.04s.
- `failure_composition.mp4`: g000/p01/t1, BlueCube, timeout, 12.04s.
- `success_spatial.mp4`: g001/p07/t1, BlueCube, success, 6.12s.
- `failure_spatial.mp4`: g001/p00/t0, RedCube, timeout, 12.04s.
- `success_combined.mp4`: g002/p21/t3, YellowCube, success, 7.80s.
- `failure_combined.mp4`: g001/p01/t0, RedCube, timeout, 12.04s.
