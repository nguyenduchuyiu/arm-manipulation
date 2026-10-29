# E131 — full held-out GT-cloud BC evaluation

Status: COMPLETE.

Evaluate final23520 of E131, initialized from full E118PointCloudReach120000,
trained jointly point encoder/goal/state/FM with BC only on all2048E117train
expert demos,stride1,B128,23520updates. No RGB or task embedding. Full optimizer,
LR and normalization settings are copied into config.json from the checkpoint.

Test all original1408held-out episodes:1024composition,256spatial,128combined.
H25/K15, Euler10, noise seed0,300frame limit,25Hzpolicy/50Hzservo/500Hzphysics.
Use GT visible target mask, actual front depth and calibration to construct
128base-frame XYZ points; no GT center,object pose or hidden surface input.
Missing<8filtered pixels fails explicitly as missing_target_cloud. No stale
cloud fallback. Correct-target lift>4cm,two jaws,10consecutive frames defines
success; wrong-object lift fails. No videos. This is an oracle segmentation
baseline, not a deployed perception system. No training in this evaluation.

Run `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.top_pick.object_evaluate`.
Output:e131_object/test_k15. Files:config.json,results.jsonl,summary.json,
progress.json,verification.json,and saved action/state/input-cloud traces.
Results pending.

Results:
{
  "cases": 1408,
  "successes": 1325,
  "success_rate": 0.9410511363636364,
  "elapsed_s": 1024.689148180987,
  "reasons": {
    "success": 1325,
    "timeout": 82,
    "wrong_object": 1
  },
  "by_group": {
    "combined": {
      "cases": 128,
      "successes": 120,
      "success_rate": 0.9375
    },
    "composition": {
      "cases": 1024,
      "successes": 960,
      "success_rate": 0.9375
    },
    "spatial": {
      "cases": 256,
      "successes": 245,
      "success_rate": 0.95703125
    }
  },
  "by_target": {
    "BlueCube": {
      "cases": 352,
      "successes": 330,
      "success_rate": 0.9375
    },
    "GreenCube": {
      "cases": 352,
      "successes": 333,
      "success_rate": 0.9460227272727273
    },
    "RedCube": {
      "cases": 352,
      "successes": 333,
      "success_rate": 0.9460227272727273
    },
    "YellowCube": {
      "cases": 352,
      "successes": 329,
      "success_rate": 0.9346590909090909
    }
  },
  "status": "complete"
}

Completed audit: all1408records/traces/checkpoint hashes pass. Compared
with E122K15 on the same manifest and matching protocol fields:1325versus1144
successes,+12.855percentage points. Comparison is an oracle-input/architecture
baseline, not an isolated perception-module ablation. See comparison_e122.json.
