# E131 — paired outside-X evaluation

Status: COMPLETE.

Same E131final23520,training and GT-visible-cloud input as the full held-out
K15test; full training config is copied into config.json. No new training.
Reuse the exact16initial states of E122test_xy:8pairs,4colors x2X edges,
target30mm outside versus30mm inside pick-train X[0.30,0.78]m. Other object
positions/yaw/home are paired. These are physically validated initial states,
not extrinsic shifts. H25/K15/Euler10/seed0/300frames,25/50/500Hz rates,
same success/missing-cloud rules as the main test. No videos. GT selection
makes this an oracle-segmentation diagnostic, not a perception evaluation.

Run `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.top_pick.object_evaluate --xy`.
Output:e131_object/test_xy. Save exact manifests/protocol, all16results/traces
and summary. This small pilot is not a precise population success estimate.
Results pending.

Results:
{
  "cases": 16,
  "successes": 15,
  "success_rate": 0.9375,
  "elapsed_s": 14.823790958005702,
  "reasons": {
    "success": 15,
    "timeout": 1
  },
  "by_group": {
    "inside": {
      "cases": 8,
      "successes": 8,
      "success_rate": 1.0
    },
    "outside": {
      "cases": 8,
      "successes": 7,
      "success_rate": 0.875
    }
  },
  "by_target": {
    "BlueCube": {
      "cases": 4,
      "successes": 4,
      "success_rate": 1.0
    },
    "GreenCube": {
      "cases": 4,
      "successes": 3,
      "success_rate": 0.75
    },
    "RedCube": {
      "cases": 4,
      "successes": 4,
      "success_rate": 1.0
    },
    "YellowCube": {
      "cases": 4,
      "successes": 4,
      "success_rate": 1.0
    }
  },
  "status": "complete"
}

Completed audit:16records/traces pass;inside8/8,outside7/8. GreenCube at
x_low is the only outside timeout. No missing-target-cloud failures. Earlier
E122XY test used CPU FP32; this uses CUDA BF16. This8outside-case pilot tests
pick-training bounds, not previously unseen reach-pretraining workspace.
