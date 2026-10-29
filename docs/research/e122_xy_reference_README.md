# E122 XY pilot: CPU reference on existing in-distribution test cases

Status: COMPLETE. No training. E122 final23520, original training config preserved in the parent test config. Frozen ViT; same trained geometry/query/FM. First four original held-out manifest episodes, selected by manifest order independently of their outcomes. Historical K15 CUDA BF16 outcomes: three successes, one failure. CPU FP32 uses its own seed0 noise stream, so trajectories need not match CUDA even with the same seed.

Protocol: original initial states, scene and camera; H25/K15,Euler10,300frames,25Hzpolicy/50Hzservo/500Hzphysics; same success metric. This checks gross CPU execution differences, not statistical equivalence. Parent test: ../README.md. Files: config.json,manifest.jsonl,results.jsonl,summary.json. No video.

{"cases": 4, "successes": 2, "success_rate": 0.5, "elapsed_s": 50.67204111498722, "reasons": {"success": 2, "timeout": 2}, "by_group": {"composition": {"cases": 4, "successes": 2, "success_rate": 0.5}}, "by_target": {"BlueCube": {"cases": 1, "successes": 1, "success_rate": 1.0}, "GreenCube": {"cases": 1, "successes": 0, "success_rate": 0.0}, "RedCube": {"cases": 1, "successes": 1, "success_rate": 1.0}, "YellowCube": {"cases": 1, "successes": 0, "success_rate": 0.0}}, "status": "complete", "historical_cuda_successes": 3, "matched_outcomes": 3}
