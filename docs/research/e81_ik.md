# e81_ik

Status: complete. Find multiple physically usable joint configurations and
trajectories for the same exact EE XYZ, using the original e08 arm/table model.
Orientation remains unconstrained, as in e08. No training, weights or FM sampling.

16 accepted e08 goals: first nearest the coordinate-wise workspace median, then
farthest-point sampling. 128 multistart IK solves per goal; seed81. Analytic
position Jacobian, bounded SciPy TRF, max150function evaluations, tolerances1e-9.
Keep endpoint position error<=0.5mm and collision-free configurations at9gripper
closure settings. Greedy diverse selection requires >=15degrees shortest-angle
difference in at least one of the first4arm joints; wrist roll is excluded.
These counts refer to separated solutions, not a topological count of IK branches.

For up to8solutions/goal: collision-checked path from the same settled home,
original50Hz absolute servo commands and500Hz physics, quintic movement,
Jacobian-based alignment, close0.6s,hold0.5s. Original e08 acceptance criteria:
arrival/final<=1mm, closing drift<=2mm, penetration<=0.2mm, gripper closes.
No policy H/K, Euler steps, learned checkpoint or inference noise applies.

Output: `/workspace/checkpoints/e81_ik`. Exact actual configuration: `config.json`.
Results: `summary.json`, per-goal `v*/summary.json`, `candidates.npz`, and verified
trajectory files `v*/t*.npz`. Reusable solver: `workspace_analysis/multi_ik.py`.
Tmux `huy:e81`; log `/workspace/memory_occlusion/logs/e81_ik.log`.

Run: `.venv/bin/python workspace_analysis/multi_ik.py --output /workspace/checkpoints/e81_ik`


Completed: {"status": "complete", "goals": 16, "starts_per_goal": 128, "goals_with_multiple_ik": 9, "diverse_count_min": 1, "diverse_count_median": 2.0, "diverse_count_max": 15, "goals_with_multiple_verified_trajectories": 9, "verified_trajectory_count": 43, "verified_count_min": 1, "max_final_EE_error_mm": 0.12512475782138296, "max_closing_hold_EE_error_mm": 0.8535422894008349, "elapsed_s": 10.15596255401033}

Per-goal results and actual trajectories: summary.json and v*/.

Independent saved-state audit passed: verification.json. Six actual final arm poses at the same goal are shown in same_goal.png; coordinates/joints: example.json.

Preferred visualization: same_goal_side.png (side view avoids arm-plane occlusion). Render FK checks: render_checks.json. Original same_goal.png is retained.
