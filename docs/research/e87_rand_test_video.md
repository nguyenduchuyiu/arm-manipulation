# e87_rand/test_video

Status: complete. Snapshot checkpoint: checkpoint.pt, e87_rand step35000.
Selected across available e87/e88/e89 validation histories by stable5mm success,
then median final EE error. Selection record/hashes: checkpoint_selection.json.
Selected validation5/32success@5mm, median15.9085mm; training still active.

Training config: parent config.json and docs/research/e87_rand.md. Random init,
e86_grid measured EE labels, state10/output10,H25,25Hz,stride2,B256,BF16,fusedAdamW,
LR2e-4warmup100/cosine to2e-5 at60000,all learned parameters train.
No new training or weight updates in this video test. Snapshot prevents trainer
best.pt replacement from changing the evaluated weights.

Test: four goals from held-out test split, one deterministic seed90 sample from
each quartile of distance from initial EE, selected before rollout outcomes.
Noise seed0,H25/K15,flow10,300frame budget; same poseIK/servo/500Hzphysics.
Original NexArm STL visuals; rendering model's dynamics checked equal to physics.
Replay saved servo commands and check measured EE against rollout traces.
Videos25fps: green actual EE trail,blue complete recorded expert EE path,
orange current H25 prediction,red goal. Expert path is a visual reference only;
no expert future/joints are fed to policy or controller during learned rollouts.
Emit four videos, combined video,trajectory figure and numerical traces/metrics.
Results complete; summary.json, rollout.json and replay_checks.json. Exact actual test settings: config.json after setup.


Actual results:

```json
{
  "cases": 4,
  "final_mean_mm": 28.87219303561944,
  "final_median_mm": 27.781224010815933,
  "final_p90_mm": 42.70807427158992,
  "collision_rate": 0.25,
  "clipped_frame_fraction": 0.12,
  "success_1mm": 0.0,
  "success_2mm": 0.0,
  "success_5mm": 0.0,
  "success_10mm": 0.0,
  "success_20mm": 0.0,
  "ik_over_1mm_fraction": 0.0,
  "ik_mean_rotation_deg": 0.09083113283581605
}
```


Case4 contact diagnosis: collision is self-contact between base_link_collision_0 and link_2_collision_0, not table/gripper contact. Penetration peaks at0.302205mm at3.012s; exceeds0.201mm metric threshold during2.872–3.672s (401physics samples,0.802s). Robot keeps moving; collision is an episode flag and does not terminate rollout. Collision geoms are separate from original visual meshes. Exact pair/positions/forces/replay check: contact_diagnosis.json.
