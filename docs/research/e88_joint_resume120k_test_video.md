# e88_joint/resume120k/test_video

Status: complete. Snapshot checkpoint: checkpoint.pt, e88_joint step115000.
Source: /workspace/checkpoints/e88_joint/resume120k/best.pt, validation-selected
best after120k total updates. Snapshot SHA256/provenance in checkpoint_selection.json.

Training: same e86_grid measured EE labels,H25,25Hz,stride2,input6 measured five
arm joints(rad)+measured right-jaw(m),goal baseXYZ3,output10 deltaXYZ/relative
rotation6D fixed chunk-start pose/absolute jaw. Original random seed0; continued
model/fusedAdamW/RNG/sampler after60k through120k,LR2e-5, B256/BF16,all learned
modules train. Train-only joint/goal/output pose mean/std; jaw fixed range[-1,1],
physical metres on disk/sample. Parent config.json contains actual full settings.
No new training/optimizer updates in this test.

Four held-out test goals chosen by same seed90 initial-distance quartiles as the
e87 step35k video, before observing rollout results; enables matching goal comparison.
H25/K15,flow10,noise0,max300policy frames,25Hzwaypoints/50Hzmidpoint servo,
500Hzphysics, same bounded PoseIK/bias-force compensation. No FK pose input to
policy; current measured EE is used only by controller to decode chunk-relative
predictions. Original NexArm visual STL meshes, dynamics checked equal to physics.

Videos25fps: green actual EE,blue dashed full recorded expert path,orange current
full H25 prediction,red goal. Expert future is visual only. Static base-frame
3D plot includes actual trajectory and executed IK pose targets. Replay saved
servo commands and verify against original rollout traces; fully decode combined
MP4 to check encoding. Four individual videos plus comparison.mp4,trajectories.png
and .pdf,overview.png,rollout.npz,rollout_cartesian.npz,summary.json/replay_checks.json.
Results complete; summary.json, rollout.json and replay_checks.json. Actual test config in config.json. Code: workspace_analysis/fm_ee_video.py.
Tmux:huy:vid88; log:workspace_analysis/results/e88_joint/resume120k/test_video/logs/video.log.


Actual results:

```json
{
  "cases": 4,
  "final_mean_mm": 4.539420498951385,
  "final_median_mm": 4.8912758959342435,
  "final_p90_mm": 5.968098073714432,
  "collision_rate": 0.0,
  "clipped_frame_fraction": 0.24333333333333335,
  "success_1mm": 0.0,
  "success_2mm": 0.25,
  "success_5mm": 0.5,
  "success_10mm": 1.0,
  "success_20mm": 1.0,
  "ik_over_1mm_fraction": 0.0,
  "ik_mean_rotation_deg": 0.025851307925741345
}
```
