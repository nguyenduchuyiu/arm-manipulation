# e88_joint/resume120k/test_boxes

Status: complete. Frozen e88 joint-input best115k checkpoint after120k training.
Source checkpoint and recovered original OATFlow scene/metric provenance: provenance.json.

Challenge: actual4OATFlow boxes, center of each box as the fixed absolute EE goal.
Use original environment/scene.xml body definitions and fixed slots,32mm cubes,
30g,original inertial/friction/visual/collision geometry. Reuse exact configured
e86 robot/table physics; no covers/occlusion/context furniture. Original NexArm
STL meshes. Four independent resets with all4boxes present, common e86 robot home.
Settle0.5s; freeze goal at target measured center after settling. Do not snap to
grid or add vertical grasp offset. Model gets joint5+measured jaw and goalbaseXYZ3.

Same input/output normalization and best checkpoint, no training or optimization.
Original training: B256,BF16,fusedAdamW,seed0,all learned modules trained;
60k random-init then60k continuation atLR2e-5. Output deltaXYZ3/relative6D6 with
fixed measured chunk-start pose anchor plus absolute FM-predicted jaw command.
Policy stage300frames,H25/K15,flow10,noise0,25Hzwaypoints/50Hzmidpoint servo,
500Hzphysics,same PoseIK max40nfev. No forced jaw closure, manual EE orientation,
expert approach, object attachment or pose teleportation during execution.

Grasp diagnostics: both original jaw collision geoms contact target, count
consecutive frames; inspect object displacement and finger contacts. After FM
reach/close, a clearly separate hand-coded IK lift probe translates actual EE
up80mm over3s and holds1s, maintaining measured orientation and the last clipped
FM jaw command. This probe tests retention only; it is not learned lifting.
Pick success follows original OATFlow physical rule: target center rises>40mm
and both jaws contact for10consecutive25Hzframes. Primary FM reach/close and
probe results reported separately. Cube-table/jaw-target contacts are expected;
report unintended robot collisions separately. Check no distractor is lifted.

Export four videos plus combined video with original robot, actual EE trajectory,
current H25 plan and all four cubes; show overview and target close-up. Save full
qpos/qvel/commands/plans/object/jaw/contact traces. Verify command replay and
visual model dynamics. Exact scene.xml/config.json/summary.json beside artifacts.
Results: summary.json and rollout.json. Local canonical doc:docs/research/e88_joint_test_boxes.md.
Tmux:huy:boxes; log:workspace_analysis/results/e88_joint/resume120k/test_boxes/logs/test.log.


Actual results:

```json
{
  "status": "complete",
  "checkpoint_step": 115000,
  "cases": 4,
  "records": [
    {
      "target": "RedCube",
      "goal_world_m": [
        0.41,
        -0.24,
        0.01589224457372347
      ],
      "reach_final_error_mm": 60.54671403488937,
      "reach_jaw_measured_m": -0.02549983861466552,
      "reach_jaw_command_m": -0.02548784576356411,
      "reach_final_jaw_contacts": 0,
      "reach_longest_bilateral_frames": 0,
      "reach_bilateral_contact_held": false,
      "reach_target_shift_mm": 5.8440640079070825,
      "probe_target_max_lift_mm": 6.060384832862198e-09,
      "probe_target_final_lift_mm": 6.0603431994987744e-09,
      "probe_final_jaw_contacts": 0,
      "longest_lifted_bilateral_frames": 0,
      "grasp_retention_success": false,
      "distractor_max_lift_mm": 7.991166756093904e-09,
      "unintended_collision": true,
      "peak_unintended_penetration_mm": 0.7518315507895414,
      "peak_unintended_pair": [
        "link_6_left_jaw_collision_0",
        "link_6_right_jaw_collision_0"
      ],
      "clipped_frames": 125,
      "FM_ik_position_max_mm": 0.04485873529464497,
      "probe_ik_position_max_mm": 1.2357541961148856e-06
    },
    {
      "target": "BlueCube",
      "goal_world_m": [
        0.49,
        -0.24,
        0.01589224457372347
      ],
      "reach_final_error_mm": 58.119687840454795,
      "reach_jaw_measured_m": -0.025453244117232557,
      "reach_jaw_command_m": -0.0255,
      "reach_final_jaw_contacts": 0,
      "reach_longest_bilateral_frames": 0,
      "reach_bilateral_contact_held": false,
      "reach_target_shift_mm": 4.727394455836883,
      "probe_target_max_lift_mm": 6.060374424521342e-09,
      "probe_target_final_lift_mm": 6.060374424521342e-09,
      "probe_final_jaw_contacts": 0,
      "longest_lifted_bilateral_frames": 0,
      "grasp_retention_success": false,
      "distractor_max_lift_mm": 7.991166756093904e-09,
      "unintended_collision": false,
      "peak_unintended_penetration_mm": 0.0,
      "peak_unintended_pair": null,
      "clipped_frames": 153,
      "FM_ik_position_max_mm": 0.5979473101798044,
      "probe_ik_position_max_mm": 1.3649950688783666e-06
    },
    {
      "target": "GreenCube",
      "goal_world_m": [
        0.61,
        -0.24,
        0.01589224457372347
      ],
      "reach_final_error_mm": 34.016571927930286,
      "reach_jaw_measured_m": -0.01017720413633448,
      "reach_jaw_command_m": -0.025448618456721306,
      "reach_final_jaw_contacts": 2,
      "reach_longest_bilateral_frames": 103,
      "reach_bilateral_contact_held": true,
      "reach_target_shift_mm": 20.78717903792858,
      "probe_target_max_lift_mm": 79.84298780890795,
      "probe_target_final_lift_mm": 79.84210076295717,
      "probe_final_jaw_contacts": 2,
      "longest_lifted_bilateral_frames": 63,
      "grasp_retention_success": true,
      "distractor_max_lift_mm": 7.991166756093904e-09,
      "unintended_collision": true,
      "peak_unintended_penetration_mm": 6.511334125727477,
      "peak_unintended_pair": [
        "link_6_gripper_base_collision_0",
        "GreenCube_collision"
      ],
      "clipped_frames": 145,
      "FM_ik_position_max_mm": 0.04150801141195833,
      "probe_ik_position_max_mm": 1.1212010843366031e-06
    },
    {
      "target": "YellowCube",
      "goal_world_m": [
        0.69,
        -0.24,
        0.01589224457372347
      ],
      "reach_final_error_mm": 87.28247252669442,
      "reach_jaw_measured_m": -2.4166961136362415e-05,
      "reach_jaw_command_m": 0.0,
      "reach_final_jaw_contacts": 0,
      "reach_longest_bilateral_frames": 0,
      "reach_bilateral_contact_held": false,
      "reach_target_shift_mm": 14.507996863172085,
      "probe_target_max_lift_mm": 6.060377893968294e-09,
      "probe_target_final_lift_mm": 6.060367485627438e-09,
      "probe_final_jaw_contacts": 0,
      "longest_lifted_bilateral_frames": 0,
      "grasp_retention_success": false,
      "distractor_max_lift_mm": 0.06906361282112139,
      "unintended_collision": true,
      "peak_unintended_penetration_mm": 6.876600271029123,
      "peak_unintended_pair": [
        "link_6_gripper_base_collision_0",
        "YellowCube_collision"
      ],
      "clipped_frames": 245,
      "FM_ik_position_max_mm": 8.453390377562478,
      "probe_ik_position_max_mm": 1.220185966321118e-06
    }
  ],
  "FM_bilateral_contact_cases": 1,
  "retention_probe_successes": 1,
  "replay_check": "passed",
  "visual_dynamics_check": "passed",
  "video_decode_check": "passed"
}
```


Interpretation: GreenCube has103consecutive bilateral-contact frames during FM
reach/close, then passes the separate IK retention probe (max lift79.843mm,
63consecutive lifted bilateral-contact frames). Other3targets fail retention.
This is not a learned FM lift. GreenCube also has unintended gripper-base contact
(max penetration6.511mm), so this is not a clean collision-free grasp. The box
was displaced20.787mm during reaching. Reach EE-center errors are60.547,58.120,
34.017,87.282mm for red/blue/green/yellow. Every saved-command replay is bit-exact.
Nearby train-goal coverage (XY Linf radius20mm): minimum Z30,30,30,50mm respectively;
requested actual object centers are15.892mm. See goal_coverage.json; this does not
prove continuous geometric infeasibility, only lack of those low nearby training goals.


Lighting diagnosis: the video had0world lights and headlight ambient0.5/diffuse0.8/specular0.5; original OATFlow has2world lights plus headlight0.3/0.4/0.15. Same GreenCube frame300 and identical camera/geometry/physics rendered in lighting_comparison.png; exact settings in lighting_diagnostics.json. replay_ee_pose.visual_model now accepts an explicit lighting_scene to recover original illumination; prior videos remain intact.
