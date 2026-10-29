# E133 — Recorded SAM3 two-view policy videos

Status: COMPLETE.

Replay four existing E133 test trajectories: first heldout RedCube success; first GreenCube success in the x_high/y_high OOD6–8cm cell; first BlueCube timeout in the x_low/y_high OOD6–8cm cell; first YellowCube non-target-lift failure. This intentionally selected gallery illustrates outcomes; it is not a random evaluation subset.

Frozen official SAM3 BF16 and frozen E131 final23520, no new training/inference, physics stepping, noise sampling or actions. Source test protocol: H25/K15/Euler10/seed0/300frames, policy25Hz / servo50Hz / physics500Hz. E131 training and experiment architecture are unchanged; see e133_README.md. Model startup, real inference waits, IK computation time and transport are excluded from video playback.

Replay saved qpos at25FPS (1x simulated physics time), two native cameras enlarged2x. Lower panels show the exact128point input cloud at its last saved replan, orange current GT cube wireframe for diagnosis only, and original SAM3 input RGB with its recorded predicted-mask overlay. The two mask thumbnails hold the actual input image until the next replan; masks are never applied to a different frame. Input frame/age is displayed explicitly. GT annotations are visualization only and were not policy inputs. A cloud is not recomputed between replans.

Save four MP4s, posters, configuration/source hashes, metadata and decoded-video/physical-pose verification under e133_fuse/videos. Non-target-lift evaluator labels do not require a bilateral grasp; the selected case has at most one jaw contact on the wrong object.

Run: `MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv/bin/python -u -m OATFlow.top_pick.sam3_video`. It rejects an existing output directory.

Videos verified:
[
  {
    "label": "heldout_success",
    "suite": "heldout",
    "record": {
      "path": "g000/p01/t0",
      "group": "composition",
      "layout": 0,
      "permutation": 1,
      "target": "RedCube",
      "slot": 0,
      "success": true,
      "reason": "success",
      "missing_detail": null,
      "frames": 124,
      "bilateral_lift_streak": 10,
      "wrong_object": null,
      "max_target_lift_mm": 111.33324903691377,
      "max_ik_error_mm": 0.06237386286457789,
      "observation_count": 9,
      "single_view_observations": 0,
      "dual_view_observations": 9
    },
    "video": "heldout_success.mp4",
    "frames": 125,
    "duration_s": 5.0,
    "source_hashes": {
      "/workspace/robotics_assets/experiments/e133_fuse/test_heldout/g000_p01_t0.npz": "a11d56a19003130da8deecc7e83594095862f98545f10a8abb7e594947b829b0",
      "/workspace/robotics_assets/experiments/e133_fuse/test_heldout/g000_p01_t0_observations.json": "c33d1c27019c08dc39d3a719f29d0f5feae3fa7e98b592d53f981ca45b06f347"
    },
    "replay_pose_max_error_m": 0.0,
    "video_fully_decoded": true
  },
  {
    "label": "ood_success",
    "suite": "ood",
    "record": {
      "path": "x_high_y_high/b2/t2_r0/outside",
      "group": "outside",
      "layout": 2302000,
      "permutation": 0,
      "target": "GreenCube",
      "slot": 2,
      "success": true,
      "reason": "success",
      "missing_detail": null,
      "frames": 140,
      "bilateral_lift_streak": 10,
      "wrong_object": null,
      "max_target_lift_mm": 117.96442871807432,
      "max_ik_error_mm": 0.120448427173075,
      "observation_count": 10,
      "single_view_observations": 5,
      "dual_view_observations": 5,
      "pair": 2302000,
      "edge": "x_high_y_high",
      "target_xy": [
        0.8526839476916176,
        -0.02384025221713803
      ],
      "cell": 23,
      "band_index": 2
    },
    "video": "ood_success.mp4",
    "frames": 141,
    "duration_s": 5.64,
    "source_hashes": {
      "/workspace/robotics_assets/experiments/e133_fuse/test_ood/x_high_y_high_b2_t2_r0_outside.npz": "933d223c41753793a6aebfe478fe53293c7890f16a064f90bc8a53010e4688ac",
      "/workspace/robotics_assets/experiments/e133_fuse/test_ood/x_high_y_high_b2_t2_r0_outside_observations.json": "d19f2b33004d04c4445bb614787ca4d8b2d8aeea6826ae86cd1d80cfbdeb4f5a"
    },
    "replay_pose_max_error_m": 0.0,
    "video_fully_decoded": true
  },
  {
    "label": "ood_timeout",
    "suite": "ood",
    "record": {
      "path": "x_low_y_high/b2/t1_r0/outside",
      "group": "outside",
      "layout": 1701000,
      "permutation": 0,
      "target": "BlueCube",
      "slot": 1,
      "success": false,
      "reason": "timeout",
      "missing_detail": null,
      "frames": 300,
      "bilateral_lift_streak": 0,
      "wrong_object": null,
      "max_target_lift_mm": 17.667699813268992,
      "max_ik_error_mm": 9.090145269074888,
      "observation_count": 20,
      "single_view_observations": 0,
      "dual_view_observations": 20,
      "pair": 1701000,
      "edge": "x_low_y_high",
      "target_xy": [
        0.22613853147661941,
        -0.029045691859147276
      ],
      "cell": 17,
      "band_index": 2
    },
    "video": "ood_timeout.mp4",
    "frames": 301,
    "duration_s": 12.04,
    "source_hashes": {
      "/workspace/robotics_assets/experiments/e133_fuse/test_ood/x_low_y_high_b2_t1_r0_outside.npz": "3b0392f06208272ab75377c37f57837f6cde4603f8de44db07afd6ba2f388a8f",
      "/workspace/robotics_assets/experiments/e133_fuse/test_ood/x_low_y_high_b2_t1_r0_outside_observations.json": "52ebfdfcb3e55f706b7a4ea541f2c031233bb475862190b4e05ff6b742ac9ec1"
    },
    "replay_pose_max_error_m": 0.0,
    "video_fully_decoded": true
  },
  {
    "label": "non_target_lift",
    "suite": "heldout",
    "record": {
      "path": "g028/p11/t3",
      "group": "composition",
      "layout": 28,
      "permutation": 11,
      "target": "YellowCube",
      "slot": 1,
      "success": false,
      "reason": "wrong_object",
      "missing_detail": null,
      "frames": 157,
      "bilateral_lift_streak": 0,
      "wrong_object": "GreenCube",
      "max_target_lift_mm": 2.933997628869842,
      "max_ik_error_mm": 0.03419136049035124,
      "observation_count": 11,
      "single_view_observations": 3,
      "dual_view_observations": 8
    },
    "video": "non_target_lift.mp4",
    "frames": 158,
    "duration_s": 6.32,
    "source_hashes": {
      "/workspace/robotics_assets/experiments/e133_fuse/test_heldout/g028_p11_t3.npz": "dcce2cfeda309caad0fdae0d2ebbd4e1822436c857b2188b5ba50fc65e674d5c",
      "/workspace/robotics_assets/experiments/e133_fuse/test_heldout/g028_p11_t3_observations.json": "0d9fc6bf029d96a1662b91e3cc359715a361ddb6de15505598a40c55e9bbc7d1"
    },
    "replay_pose_max_error_m": 0.0,
    "video_fully_decoded": true
  }
]
