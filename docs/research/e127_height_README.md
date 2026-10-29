# E127 rear-camera height comparison

Status: COMPLETE.

Compare fixed rear-camera heights 0.6, 0.8 and 1.0 m with identical XY
[0.54, 0.42] m, vertical FOV 42.65386164 degrees, native 320x240 resolution,
and look-at [0.54, -0.12, 0.035] m. Pitch changes to retain the same look-at.
Replay the same 138 saved expert states of E127 g000/p00/t0 at 25 Hz, with
no physics stepping, new actions, policy inference or training. Original E127
scene, calibration, states and metric-depth files remain unchanged.

Render RGB, visible instance masks and object-alone projected masks at each
height/frame. Visible fraction is visible pixels divided by the object's
unoccluded projected area. An object counts as visible at >=8 pixels; this
threshold does not imply useful depth or complete shape coverage.

| Height | Downward pitch | Frames with all four visible | Mean visible fraction | Mean unoccluded object area |
|---|---:|---:|---:|---:|
| 0.6 m | 46.30 degrees | 88/138 | 70.01% | 227.49 pixels |
| 0.8 m | 54.78 degrees | 70/138 | 83.23% | 158.96 pixels |
| 1.0 m | 60.77 degrees | 104/138 | 86.52% | 113.79 pixels |

Per-color visible-frame counts (Red, Blue, Green, Yellow):
0.6 m: (116, 110, 138, 138); 0.8 m: (70, 138, 138, 138);
1.0 m: (104, 138, 138, 138). No object leaves the field of view.
The 1.0 m view has the strongest overall visibility among these candidates,
but projected object area is approximately half the 0.6 m value. Raising the
camera does not remove gripper occlusion: the 0.8 m view hides the red grasp
target more frequently. This single episode does not establish dataset-wide
optimality or measure localization/policy improvement. Lateral placement has
not been tested in this comparison.

Run from the repository:
`MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.data_tools.rear_heights`.
Output: `/workspace/robotics_assets/experiments/e127_back/height_1`.
Artifacts: comparison.png (frames 0, 99, 137), frame_000.png, frame_099.png,
frame_137.png, height_comparison.mp4, scene.xml, visibility.npz, summary.json,
verification.json and config.json. Candidate views provide RGB and visibility;
no candidate-height metric-depth videos are generated or substituted.

Verification passed: all visible masks are subsets of object-alone masks,
source hashes are unchanged, and all 138 video frames decode successfully.
No checkpoints, optimizer, epochs, LR, H/K or flow settings apply.
