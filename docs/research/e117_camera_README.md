# E117 wide-pick camera samples

Status: COMPLETE. Four rendered expert cases,580frames at25FPS (23.2s); video
fully decoded. All four targets are visible initially in both cameras in every case. User requested a few wide-random pick samples with the same
camera configuration as E116. Reuse accepted E117 layouts0/36, permutations0/1,
and select four different target identities. No new layout search, full dataset
collection, learned policy, tracker or training. Training weights/epochs/batch/LR,
H/K/flow/noise seed not applicable. Regenerate only four deterministic successful
contact-expert trajectories from the retained accepted layouts, using E117's
unchanged home-start scene setup and pick-only oracle.

Render camera placement/intrinsics copied from current E116 scene: front-oblique
RGB-D320x240, front depth2mm/unit (invalid0), wrist RGB synthetic equidistant175deg
diagonal fisheye with the same mounting. No top/left/right and no wrist depth.
Raw images/calibration retain native orientation; composite viewer rotates wrist
RGB180deg for readability. Geometric clipping1mm/10m. Fisheye lens is synthetic,
not measured calibration of a real Arducam or the user's unidentified camera.
Both old and current physics scenes have identical robot/object/contact parameters;
only camera definitions are replaced. Export per-frame camera poses and K/models.

Outputs planned:e117_wide/camera_samples/samples.mp4 showing four pick episodes,
individual initial/grasp sample PNGs, contact sheet, calibration_samples.npz,
2mm front-depth PNGs, config.json and summary.json. RGB/depth are ideal simulation
observations; no supervised coordinates enter an inference policy here.
Verify previous deterministic episode frame counts and successful stable bilateral
lifts, camera placement/FOV equality, frame counts, view/depth schema and depth
quantization; decode the complete resulting video. Keep requested visual samples
and small calibration/metrics; temporary orchestration is removed after completion.

Actual selected cases:g000/p00/t0 Red138frames,g000/p01/t2 Green153frames,
g036/p00/t1 Blue139frames,g036/p01/t3 Yellow150frames. Recreated episode lengths
and stable bilateral hold metrics exactly match retained E117 pilot records:
25/19/23/20frames respectively. XML camera definitions copied exactly from E116;
robot/object inertia, geometry, friction, joints and actuator arrays unchanged.
Front and wrist calibrated per frame; metadata580x2 transforms and camera types
verified. Depth quantization maximum1.000065mm, readable uint16 front-depth PNGs.
Initial visible target pixel counts:front257..379,wrist159..239 over all four objects.

samples.mp4 contains all four episodes consecutively;initial_samples.png contains
four starting camera/depth rows. sample_XX_000.png are individual initial views;
close/end PNGs and depth_front_XX_*.png retained. calibration_samples.npz records
native unrotated camera coordinates; sample_id groups per-case timestamp_s, which
resets to0 for each episode. All view rendering is ideal simulation; synthetic lens
and geometric depth do not imply real sensor calibration or noise/focus behavior.
Temporary four-case orchestration was removed after media/schema verification;
retained scene/layout paths and permanent setup/expert/camera helpers specify it.
No full E117 dataset or training was launched.
