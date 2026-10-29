# E131 — replay the single non-target-lift failure

Status: COMPLETE.

Selected by reason==wrong_object from the completed1408case K15test; exactly
one case:g028/p04/t3,composition,target YellowCube,non-target BlueCube.
E131final23520: full2048E117train demos,stride1,B128,23520updates,all point
encoder/goal/state/FM parameters finetuned from E118best120000 with BC only.
Original optimizer/LR/input/action details remain in ../config.json and are
copied into render config.json. No training/model inference in this replay.

Replay exact saved initial+143qpos states (144images), with saved10input
clouds at actual K15replans0..135. Original H25/K15/Euler10/seed0/300frame
limit,25Hzpolicy/50Hzservo/500Hzphysics. No physics stepping,new actions or
resampled clouds. Video is10FPS,0.4x original speed; original timestamps are
shown. A cyan cloud is the recorded GT-selected target INPUT held from the
last replan, not a predicted cloud. Panel labels show its source frame/age.

Render front RGB with target/non-target labels and saved input-cloud overlay,
plus an oblique physical view and measured EE path. Show actual object lift
and jaw contacts. The evaluator's wrong_object means any non-target rises
>4cm; it does NOT require bilateral contact or a stable wrong-object grasp.
Verify those distinctions using saved contacts and actual rendered motion.

Output:e131_object/test_k15/wrong_1. Main files:wrong_object.mp4,
start.png,contact.png,last.png,contact_sheet.png,metrics.json,summary.json,
verification.json,config.json. Run `MUJOCO_GL=egl .venv/bin/python -u -m
OATFlow.top_pick.object_video` in tmux with tee. Verify source hashes, all
frames decode, replayed object poses agree with stored positions, all saved
clouds coincide with target box geometry at their capture frames, and exact
cloud cadence. No checkpoint is modified. Results below.


Results:144frames decoded,14.4s at10FPS (original motion5.72s). Replay object
poses match saved positions within1e-6m; source trace/results/checkpoint hashes
are unchanged. Every recorded input cloud remains on Yellow geometry, with
maximum point-to-surface distance1.395mm. No target-cloud selection error is
observed. Its centroid stays at least53.09mm from the Blue body center.

Blue rises41.33mm and triggers wrong_object. It never has bilateral jaw
contact (maximum1/2;zero bilateral frames), and has zero jaw contacts in the
terminal frame. Collision inspection of saved states shows Blue contacting
the right jaw and gripper base near the end, then no contacts at frame143.
The evidence supports collision-induced non-target lifting/release, rather
than a stable two-finger grasp of the wrong semantic target. Yellow remains
on the table. Benchmark outcomes are preserved unchanged; the evaluator
counts any non-target lift above40mm as failure.
