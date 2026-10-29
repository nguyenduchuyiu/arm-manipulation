# E119 final-checkpoint full held-out pick evaluation

Status: planned. Run after exact10epoch training. Use immutable final checkpoint,
not best chosen from test. Training:subset4layouts/256demos,37,536stride1chunks,
10epochs/2940updates,B128; frozen ImageNet ViT, target query/goal/cloud head and
E118120k-initialized FM train, FM LR2e-5/visual1e-4,warmup100,cloud Chamfer weight1.
Full architecture/config:../README.md,../train/config.json,config.json.

Test ALL3200held-out episodes:2048new-layout/seen-permutation,128seen-layout/new-
permutation,1024both-new. H25/K10,Euler10,noise0,300policy frames,25/50/500Hz.
Policy sees native front RGB letterboxed to320square,targetID,measured joints;
no depth/masks/GTcloud/XYZ. Same camera/ViT preprocessing as training.
Correct target lifted>4cm with bilateral jaw contacts for10consecutive frames;
wrong-object lift>4cm fails. Learned grasp success, separate from reach precision.
No videos in full evaluation. Save per-case scores, physical qpos,commands,objects,
contacts and initial states for future video/debugging. No test-based selection.

Artifacts:e119_pick/test_k10. Log:e119_pick/test_k10.log,live ssh_tmux:e119.
Source:OATFlow/top_pick/cloud_evaluate.py. Supports --resume after interruption,
keeping completed results and the fixed checkpoint. Actual checkpoint SHA/step,
manifest hash and controller settings:config.json. Live totals:progress.json;
full results:results.jsonl; final:summary.json. No result yet.
