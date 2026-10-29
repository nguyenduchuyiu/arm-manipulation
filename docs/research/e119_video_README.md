# E119 K10 first completed case video

Status: COMPLETE;301frames fully decoded, physical trace unchanged. Render first completed test case g000/p01/t0, RedCube,
unseen color permutation on a training layout. Selection is file order,
independent of success; observed result timeout. Uses test_k10 final checkpoint
step2940 after exact10epochs/256demo subset, not best1250. No policy/physics rerun.
Training frozenViT + target-query decoder + auxiliary128point Chamfer(weight1)
+ FM initialized from E118120k; all visual/FM except ViT train. Details:../README.md
and ../config.json. Actual test:H25/K10/Euler10/noise0/300frames,25/50/500Hz.

Render saved initialqpos+300qposstates. Left native front RGB enlarged2x, right
oblique actual robot. OrangeGTtarget center, cyan auxiliary predicted point cloud,
green measured EE path. Recompute auxiliary cloud only from current native RGB
and targetID at recorded replan states, hold between; not fed back into FM.
GTcenters/contacts/lifts are overlays only. Full test continues independently.

Output:rollout.mp4,poster.png,auxiliary_clouds.npz.25FPS,1344x700,301frames,
including initialstate. Exact checkpoint/trace SHA and metrics:config.json;
full video decode/framecount and unchanged physical trace:verification.json.
Code:OATFlow/top_pick/cloud_video.py;log:e119_pick/video_1.log,tmux:e119_video.
