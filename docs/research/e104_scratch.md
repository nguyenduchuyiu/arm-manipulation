# e104_scratch

Status: complete;preflight,training and paired K10 checks passed. User-requested ablation of e103
with no learned FM initialization. Same10epochs and final-step K10 comparison.

Instantiate fresh ReachPolicy/FlowMatchingHead with seed102 and retain every
random FM parameter,including proprio/state/context/action/time projections,
16transformer layers and output head. Load only normalization buffers from the
reach source (state/action mean/std plus unused inherited XYZ buffers). The
frozen ImageNet ViT-B/16,visual projection/target/position/decoder/goal and
center-head initial tensors are exactly equal to fresh e103. CUDA/CPU training
RNG is also exact. No e103 trained weights or optimizer state initialize e104.

Preflight:all185FM parameter tensors exactly match the fresh seed102 model and
all185differ from pretrained reach. All non-FM initial tensors and normalization
buffers match e103. Same first128training IDs hash,noise/time RNG and center
loss1.1697862148/center error116.4693mm. Matched e103 first FM loss1.3317035437
exactly reproduced;random FM first loss1.7664154768. Preflight uses autograd-enabled
training forward,as the no-grad kernel gives a small numerical difference.

Reuse unchanged e102_top/cache and e103_center/centers.npy/train-only statistics.
Seed102,285600train chunks,2232updates per full pass (last batch32),batch128,
10epochs/22320updates/2856000samples. FM LR2e-5/visual1e-4,100step warmup then
constant,AdamW(.9,.95),wd1e-4,clip1,BF16. Same hidden256 normalized current-target
XYZ MSE weight1. Same H25x10D anchor EE deltaXYZ,relative rotation6D,physical jaw.
Same512fixed validation chunks/noise/time,validation step1/every250/final.
Best by action FM loss retained separately. Primary test uses final22320.

Exact comparison:e103 final22320/test_k10_2 gives118/128(32/32train,32/32unseen
order,26/32new positions,28/32both new). e104 uses same128cases/order/states,
H25/K10,Euler10,seed0,300frames,25Hzpolicy/50Hzservo,PoseIK40nfev,no temporal
ensemble. Success correct target>4cm,bilateral jaws10frames;wrong lift fails.

Artifacts /workspace/robotics_assets/experiments/e104_scratch. Actual train/config.json,
preflight.json,source snapshot and logs. Code:model.py/center.py add independent
fm_pretrained flag;center_train.py accepts --scratch-fm --epochs10 --output.
Default initialized behavior retained;load_policy checkpoint loading unchanged.
run.sh in tmux e104 runs train then K10 test with tee logs. Existing e103 outputs
are retained. This ablation holds hyperparameters fixed;it does not tune LR or
training duration specifically for random initialization.

Completed22320updates/2856000samples/exact10epochs in3235.54s. All training
updates finite,each epoch285600samples with final32-sample batch. Final22320
optimizer states/epoch cursor verified;latest/final weights identical. All150ViT
tensors and eight normalization buffers unchanged;185random FM tensors trained.
Final sample permutation,cursor and CPU/CUDA/NumPy RNG states match e103 exactly,
confirming unchanged data/noise/time sampling across the full10epochs despite the
e103 pause/resume at3000. Source hashes verified. train/verification.json.

Best fixed validation FM0.01000532 at14750,retained separately,not tested.
Final22320 FM validation0.0130700,center error7.5500mm. e103 final22320
FM validation0.0115230,center error7.6102mm. These mixed-frame diagnostic losses
are not physical success metrics or a unique explanation of rollout differences.

Same128learned K10cases:e104 train21/32,unseen order21/32,new positions21/32,
both new17/32,total80/128(62.5%). e10332/32,32/32,26/32,28/32,total118/128
(92.1875%). Paired71both succeed,9random-only successes,47pretrained-only
successes,1both fail. New48timeouts,zero wrong lifts. Red24/32,blue23/32,
green20/32,yellow13/32. Exact128case order/target/slot and384initial state arrays
match. Test234.856s. With this matched seed,budget and learning rates,reach FM
pretraining improves success by29.6875percentage points. Random FM still learns;
the run does not test a separately tuned scratch recipe or larger budgets.

Artifacts train/config.json,metrics.jsonl,validation.jsonl,validation.png,
verification.json;test_k10/config.json,summary.json,results.jsonl,comparison.json,
comparison.png and comparison.mp4. Seven fixed previous red-target examples,
left pretrained,right random,both final22320.69s,1280x880,25fps;last frame held
for shorter source,original playback speed. Exact sources/outcomes/media checks
in video_manifest.json. Examples are illustrative;full128cases form aggregate
metrics. Native videos sample every3policy frames plus terminal frame,repeat
each image3times at25fps;outcome times use physical steps rather than padded
media length. Train/test main pane moved to ssh_tmux:1 at user request;it retains
the final output after successful process exit. No new conditioning probe run.
