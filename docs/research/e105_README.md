# e105_tcow

Status: starting corrected full one-epoch training in train_2. Earlier train was
interrupted after1update on user request; it is not resumed or evaluated.

Task: original top-only shuffle history, then learned cover opening/deposit and
target grasp/lift. Official original TCOW RGB+query backbone and reach FM115000;
new pooling/hidden256/XYZ head/goal960. Full finetune when expert actions exist.
No masks, target ID or GT XYZ policy inputs; initial radius3UVdot only.

Within each shuffled shared context: chronological context endpoints stride10,
then chronological expert endpoints stride3. Context batches train TCOW/XYZ only:
action mask all false, FM and goal forward/grad/AdamW updates skipped. Expert
batches train end-to-end L_FM +1*L_XYZ. XYZ supervision covers all30 observed frames.
No XYZ prediction is fed directly to FM. Four target queries are paired in time.
RGB cache is context union(stride3,stride10,terminal), expert stride3+terminal;
30past/current frames sampled uniformly in raw time, nearest past cached frame.
Crop top320square rows40:280 to240x320. No feature cache.

2048train episodes/512shared contexts:61688context clips +234434expert clips =
296122clips per epoch. Batch4 directly (small terminal batches), no accumulation,
no gradient checkpointing.15422context +58808expert =74230updates per epoch.
BF16, AdamW(.9,.95),wd1e-4,clip1; warmup100 then constant TCOW1e-5/FM2e-5/new1e-4.
Reach action/state statistics preserved; XYZ fitted only on train,10mm std floor.
H25 measured fixed-anchor EE deltaXYZ3 + relative rotation6D6 + absolute jaw1,
25Hz, padding masked. Exact settings/hashes: train_2/config.json and schedule.json.

24-update preflight6 passed. Median batch times context0.880s/expert0.967s;
estimated one epoch20–22hours including overhead; peak allocated VRAM19.88GiB.
Timing: epoch_estimate.json. Verified bit-exact FM/goal weights after context
optimizer steps, including after expert momentum exists. Nine retained point
tracker tests pass. Controller-only measured-EE oracle4/4, not a learned result.

Validation32episodes/64fixed expert chunks, noise/time seed105000; best selected
by FM loss. Planned test uses train_2/final.pt,128cases (32each train/composition/
spatial/combined), H25/K10/Euler10/seed0/900frames,25Hzpolicy/50Hzservo/500Hzphysics.
Recorded post-shuffle initial state; all subsequent cover/pick actions learned.
No expert opening, home reset or separate IK lift. Full success: correct cover
deposited, target>4cm bilateral lift10frames, other cover shift<=15mm; wrong cube
lift fails. Test not yet run. Live train/test output: ssh_tmux window1 and logs/.
