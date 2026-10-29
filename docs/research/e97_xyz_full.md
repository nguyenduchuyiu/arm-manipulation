# e97_xyz/full

Status: complete. All 1500 full-finetune updates and opening-selected tests finished.
User requested full finetuning after the partial-backbone pilot.
Warm-start model weights from /workspace/checkpoints/e97_xyz/latest.pt, saved
partial-stage step1000; fresh AdamW for the changed full-finetune parameter set.
All125,313,539model parameters trainable, including patch/position/time embeddings,
all12backbone blocks, fusion, target/view embeddings and XYZ head. No freeze stage.

Additional1500full-finetune optimizer updates, logical batch2/micro1,BF16,
new modules LR3e-4,backbone3e-5,50update warmup/cosine floor0.1x,WD1e-4,
clip1,seed97. XYZ-only loss:0.5endpoint +0.5all30sampled causal prefix frames.
Train-only XYZ mean/std with10mm std floor. Same e96 subset as pilot:512train,
32val,64composition/32spatial/64combined held-out test episodes. RGB retained
every3source frames in host RAM; clip includes frame0 and no future observations.
Target ID and timestamps only, no GT masks/centers/cover states in model inputs.

Validation every250updates originally recorded all phases and selected hidden error.
Per user correction, best.pt is now step1250: minimum exposed mean over all six
validation checkpoints, 106.232727mm at t_obj (cover opened, before grasp).
Correct held-out evaluation is in test_open; interrupted original hidden-selected
test produced no results. Original training source is archived in source_train.py
and original config.json is retained. Current trainer selects exposed-only val.
Test all three held-out groups only after fit. Output /workspace/checkpoints/e97_xyz/full:
README,config,subset,normalization,progress,metrics,best/latest,test and predictions.
Best contains weights only; latest includes optimizer for recovery. No duplicate
final checkpoint. No e88 rollout/action training or calibration/depth input.

Tmux huy:e97_xyz; stdout/stderr tee workspace_analysis/logs/e97_xyz.log.
Earlier partial run stopped at~1090updates (latest saved1000); no test results.

Opening mean error: val 106.233mm, composition 101.168mm, spatial 107.162mm, combined 107.133mm. All groups have0% within20mm.
The model predicts near a constant center; this run did not learn useful localization.
Trained-checkpoint future pixel gradients and future RGB/timestamp perturbations
are exactly zero on the tested prefix. Test metrics/arrays/examples in test_open/.
