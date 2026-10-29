# e97_xyz

Status: interrupted by user change to full finetuning; latest saved step1000. Native three-camera amodal XYZ subset training/test; no mask
or action outputs/losses. e96 audited data, RGB top/left/right,320px full frames
resized240x320,25Hz source retained every3frames in RAM. Each input is30sampled
past/current frames including frame0, target ID and actual timestamps. No GT
centers, masks, cover assignment, future frames or joint state as model inputs.

Train8layouts0..7 x16seen permutations x4queries=512episodes. Val layouts32/33
x4seen permutations[0,9,16,18]x4queries=32episodes. Tests: composition layouts0/1
x8unseen permutations x4=64; spatial layouts36/37 x4seen permutations x4=32;
combined layouts36/37 x8unseen permutations x4=64. Manifest paths in subset.json.
Checkpoint selection uses val hidden mean XYZ error; test is used only after fit.

Native TimeSformer-like width768/12blocks, fusion after4/8/12,width256,8context
plus1target token. Initialize pretrained native RGB backbone tensor artifact
/workspace/checkpoints/vision_init/tracking_backbone.pt, fresh fusion/XYZ head.
First500updates freeze backbone; then tune last4blocks.1500updates, logical
batch2/micro1,AdamW,WD1e-4,gradclip1,BF16. New modules LR3e-4,backbone1e-5,
50update warmup/cosine floor0.1x. Seed97. Loss0.5endpoint MSE +0.5allprefix MSE
on train-normalized XYZ, std floor10mm. Phase sampling visible15%,moving
hidden35%,postshuffle20%,expert30%. Context frames supervise localization.

Validate every250updates at visible, each swap midpoint, postshuffle, exposure
and lift endpoints. Best/latest checkpoints only. Final all three test groups
report mean/median/p90 and1/2/5/10/20/50mm fractions, hidden/visible and phases.
No e88 action rollout. No H/K/action/flow/noise execution settings apply.

Output /workspace/checkpoints/e97_xyz; actual config.json, subset.json,
normalization.json,progress.json,metrics.jsonl,best.pt/latest.pt,test.json,
per-group predictions.npz and sample lists. RGB cache in host RAM only, shared
contexts decoded once. Fresh output required. No calibration/depth required.

Run:python -m OATFlow.tracking.train --steps 1500 --output /workspace/checkpoints/e97_xyz
Tmux huy:e97_xyz; log workspace_analysis/logs/e97_xyz.log (stdout/stderr tee).

Full finetuning continues in e97_xyz/full; see e97_xyz_full.md. No partial-stage tests run.
