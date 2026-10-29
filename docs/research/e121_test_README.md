# E121 — Full held-out evaluation with400spatial patches

Status: planned, not run.

Compared with E120, remove20×20→8×8 spatial pooling from both RGB ViT features
and XYZ patch features. Keep400native positions through channel fusion and query
decoder. FM sees the shared target latent plus measured state, not raw cloud.
Frozen ImageNet ViT; train XYZ patch/fusion/query decoder/projections/aux cloud
head and complete FM initialized from E118reach120k. No learned E120 weights.

Training: E117 part1/8,256demos/37,536stride1chunks,80epochs/23,520updates,B128,
seed102,FM LR2e-5/other LR1e-4,warmup100,AdamW(.9,.95),wd1e-4,clip1,BF16,
FM + weight1 target-cloud Chamfer. H25 anchor-relative EE deltaXYZ,rotation6D,jaw.

Test immutable final exact80epoch snapshot, all3,200held-out episodes:
spatial2,048,composition128,combined1,024. H25/K10,Euler10,noise0,300frames,
policy25Hz/servo50Hz/physics500Hz. Inputs full front RGB/depth/calibration,
target ID and measured joints. No GT mask/center in policy observations.
Correct-object lift>4cm with bilateral contact for10consecutive frames;
wrong-object lift fails. Save physical traces, no video rendering here.

Checkpoint SHA/protocol: `config.json`; outcomes:`results.jsonl`;
partial aggregates:`progress.json`; final:`summary.json`.
