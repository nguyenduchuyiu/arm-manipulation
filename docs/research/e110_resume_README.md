# e110_trackon / train_2 — resume to 100 total epochs

Status: preparing resume. Source is ../train/latest.pt, epoch10/step1060/1,347,260samples. Preserve its learned projection/context/XYZ weights, complete AdamW moments and step counters, NumPy/Torch/CUDA RNG, XYZ normalization, best validation threshold and all frozen sources. Main target:100total epochs, so this run adds90epochs/9,540updates and ends at10,600updates/13,472,600samples. All existing train artifacts stay intact.

Unchanged: full2,048query episodes,134,726stride10endpoints per epoch,32validation episodes/184fixed phase points, batch1280, four66token/960Dcontext blocks, XYZ-only normalized MSE weight1. Frozen latest official Track-On-R/DINOv3, ImageNet ViT-B/16 on static cam-left (temporary wrist substitute), and entire reach-trained FM/state encoder. FM source step115000,10D EE-delta/rotation6D/jaw normalization preserved; no action training, mask head or manipulation rollout. All gradients update only visual projections/view embeddings/context decoder/XYZ head. No accumulation or gradient checkpointing.

Reuse ../train/query_cache NPZ features and exact raw-frame indices; do not rerun Track-On. Reconstruct only current cam-left ViT features in CPU RAM. All frozen source hashes and model/vision source hashes must match the original run. Verify exact row identities, query shapes/indices, finite features and all GT XYZ rows against saved train/val arrays. Store compact source-reference symlinks rather than duplicate query caches. CPU feature/state/label/index storage8.17GiB; restart rebuilds side RAM cache. This run retains no dense feature cache on disk.

AdamW(.9,.95),weight_decay1e-4,clip1. The original10epoch cosine reached its LR floor; resume keeps those stored rates constant: decoder/projections1e-5, XYZ head1e-6. No warmup restart or LR reset. BF16 context autocast, FP32weights/loss, TF32allowed. Best checkpoint is selected over both original and resumed epochs by equal-weight macro mean of five validation phase errors. Save learned context/normalization plus optimizer/RNG and frozen-source references, not frozen model duplicates.

Before first update: match XYZ normalization and frozen hashes, restore every optimizer state atstep1060, restore NumPy/Torch/CUDA RNG, and reproduce source final validation mean38.1657mm to within0.05mm. First resumed backward must have finite nonzero gradients for all trainable tensors and no FM gradients. Log loss/mm/LR/ETA every10steps and validate each epoch. No separate persistent preflight folder. Training-only estimate~2hours for90additional epochs; report measured cache time separately.

Run from /workspace/arm-manipulation:

```bash
.venv/bin/python -u -m OATFlow.shuffle_bc.train_trackon_xyz --resume /workspace/robotics_assets/experiments/e110_trackon/train/latest.pt --epochs 100 --batch 1280 --output /workspace/robotics_assets/experiments/e110_trackon/train_2
```

Trainer creates a fresh output and copies this README. Runtime source snapshots/config.json document exact code and settings. Run in ssh_tmux:1 with stdout/stderr through tee into experiment train_2.log. Results pending; source validation is baseline only. Separate held-out evaluation and action rollout not run; H/K/noise/frame budget not applicable in this XYZ stage.

Independent resume verification passed on4real frame0queries: every AdamW exp_avg/exp_avg_sq matches source checkpoint bit-exact, all49steps restored at1060and advanced to1061after one finite-gradient update, frozen FM gradients absent. Check executed in a separate process and saved no model weights; original checkpoint unchanged. Exact check: optimizer_check.json / docs/research/e110_resume_check.json. Main remains at side-cache reconstruction; its own initial validation parity/resume_checks.json are required before updates.
