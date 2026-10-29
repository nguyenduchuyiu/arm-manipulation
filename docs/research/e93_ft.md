# e93_ft

Status: stopped by user; latest checkpoint step2000, best step1500. Final test not run.

Controlled comparison with e92_xyz: enable training of the entire ViT-B/16,
including CLS/position embeddings, patch projection, all12blocks and final norm.
Same ImageNet1K_V1 initialization, seed92 and initial head; do not resume from
e92's trained head. Same3RGB320 views,64tokens/view/192total, width192 two-block
ID-query fusion/XYZ head. InputRGB+ID only; actual base XYZ metres, all4IDs.

Same e91 subset and exact e92 sample_index:768train/192val/192test episodes,
22,475/5,730/5,674sampled states,8/2/2independent layouts, all24permutations.
Same ImageNet RGB normalization and train-only XYZ mean/std, verified exactly.
No augmentation. Same AdamW LR3e-4 for BOTH backbone and head, WD1e-4,
200update warmup/cosine to0.1xLR, global grad clip1,10,000updates,
effective batch128image states/512object labels; microbatch32, accumulate4.
AMPbfloat16. Features castfloat16 before head as in e92; recompute every batch.
Selected RGB states cached in host RAM (~31.2GB); no stale feature cache.
Validation every500updates on the identical full validation subset, select
best mean Euclidean XYZ error. Effective56.95image epochs.

Artifacts `/workspace/checkpoints/e93_ft`: config.json, subset.json,
sample_index.json, normalization.json, gradient_check.json, progress.json,
train_metrics.jsonl, metrics.jsonl, best.pt/latest.pt (backbone+head+optimizer+RNG).
Final test.json, predictions.npz, evaluation.png and4qualitative videos planned.
No FM/actions, policy rollouts, IK, horizon or flow/noise settings apply.
Train/test status and actual results will be updated after launch/completion.

Run: `PYTHONPYCACHEPREFIX=/tmp/e93_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u workspace_analysis/train_object_xyz_ft.py`
Resume the same run with `--resume`. tmuxhuy:e93; log workspace_analysis/logs/e93_ft.log.
Code: workspace_analysis/train_object_xyz_ft.py; shared object_xyz.py;
metrics/videos reused from train_object_xyz.py. Frozen e92 remains reproducible.

Checks passed: same initial head as frozen-construction seed92; all trainable ViT
parameters; full effective-batch accumulation preflight2updates (~14.01GiB peak);
actual first-update nonzero patch/position/last-block/head gradients and changed
backbone SHA256 (gradient_check.json); exact e92 frames and XYZ normalization.
Checkpoint reload/inference passes for both finetuned e93 and original frozen e92.

Trainable:85,955,328backbone +1,091,331head =87,046,659parameters.
Observed ~1.26s/update after warmup, ~4hours including validation overhead.
No validation/test accuracy yet; first full validation at step500.

User switched to full-data frozen ViT experiment e94_full. Checkpoints preserved.
