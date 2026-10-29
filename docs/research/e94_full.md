# e94_full

Status: complete;10,000updates, best step10000.

Expand frozen e92_xyz to every e91/e55 episode using original layout-disjoint
splits:3,072train/384val/384test episodes,32/4/4layout groups. All24permutations,
all4query trajectories, and all4object labels per sampled RGB state retained.
Same temporal sampling as e92: every5frames plus final frame, giving
90,433train/11,513val/11,374test states. Original videos remain25Hz.
Full data means all episodes; validation/test are never used for training.

Same seed92, official ImageNet1K_V1 ViT-B/16 initialization and random XYZ head,
no resume from trained e92 weights. ViT fully frozen/eval,320px,64tokens/view,
192total. Same1,091,331parameter ID-conditioned fusion head. RGB+ID only;
measured center XYZ in workspace base frame metres, including lifted objects.
ImageNet RGB normalization; XYZ mean/std computed on full training split only.
No augmentation. AdamW3e-4, WD1e-4, batch128states/512object labels,
10,000updates,200warmup/cosine to0.1xLR, grad clip1; about14.15image epochs.
Frozen features cached float16 in host RAM (~31.12GiB all splits); only each
training/evaluation batch is copied to GPU. No feature dump to disk.
ViT encoding batch up to96images; pooling/normalization unchanged.

Select best full-validation mean XYZ error every500updates. Final test on all
4test layouts; additionally report the exact e92 val/test subsets for a matched
comparison. Metrics: mean/median/p90mm,1/2/5/10/20/50mm, perID, lifted/table,
train-mean-per-ID baseline and shuffled-image diagnostic. Four qualitative
prediction videos exported. No FM/action prediction, IK or policy rollout.

Artifacts `/workspace/checkpoints/e94_full`: config.json, split.json,
sample_index.json, normalization.json, frozen_check.json, train_metrics.jsonl,
metrics.jsonl, progress.json, best.pt/latest.pt, test.json, predictions.npz,
evaluation.png and test_videos/. All training, frozen-checksum and test/video checks passed.

Run: `PYTHONPYCACHEPREFIX=/tmp/e94_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u workspace_analysis/train_object_xyz_full.py`
tmuxhuy:e94; log workspace_analysis/logs/e94_full.log.
Code: train_object_xyz_full.py; shared object_xyz.py and train_object_xyz.py.

Preflight passed: CPU-cache and GPU-resident batches produce identical
predictions; frozen ViT has no gradients, head has gradients; 96-image encoding
works (~1.35GiB peak in preflight). Same random head initialization verified.

Full test mean5.552mm, median4.793mm. Shared e92 test mean5.127mm. Exact test.json.

User requested continuation to total40k updates (~56.62epochs): see e94_full_resume40k.md and /workspace/checkpoints/e94_full/resume40k/. Original10k results above preserved.
