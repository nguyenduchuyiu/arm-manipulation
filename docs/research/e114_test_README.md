# e114: Target-mask tests and video comparison

Status: complete. Five individual videos and comparison.mp4 fully decoded,2948total frames/117.92seconds at25FPS. Best checkpoint unchanged; validation replay difference exactly0.

Checkpoint: ../train_3/best.pt, selected at epoch9/step4608 after10totalepochs
(5120updates). Full2048train episodes, one causal30frame clip per episode per
epoch; top camera crop240x320, frame0 visible mask prompt. All TCOW backbone and
native target-mask readout parameters fine-tuned. Original occluder/container
channels and unused flag head removed. Original pretrained backbone AND target
readout initialization, seed114, AdamW backboneLR1e-5/headLR1e-4,50update warmup,
weight decay1e-4, batch4, BF16, no gradient checkpointing. Target-only original
TCOW segmentation loss. Exact training settings: ../train_3/config.json.

The completed trainer evaluated best on64unseen-permutation,32unseen-layout,
64both-unseen episodes at fixed endpoint probes (not entire held-out splits).
Their metrics are ../train_3/test_{composition,spatial,combined}.json. Best
validation is independently replayed on exactly the same32episodes/probes here
and compared with the stored result before producing videos.

Five full recorded expert videos: deterministic first eligible train/red,
unseen-permutation/blue, unseen-layout/green, both-unseen/yellow; plus the lowest
mean hidden-probe IoU episode among the64evaluated combined-test episodes.
Case paths and selection rules are in config.json. These illustrative videos
are not a random sample or an estimate of grasp success.

Each video compares GT amodal target mask (green), original pretrained TCOW
(orange), and fine-tuned TCOW (magenta). One visible mask query at frame0 only;
later GT masks are labels/metrics/display only. Both models see identical causal
30frame past/current RGB clips, with frame0 retained. Predictions at cached
stride3endpoints plus segment terminal frames;25FPS full raw-frame playback,
holding the most recent prediction without future interpolation. Display zoom
is centered on GT for inspection and never enters either model.

Output1536x864H264MP4, threshold0.5. Exact dense causal-hold meanIoU is recorded
separately from the trainer's sparse endpoint-probe metrics. Every individual
MP4 and the concatenated comparison.mp4 will be fully decoded and counted.
best.pt file hash is checked unchanged after inference. No fine-tuning during
test. No FM/control/IK rollout: all arm motion is the recorded expert. H/K,
flow steps and action noise seed are not applicable.

Run from repo root:

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python -m OATFlow.tracking.mask_video --output /workspace/robotics_assets/experiments/e114_mask/test_videos
```

Artifacts: config.json, validation_replay.json, summary.json, five individual
MP4s, comparison.mp4, prompt/shuffle/exposed PNGs, and packed mask predictions
with raw frame indices/GT/IoU in NPZ files. No persistent feature cache.

## Dense video results

These values average every raw frame with causal stride3prediction hold; they
are distinct from the160episode fixed-probe test results.

| Video | Episode | Original meanIoU | Fine-tuned meanIoU | Original hiddenIoU | Fine-tuned hiddenIoU |
|---|---|---:|---:|---:|---:|
|train|g000/p00/t0|36.62%|82.97%|9.47%|78.45%|
|unseen_permutation|g000/p01/t1|38.13%|87.64%|8.80%|80.85%|
|unseen_layout|g036/p00/t2|25.49%|58.42%|8.16%|37.80%|
|both_unseen|g036/p01/t3|31.80%|69.70%|8.26%|57.74%|
|hard_case|g036/p21/t2|26.18%|58.90%|8.42%|38.38%|

Qualitative inspection: the fixed unseen-layout case and the deliberately hard
combined case retain a target-mask offset under the cover (post-shuffle frame
IoU about0.30), then localize the visible target more accurately after opening.
The mask model improves on original weights but does not solve every occlusion.
All robot motion remains recorded expert motion, not learned action control.
