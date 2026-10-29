# e114: Cover identity from the target mask

Status: complete. All1408held-out episodes evaluated;4352total probes. This evaluation distinguishes correct cover identity from exact target-mask IoU. Model and weights unchanged.

Best TCOW target-mask checkpoint: train_3/best.pt, epoch9/step4608 after10epochs/
5120updates. Full2048train episodes, native TCOW backbone and pretrained target
patch head fully fine-tuned, batch4, backboneLR1e-5/headLR1e-4, seed114, BF16,
target-mask supervision only. Training configuration: ../train_3/config.json.
One top-camera visible mask prompt at frame0; only causal past/current RGB enters
later inference. No GT cover, target ID, XYZ or robot state is model input.

All held-out episodes:1024unseen-permutation +256unseen-layout +128both-unseen,
1408total/352shared contexts. No training/validation cases included. Evaluate the
final context frame after shuffle and midpoint of each shuffle run. This does
not evaluate every hidden frame and is not a closed-loop manipulation test.
The previous160episode subset is identified separately for direct comparison.

Prediction is a binary target mask at probability0.5. Assign it to the unique
cover with greatest overlap with that cover's GT amodal silhouette, requiring
strictly more than50%of predicted-mask area inside that cover. Empty masks,
ties, and insufficient support count as unresolved, not correct. There is no
nearest-cover fallback. GT silhouettes are evaluation geometry only. Ground
truth identity is the stored physical container ID; after shuffle independently
verify it against the episode's selected_cover metadata. Apply the same cover
assignment to the GT object mask as an oracle check, since overlapping cover
projections may be ambiguous during motion.

Separately report which of four GT object silhouettes maximizes predicted-mask
IoU. This is a spatial object-binding diagnostic and distinct from cover accuracy.
No overlap or ties count as unresolved object binding. Neither metric establishes
an FM cover-selection policy or successful grasping.

Inferences use the same stride3cached context frames and30frame causal prefix
sampling as the prior test, batch4, BF16, eval/inference mode. Reproduce all160
previous post-shuffle target IoUs and require maxdifference<1e-7. Checkpoint file
hash unchanged after inference. H/K, flow steps, noise seed: not applicable.

Run from repo root:

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python -m OATFlow.tracking.mask_cover --output /workspace/robotics_assets/experiments/e114_mask/test_cover
```

Output: config.json, records.json, summary.json, progress.json, and packed binary
predictions.npz. Context RGB/masks are cached in RAM only, no persisted features.

## Results

| Group | Post-shuffle correct cover | Shuffle-midpoint correct cover |
|---|---:|---:|
| Unseen permutation |1024/1024=100%|2142/2144=99.91%|
| Unseen layout |256/256=100%|544/544=100%|
| Both unseen |128/128=100%|256/256=100%|
| Total |1408/1408=100%|2942/2944=99.93%|

The previous160episode subset is160/160correct at post-shuffle. Spatial binding
to the correct one of four GT object silhouettes is also1408/1408post-shuffle,
but2936/2944during sampled shuffle midpoints. This is not pixel-perfect geometry.

Two wrong-cover probes: g025/p08/t0(RedCube) and g025/p08/t3(YellowCube), both at
raw frame294/11.76seconds in the same shared shuffle context. True physical
container is cover_b(ID6), while the predicted masks lie entirely inside
cover_a(ID5), with zero target-mask IoU at that instant. Both predictions recover
the correct cover by the final context frame. They are actual tracker errors,
not ambiguous cover projections: the same assignment applied to GT object masks
is correct at4352/4352probes. No empty/ambiguous prediction was counted correct;
there were zero unresolved predictions in this test.

Episode-level:1406/1408episodes have correct cover at every tested probe; the two
exceptions share one context. This tests fixed probes, not every hidden frame.
Thus100%post-shuffle cover assignment is supported on this held-out dataset;
100%continuous cover tracking is false. No closed-loop cover-opening/grasp test
was run. Covers are associated using GT silhouettes for evaluation, not a learned
cover-selection head.

Checkpoint hash unchanged. Prior160post-shuffle target IoUs reproduced within
2.964e-8, consistent with FP32-vs-FP64 metric arithmetic. Detailed records and
packed masks are retained; raw RGB/features are not persisted by this test.
