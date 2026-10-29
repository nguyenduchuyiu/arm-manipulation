# e97_xyz/test_debug/binding

Status: complete. Native causal architecture with ImageNet ViT-B/16 spatial
weights, cloned temporal attention and zero temporal output residuals. RGB
ImageNet normalization, no history gates or extra positional-memory adapter.
All125,313,539parameters trainable, full finetune throughout. Native modules
only; official local RGB weight artifact, no third-party tracker runtime.

Approved e97 subset512train/32val. Stage1:300updates, batch16 as4shared
contexts x4IDs on the same first reveal RGB, all4targets visible. Stage2:
300updates, batch16 mixing2shared reveal contexts x4IDs with8opening examples
(2per ID). This removes the query-specific opened-cover shortcut during initial
ID/color binding, then adapts to opened scenes. Each training input T1 for this
diagnostic; XYZ endpoint loss, train-only original XYZ mean/std, seed97.
AdamWhead3e-4/backbone3e-5,WD1e-4,20updatewarmup/cosine0.1floor each stage,
BF16,gradientclip1. Validation reveal/opening every100updates; select opening
mean only. Final ordinary30past/current-frame opening val and held-out
composition64/spatial32/combined64 tested on saved best, no test optimization.
No hidden tracking claim. No H/K/actions/flow/robot rollout. README/config/curve,
best weights only, predictions and trained-checkpoint causality checks saved.
Tmuxhuy:e97_debug; stdout/stderr tee workspace_analysis/logs/e97_debug.log.

Complete. Best step500, T1openingval42.444mm, T30openingval42.613mm. Tests in test.json, causality checks passed.

Diagnostic complete. Superseded checkpoint weights removed; metrics retained.
