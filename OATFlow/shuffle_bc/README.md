# Shuffle TCOW behavior cloning

Original top-only shuffle history plus a frame0 point query. Within each shared
context, chronological endpoints stride10 train only TCOW/XYZ with all action
labels masked. Then expert endpoints stride3 train TCOW+goal projection+FM+XYZ
end-to-end. Batch4 directly; no gradient accumulation or checkpointing.

Original official TCOW RGB+query backbone and reach FM115000. New pooled770 to
hidden256 supplies auxiliary XYZ and goal960; XYZ never enters FM directly.
No masks, targetID or GTXYZ policy inputs; measured joints enter only the expert
policy. Thirty past/current top frames240x320, queried by a radius3dot at frame0.
Past frames uniformly sample raw time using nearest cached past RGB; context
cache union(stride3,stride10,terminal), expert stride3+terminal.

2048train episodes:61688context+234434expert=296122clips,74230updates per epoch
at batch4. New full one-epoch run is train_2.24update preflight measured about
20–22hours/epoch and19.88GiB peak VRAM on RTX3090. See actual config/progress.

```bash
.venv/bin/python -m OATFlow.shuffle_bc.train --epochs 1
MUJOCO_GL=egl .venv/bin/python -m OATFlow.shuffle_bc.evaluate
```

Fresh outputs required; long runs in tmux with stdout/stderr through tee.
Default actual outputs: experiments/e105_tcow/train_2 and test_k10. Original
expert-only train was interrupted after1update; do not resume it for this recipe.
latest.pt contains optimizer/RNG/sampler state for future continuation; CLI starts
fresh. Full task rollout starts post-shuffle; policy performs cover opening,
deposit and target grasp/lift. H25/K10/Euler10,seed0,900frames. Correct cover must
be deposited, target lift>4cm bilateral10frames, other cover shift<=15mm; wrong
cube lift fails.128cases across train/unseen permutation/layout/both. Best selected
by fixed64chunk FM validation; final is primary test. Controller-only oracle4/4
is not a learned result. Full note: docs/research/e105_tcow.md.
