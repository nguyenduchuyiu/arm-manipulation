# Task-conditioned MobileSAM

MobileSAM TinyViT and original positional/dense prompt encodings stay frozen.
One learned256D sparse prompt per target ID and the pretrained single-mask decoder
are trained from RGB and visible target-instance masks. There are no input points,
boxes, GT masks, depth, positions or joint states. Target IDs match the pick data:
Red0,Blue1,Green2,Yellow3.

The initial experiment covers all E117/E122 training episodes at frame0. One
canonical RGB image per layout/permutation is shared by its four task queries;
initial physical states are checked across those queries. Frozen encoder features
are cached in CPU RAM. Validation/test use the original disjoint splits. Further
evaluation includes later expert frames and the retained outside-X test states.
This is single-frame visible segmentation, not temporal/amodal tracking or FM BC.

Install the official source pinned in each experiment's config:
`uv pip install --python .venv/bin/python --no-deps -e /workspace/robotics_assets/external/MobileSAM`.
Run `python -m OATFlow.task_mask.train` in the project environment and tmux,
with stdout/stderr piped through tee. Run-specific settings/results are kept in
`/workspace/robotics_assets/experiments/e123_mask` and local `docs/research/`.

E124 continues E123's best weights on every frame of all2048original training
episodes, stride1, querying all four targets on every image. The native RGB and
instance labels are cached in RAM; frozen TinyViT runs online with the original
1024pixel SAM preprocessing. Run `python -m OATFlow.task_mask.full_train`.
Only `best.pt` and resumable `latest.pt` are retained. See `docs/research/e124_README.md`.
Continue an interrupted E124 with `python -m OATFlow.task_mask.full_train --resume`;
this rebuilds RAM caches and restores the original model, optimizer, RNG and
shuffled-frame cursor without adding epochs to the original budget.
