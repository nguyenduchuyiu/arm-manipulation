# Combinatorial pick pretraining

Lift the queried cube by 4 cm and hold it with both jaws for ten 25 Hz frames. Four physically simulated cubes are present; covers are parked. There is no shuffle or placement stage.

Each layout contains all 24 identity permutations and four target queries per permutation. Paired queries share the exact initial state. Slot centers vary between layouts; object offsets and yaw vary per permutation. Robot joints use an independent random stream shared across a permutation group. There are no artificial state changes during expert trajectories.

Reject a complete layout if the contact-based expert cannot finish every query, preserving combinatorial balance. Rejection reasons are recorded. Splits use disjoint layout seeds; this benchmark covers the expert's feasible workspace. The policy receives constant task 0, target identity, current RGB views and proprioception. Layout, permutation and slot IDs are metadata only.

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 \
python -m OATFlow.pick.collect --output "$PICK_DATA" --workers 8

python -m OATFlow.prior.features --data "$PICK_DATA" \
  --vision-weights "$VIT_WEIGHTS" --output "$VIT_CACHE" --ram-gib 10

python -m OATFlow.prior.train --data "$PICK_DATA" \
  --vision-cache "$VIT_CACHE" --output "$PICK_CHECKPOINTS" \
  --vision-weights "$VIT_WEIGHTS" \
  --action-head act --horizon 50 --epochs 10 --batch-size 256 \
  --lr 1e-4 --min-lr 3e-6 --warmup-steps 500
```

Use fresh output directories and keep artifacts outside the repository. Every valid expert frame starts a chunk (stride 1), including short masked terminal chunks. All full training batches cover the 24 permutations; every chunk is used once per epoch. ACT and its full conditioning stack, including proprio projection, initialize from scratch; only ViT loads pretrained weights. Change `--action-head act` to `fm` and add `--flow-weights "$FLOW_EXPERT"` for flow matching. Both use the same frozen ViT cache and trainable conditioning stack; details are in [prior/README.md](../prior/README.md).

```bash
python -m OATFlow.pick.evaluate --data "$PICK_DATA" \
  --checkpoint "$PICK_CHECKPOINTS/final.pt" --split train \
  --output "$TRAIN_EVAL" --execute-chunk 10
python -m OATFlow.pick.evaluate --data "$PICK_DATA" \
  --checkpoint "$PICK_CHECKPOINTS/final.pt" --split test \
  --output "$TEST_EVAL" --execute-chunk 10
python -m unittest discover -s OATFlow/tests
```

Evaluation uses learned actions, 25 Hz waypoints with 50 Hz linear interpolation, a 300-frame budget and no temporal ensemble. Reports include color confusion and success by slot; lifting a wrong object is failure. Videos and simulator traces are saved for paired queries. Report train, validation and held-out success separately; training loss does not measure grasp competence.
