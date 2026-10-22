# Point-query XYZ tracking

Native one-camera causal TCOW. Input top RGB[B,3,T,240,320] and binary
query[B,1,T,240,320]; radius3dot at frame0,zero later frames. No object ID,
initial3Dcoordinate or GT mask enters the tracker. Output normalized XYZ[B,T,3];
`predict` returns current base metres. RGB(.45,.225); train-only XYZ statistics
are checkpoint buffers. `PointXYZ(local_query=True)` adds a fixed frame0
3x3patch appearance reference and128D cosine matching. Both retained variants
load from checkpoint['model_config']/['model']; backbone keys remain unchanged.

Data: robotics_assets/datasets/shuffle_pick_multiview.30frames sampled over the
past/current prefix,frame0 retained. Full finetune,XYZ loss only. Approved subset
512train/32val/64unseen permutation/32unseen layout/64both. Select best at cover
opening; final reports include occlusion and lift. Trackers remain experimental.

Run from repository root with `.venv/bin/python`:

```bash
python -m OATFlow.tracking.train_point --local-query --steps 1500 --output /tmp/new_tracker
MUJOCO_GL=egl python -m OATFlow.tracking.pick_video --output /tmp/new_tracking_pick
python -m unittest OATFlow.tests.test_point_xyz -q
```

Omit --local-query for the point-only variant. All output directories must be
fresh. Rollout accepts --tracker and --reach checkpoint paths. Its prelude is
recorded shuffle/expert cover removal,then exact home reset; live visual FM
reach/grip uses H25/K15,flow10,300actions, followed by a separate100action IK
lift probe holding the last FM gripper command. This is not learned cover
selection or learned lifting. Saved states/commands are independently replayed.

Native target-mask fine-tuning (e114):

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python -m OATFlow.tracking.train_mask --epochs 10 --batch 4 --output /workspace/robotics_assets/experiments/e114_mask/train_3
```

Original TCOW backbone and pretrained mask readout; only target channel retained.
One initial visible mask query, top RGB only, amodal target-mask loss. No XYZ,
pooling, context decoder, or FM in this stage. Full2048train episodes, one random
causal30frame prefix per episode per epoch. Details: docs/research/e114_mask_README.md.
