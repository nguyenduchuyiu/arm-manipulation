# Top-camera object localization

Frozen ImageNet ViT-B/16 ->64tokens ->small ID-conditioned XYZ head. Retained
checkpoint: robotics_assets/models/object_localization_topview/best.pt,best27000.
Pretrained encoder: robotics_assets/pretrained/vision/vit_b_16-c867db91.pth.
Input one320x320top RGB and color ID red/blue/green/yellow=0/1/2/3. XYZ uses
checkpoint mean/std and returns base metres. Model.view retains the original
three-entry tensor shape for strict loading; inference uses the first entry.

```bash
python -m OATFlow.localization.infer top.png --id 0
```

`FrozenViT(tokens=400)` retains all20x20 spatial patch tokens without pooling;
pair it with `ObjectXYZ(tokens=400)`. The default64-token model and retained
checkpoint loading are unchanged. Inference accepts `--checkpoint PATH` and
reads the token count from its saved configuration. The400-token head has
1,155,843parameters versus1,091,331for64tokens; the difference is the spatial
embedding table. All other head layers and the frozen encoder are identical.

The no-pooling ablation is not trained yet: exact e99 reproduction requires
restoring deleted e55/e91 RGB and labels. A comparison on the retained shuffle
dataset requires training both64-token and400-token controls on the same split.

Original test mean5.876mm is from the deleted e55/e91 training/test data,not the
shuffle benchmark. Current package supplies inference; historical3view and
full-finetune training wrappers were removed.

## Two-view mask-conditioned voxel XYZ (e113)

`two_view.py`: shared frozen ViT-B/16 over top+left320square RGB,400patch tokens
per view; binary target masks supply patch occupancy embeddings. A trainable
4-layer width384 decoder with one query reads all800tokens and regresses
normalized robot-base XYZ. Frozen encoder outputs are canonical FP16 for both
online and cached inference. No joint input, tracking memory or FM.

`train_two_view.py` renders one fixed-yaw32mm cube at the accepted2cm voxel
positions visible to both cameras, caches frozen features in RAM, trains10epochs,
and tests the validation-selected checkpoint. `evaluate_two_view.py` evaluates
the completed run without optimizer updates. Run commands, exact split and
results: [e113 notes](../../docs/research/e113_train_README.md).

Retained run: `/workspace/robotics_assets/experiments/e113_voxel/train/best.pt`,
epoch10/step970, test mean20.711mm over1,547held-out voxels. This is localization
of the same visible cube/cameras, not unseen objects or occlusion tracking.

The run was resumed with AdamW/RNG/split/statistics restored to100total epochs.
Retained best: `/workspace/robotics_assets/experiments/e113_voxel/resume100/best.pt`,
epoch95/step9215; latest epoch100/step9700. Identical held-out test: mean7.345mm,
median6.832mm,79.12% <=10mm. [Resume notes](../../docs/research/e113_resume_README.md).
The trainer accepts `--resume PATH --epochs TOTAL --output NEW_DIRECTORY`.
