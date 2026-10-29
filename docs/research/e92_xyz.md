# e92_xyz

Status: complete; 10000 updates; best step9500.

Predict measured object center XYZ in workspace base frame, metres, conditioned
on RGB from three fixed calibrated e90 views and object identity 0=RedCube,
1=BlueCube, 2=GreenCube, 3=YellowCube. No joint state, EE, slot, target position,
episode index or calibration shortcut is provided to the network.

Data: e91_cam3/subset.json, 8 train/2 val/2 test layouts, 768/192/192 episodes.
All permutations retained; image states sampled every5frames (5Hz) and final
frame. Train on all four objects per image, including lifted objects, rather
than only the expert's target. Layout-disjoint original splits.

Shared ImageNet-1K ViT-B/16 is frozen/eval, including its position embedding.
At320px interpolate original14x14 positions to20x20; exclude classifier/CLS
output, pool400 spatial tokens to8x8=64tokens/view. Three views give192tokens.
ImageNet RGB normalization. Cache frozen features in RAM, not whole-dataset disk.
Train a 768->192 projection, learned view/spatial embeddings, two small
cross-attention decoder blocks with four object ID queries and an XYZ MLP.
XYZ normalized with training-only mean/std. AdamW, batch128 image states,
LR3e-4, weight decay1e-4, max10,000updates, warmup200/cosine decay,
validate every500updates, save best validation mean Euclidean XYZ error.
Random seed92. No augmentation in this first experiment.

Output: `/workspace/checkpoints/e92_xyz`; config.json, subset audit,
metrics.jsonl, best.pt, latest.pt, final held-out test.json and qualitative
prediction videos. Backbone initialization is official torchvision
ViT_B_16_Weights.IMAGENET1K_V1; exact source/hash recorded by run.
No FM/action prediction or IK in this experiment.

Tests passed: leakage/ID mapping checks, backbone frozen checksum,
mean-per-ID position baseline, held-out errors in mm and thresholds1/2/5/10/20mm,
per-ID and lifted-vs-table metrics, shuffled-image diagnostic.
Code: workspace_analysis/object_xyz.py, train_object_xyz.py, infer_object_xyz.py.
Exact training config: config.json; source split/sample audit: subset_audit.json and sample_index.json.

Actual trainable parameters: 1,091,331; effective image epochs: 56.95.

Test mean 13.908mm, median 9.862mm, p90 30.728mm. Exact results: test.json.

Run: PYTHONPYCACHEPREFIX=/tmp/e92_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u workspace_analysis/train_object_xyz.py

Test uses2 independent layouts,192episodes,5,674image states and22,696object labels.
Thresholds:1mm1.30%,2mm5.56%,5mm23.73%,10mm50.40%,20mm72.99%.
Train-mean-per-ID baseline98.04mm; shuffled test images123.53mm.
Some held-out frames still have1-3cm errors; localization accuracy is not a grasp success metric.

Prediction video: test_videos/comparison.mp4 (4original test expert episodes; circles=measured centers, crosses=predictions).

Inference: `.venv/bin/python workspace_analysis/infer_object_xyz.py TOP.png LEFT.png RIGHT.png --id 0`; IDs0red/1blue/2green/3yellow, input320x320RGB, outputXYZbase metres.
