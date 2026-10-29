# e97_xyz/full/test_open

Evaluate native three-camera amodal XYZ at cover opening, before target grasp.
Status: complete. Source best.pt selected at full-stage update 1250 by minimum
exposed validation mean across the six saved validation steps (250–1500).
Training completed 1500 full-finetune updates from partial-stage step1000 weights,
all 125,313,539 parameters trainable; 512 train episodes, batch2/micro1, BF16,
backbone LR3e-5/new modules3e-4, train-only XYZ normalization, no action output.
Exact original training configuration and source archived in parent directory.

Test: 32 validation, 64 composition, 32 spatial, 64 combined episodes; exact paths
from native data.subset. Validation uses only exposed frames; test additionally
reports reveal/shuffle/post-shuffle/lift separately. Exposed is first cached frame
at or after decision_frames.t_obj (stride3 can delay <=2 raw frames), before grasp.
30 equally spaced past/current frames include context frame0 and chosen endpoint;
RGB top/left/right resized320x240, target ID and timestamps only. No GT masks,
centers, cover states or calibration as input. No H/K/flow steps/robot rollout.
Seed97, inference BF16. Metrics use Euclidean XYZ error in base-frame mm.

Checkpoint /workspace/checkpoints/e97_xyz/full/best.pt, step1250. Trained-checkpoint
future RGB gradient and future RGB/timestamp replacement checks at input frame14.
Run: .venv/bin/python -m OATFlow.tracking.evaluate
Artifacts: config.json, metrics.json, group predictions/samples, causality.json,
progress.json. Tmux huy:e97_test; log workspace_analysis/logs/e97_test.log.

opening_examples.png: four validation query endpoints after cover removal,
true center in green, predicted center in magenta. Camera projection is used
only to draw this visualization, never supplied to the network.

Complete. Opened-target mean error (mm): val 106.233, composition 101.168, spatial 107.162, combined 107.133. Full phase metrics in metrics.json; trained-checkpoint causality checks passed.

Prediction spread/baseline diagnostic in diagnostic.json. The model is near constant
across target IDs and scenes; full finetuning did not yield useful localization.
