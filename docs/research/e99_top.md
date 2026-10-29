# e99_top

Status: complete;40,000 total updates, best step27000.

Purpose: compare one top RGB camera against the completed e94 three-camera
frozen-ViT localization model. Input is current top RGB320x320 and object color
ID; output is the object's actual center XYZ in robot-base metres. This is the
e55/e91 single-frame localization task, not the proposed point-prompt tracker.
No calibration, masks, initial XYZ, expert target/slot or other RGB views enter
the network. Calibration is used only to draw evaluation overlays.

Data: /workspace/datasets/e91_cam3, all3840 e55 episodes, original layout-disjoint
3072train/384val/384test split,32/4/4layouts. All24permutations and all4object
labels per state. Stride5plus final,90433/11513/11374states, exactly the e94
sample/frame ordering. XYZ mean/std from train only,5mm std floor; compare
normalization and sampled frames against e94 exactly before optimizing.

Frozen torchvision ImageNet1K_V1 ViT-B/16,320px, identical spatial interpolation
and weights to e94. Pool20x20patch features to8x8:64tokens, top view only.
Encode/decode only top RGB. Features float16 in host RAM, about10.38GiB across
all splits, never exported to disk. Head is exactly e94's192D projection,
view/spatial embeddings, two independent-query cross-attention blocks and XYZ
MLP. Same seed92 head initialization and1091331parameters are retained to isolate
the input change;384parameters for the two unused camera embeddings have zero
gradient. ViT has no trainable parameters and is verified by an entire-run SHA.

Fresh random head, no e94 trained head weights.40000updates, batch128states,
four object queries/state,512XYZlabels/update,56.62effective sampled-image epochs.
AdamW,WD1e-4,clip1,BF16,TF32 enabled,seed92. Match e94's two-stage LR schedule:
200update warmup then cosine3e-4to3e-5at10k; continue same AdamW moments and cosine
3e-5to3e-6over the next30k, without resetting optimizer or warmup.
MSE on normalized XYZ. Validate all held-out val layouts every500updates and
select the lowest mean Euclidean XYZ error across the entire run.

Final evaluation: best checkpoint on all45496held-out object labels; per-ID,
axisMAE,median,p90 and1/2/5/10/20/50mm rates. Also matched e92 test subset,
shuffled-image sanity check and training-mean baseline. Compare predictions
against e94/resume40k on exactly the same labels/order; reference mean4.981mm.
Four top-only prediction-overlay videos from held-out recorded expert episodes;
no robot policy/IK/flow/H/K/action rollout applies. Frame budget: source25Hz for
qualitative videos, stride5plus final for numerical evaluation.

Output: /workspace/checkpoints/e99_top. Actual settings and hashes in config.json;
split.json, sample_index.json, normalization.json, comparison_check.json,
frozen_check.json, metrics.jsonl, train_metrics.jsonl, progress.json, best.pt,
latest.pt, test.json, comparison.json, predictions.npz, evaluation.png and
test_videos/. Training and evaluation are complete; all four qualitative videos fully decoded.

Run from /workspace/arm-manipulation:

```bash
PYTHONPYCACHEPREFIX=/tmp/e99_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u workspace_analysis/train_object_xyz_full.py \
  --view top --steps 40000 --output /workspace/checkpoints/e99_top
```

tmux: huy:e99_top. Log: workspace_analysis/logs/e99_top.log.
Code: workspace_analysis/{train_object_xyz_full,train_object_xyz,object_xyz}.py.

Full test mean5.876mm, median4.899mm. Shared e92 test mean5.527mm. Exact test.json.

Final held-out results: mean5.876mm,median4.899mm,p9011.465mm;
within1/2/5/10/20mm=3.49/13.85/51.10/85.46/99.31%. Axis MAE
X/Y/Z=4.972/2.128/0.322mm. Lifted-object mean5.943mm. Matched e92 subset
mean5.527mm. Three-view reference mean4.981mm; increase0.894mm (+17.96%).
Reference within5/10mm=58.95/90.75%. Same labels/order verified exactly.

A separate batch-one inference benchmark used four actual frames of
held-out g036/p00_t0, querying all four object IDs per call. RTX4090,
BF16 autocast,10warmup calls and100timed calls per model; CUDA synchronized
wall time includes CPU uint8 RGB copy, frozen ViT/pooling, head and XYZ restore,
excludes camera capture/video decoding. Top mean5.984ms vs three-view9.139ms:
1.53x inference speedup in this small benchmark. It is not a camera-system
throughput or training-speed measurement. Exact settings/results in speed.json.

verification.json confirms all80validation records, exact saved-prediction
metric reconstruction, matching source hashes, matching LR at500/10000/10500/
40000, best selection at27000 and frozen ViT SHA after training and benchmark.
comparison.png/PDF plot validation curves and held-out error CDF. The run's
artifacts occupy about31MiB; features were RAM-only and have been released.
These results concern ID-conditioned e55 localization on a fixed camera rig;
point prompts and tracking through complete occlusion were not tested here.
