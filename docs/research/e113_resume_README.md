# e113 — Resume two-view voxel XYZ to 100 total epochs

Status: complete. Continued epoch10/step970 to epoch100/step9700,90additional
epochs, with no architecture/data/hyperparameter changes. Best epoch95/step9215.

Source `/workspace/robotics_assets/experiments/e113_voxel/train/latest.pt`;
original10-epoch results and checkpoints are preserved. New output:
`/workspace/robotics_assets/experiments/e113_voxel/resume100`.
One32mm RedCube at15,466distinct2cm voxel centers; fixed yaw0/home robot and
top+left cameras. RGB320square and binary visible masks per current view;
12,372train/1,547val/1,547test, original split copied byte-for-byte, original
train-only XYZ statistics retained. No new data generation or augmentation.

Shared ImageNet ViT-B/16 frozen/eval;400patch tokens per view; mask occupancy
embeddings, projected visual tokens and learned view/spatial embeddings feed
the same4-layer width384/6head decoder and XYZ head.10,073,475trainable
parameters. FP16 at frozen encoder output matches the original training cache.
No joint, memory, FM or action control. H/K/flow/noise budget do not apply.

Restore model, AdamW moments/step, NumPy sampling RNG, CPU/CUDA RNG states.
Batch128; lr1e-4constant; weight_decay.01; normalized XYZ MSE; BF16/TF32;
no gradient checkpointing. Rebuild frozen features in host RAM (~17.70GiB,
masks2.95GiB); no persistent feature cache. Verify baseline validation matches
epoch10 before the first resumed update, frozen encoder/model/statistics and
optimizer step integrity. Exact checks saved in resume_checks.json.

Validate every epoch; keep the lowest mean validation XYZ error over all100
epochs, including the carried-over epoch10 best. Test this selected checkpoint
after epoch100 on the same1,547held-out test voxels; report physical mm errors
and <=1/5/10mm rates. Check fresh RGB/cache inference parity and unchanged
encoder hashes. This tests interpolation for the same cube/cameras, not
occlusion tracking, new objects/cameras or grasp success.

Run from project root:
```
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -u -m OATFlow.localization.train_two_view --resume /workspace/robotics_assets/experiments/e113_voxel/train/latest.pt --epochs 100 --output /workspace/robotics_assets/experiments/e113_voxel/resume100
```

Live log: tmux `ssh_tmux:1`, saved as `e113_voxel/resume100.log`. Config/source
hashes/split/history/checkpoints/results are saved in the new run directory.

Resume integrity checks passed: decoder tensors exactly match latest.pt;
AdamW states restored with step970 and lr1e-4; original split SHA256 unchanged;
normalization and frozen encoder exact. Rebuilt-cache validation before the
first resumed update exactly reproduces20.357273mm. NumPy/CPU/CUDA RNG states
restored after preparation and checks, before epoch11.

## Results after 100 total epochs

Best checkpoint selected solely by mean validation XYZ error over all100epochs:
epoch95/step9215. Latest checkpoint is epoch100/step9700; its optimizer states
also report9700updates. Test contains the same1,547held-out voxel IDs and exactly
the same XYZ labels as the original10-epoch test. Original10-epoch checkpoint
file hashes are unchanged.

| Metric | 10 epochs, best epoch10 | 100 epochs, best epoch95 |
|---|---:|---:|
| Train mean XYZ error | 19.818mm | 5.332mm |
| Validation mean XYZ error | 20.357mm | 7.217mm |
| Test mean XYZ error | 20.711mm | 7.345mm |
| Test median error | 18.195mm | 6.832mm |
| Test P90 error | 36.026mm | 12.096mm |
| Test P95 error | 42.705mm | 13.791mm |
| Test maximum error | 81.245mm | 27.854mm |
| Test error <=1mm | 0.00% | 0.45% |
| Test error <=5mm | 1.81% | 28.12% |
| Test error <=10mm | 14.03% | 79.12% |

Mean test error decreased64.54%. Test per-axis MAE XYZ=4.379/3.185/3.547mm.
Longer training substantially improves the same model's visible-object
localization. This does not establish unseen-object/camera generalization,
off-grid continuous XYZ precision, occlusion memory or robot grasp success.

Additional90epochs plus final evaluation:1,215.2s (~20m15s), excluding88.1s
to rebuild the17.70GiB host-RAM feature cache. Masks use2.95GiB. Peak allocated
CUDA memory2.285GiB; cache was released at process exit and not persisted.
Final encoder tensor/file hashes verified unchanged. Fresh RGB vs cached XYZ
on8test examples matched exactly (max0.000mm), retaining the FP16 boundary fix.

Artifacts in `resume100/`: `best.pt`, `latest.pt`, `summary.json`,
`resume_checks.json`, `comparison.json`, combined100-epoch `history.json`,
`test_predictions.npz`, `comparison.png/.pdf`, `train_examples.png`,
`test_examples.png`. Example sheets span best/intermediate/worst errors rather
than randomly sampled examples. `comparison.json` records checkpoint and
original-run integrity checks.

Best checkpoint SHA256:
`3e78f782846a81933286a5741fb7ffe1c86602348ad8e659afedc7724b02a765`.
Latest checkpoint SHA256:
`10991701fff7dfbd9f51b70d5953b4bc182466a64403e6dfada00b6bd0784ee7`.

Architecture is unchanged; see the Mermaid diagram in
[the original experiment notes](e113_train_README.md).
