# e94_full/resume40k

Status: complete;40,000 total updates, best step39000.

Continue e94 frozen full-data XYZ training from latest.pt step10,000 to
40,000total updates (+30,000). Full training exposure grows from14.15 to56.62
image epochs, comparable to frozen e92 subset56.95epochs. Preserve10k artifacts.
Exact same full splits3,072/384/384episodes,90,433/11,513/11,374states sampled
every5frames plus final; all4IDs,3RGB320views,64tokens/view,192total.
Frozen ImageNetViT-B/16 and same1,091,331parameter ID-query fusion head.
RGB ImageNet normalization and full train-only XYZ normalization unchanged.

Resume head weights AND AdamW moments/step counters from
`/workspace/checkpoints/e94_full/latest.pt`. Batch128states/512labels,
WD1e-4, global grad clip1, no augmentation. Continue LR from checkpoint3e-5
with cosine decay to3e-6 over30,000additional updates; no new warmup/restart.
Seed92 sampling state reconstructed for the original checkpoint (training's
only CUDA random calls were10,000randint(N,128) calls); later saves include RNG.
Frozen feature cache recomputed in host RAM (~31.12GiB), never disk/GPU full-cache.

Validate every500updates on all4validation layouts, select best mean XYZ error.
Original best10k is retained as a candidate. Also log matched e92 val subset;
final all4test layouts and matched e92 test subset, mm thresholds/perID/lifted
metrics, predicted coordinates and4prediction videos. Final tests and frozen-backbone checksum passed.
No FM/action/IK/learned robot rollout, H/K, flow or noise settings apply.

Output: `/workspace/checkpoints/e94_full/resume40k`, config.json, resume_check.json,
normalization.json, sample_index.json, frozen_check.json, metrics.jsonl,
train_metrics.jsonl, progress.json, best.pt/latest.pt, test.json, predictions.npz,
evaluation.png and test_videos/. Original e94 10k remains in its parent.

Run: `PYTHONPYCACHEPREFIX=/tmp/e94_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u workspace_analysis/train_object_xyz_full.py --resume /workspace/checkpoints/e94_full/latest.pt --steps 40000 --output /workspace/checkpoints/e94_full/resume40k`
tmuxhuy:e94r; log workspace_analysis/logs/e94_resume40k.log.
Code: workspace_analysis/train_object_xyz_full.py and shared train_object_xyz.py/object_xyz.py.

Preflight passed: step10,000 weights/AdamW restored, optimizer counters10,000;
continuation LR boundary3e-5 ->3e-6 verified, total exposure56.6165epochs.

Actual resume_check.json passed: exact source head SHA, AdamW counters and
XYZ normalization restored; seed92 CUDA sampler advanced through10k original
randint calls. First continuation validation at10,500completed; original best
4.924mm retained. Observed~34updates/s, ~15minutes for30kadditional updates.

Full test mean4.981mm, median4.210mm. Shared e92 test mean4.656mm. Exact test.json.

Final held-out results (best39k): full test mean4.981mm, median4.210mm,
p90 9.795mm; success1/2/5/10/20mm =5.34/18.96/58.95/90.75/99.87%.
Matched e92 test mean4.656mm, median3.903mm, success10mm92.18%.
Compared with full-data10k mean5.552mm, mean error decreases10.29%.
Saved predictions reproduce reported metrics; entire-run ViT checksum matches.
