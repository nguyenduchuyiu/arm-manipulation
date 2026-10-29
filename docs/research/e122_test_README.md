# E122 — Original held-out K10 evaluation

Status: COMPLETE —1028/1408successes,73.0114%.

Checkpoint: `e122_full/train/final.pt`, exactly23,520optimizer updates/B128,
approximately10.217full-data epochs. FM/state initialized from E118reachbest120k;
ImageNet ViT-B/16 frozen. Geometry/fusion/query/cloud/goal modules and FM train
jointly on full E1172,048expert demos/32layouts,stride1,FM+weight1cloud Chamfer.
FM LR2e-5,other LR1e-4,warmup100,AdamW(.9,.95),wd1e-4,BF16,clip1.
Architecture:400RGB+aligned XYZpatches→fusion→target-query hidden→FM; same
hidden supervises128point target-cloud head. GT masks provide labels only.

Evaluate exactly the original1,408held-out episodes: composition1,024,spatial256,
combined128. No old-subset test episode now in training is evaluated.
H25/K10/Euler10,noise0,300frames,25Hz; native front RGB-D/calibration,
measured joints and target color. Output EE deltaXYZ/relative rotation6D/jaw.
Correct target lift>4cm with bilateral jaw contact for10consecutive frames;
wrong-object lift fails. Save traces/metrics, no videos.

Exact sources: `../train/config.json`, `../test_manifest.jsonl`, `config.json`.
Compare E120 only on `../baseline_common_test.json`, not its full3,200case rate.

Final23520:composition699/1024,spatial218/256,combined111/128;380timeouts,zero wrong-object lifts. All1408result paths verified unique,original-held-out and train-disjoint. Test1421.39s,H25/K10/Euler10/noise0/300frames. Exact metrics:summary.json;trace outcomes:results.jsonl.
