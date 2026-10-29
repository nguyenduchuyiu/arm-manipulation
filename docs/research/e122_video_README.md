# E122 — Success and failure videos

Status: COMPLETE — six videos fully decoded, original traces/checkpoint/results unchanged.

Replay the first completed success and first completed failure in each original
test group:composition,spatial,combined. Six diagnostic clips from the fixed
complete1408case results; selected outcomes do not estimate success rate.

Checkpoint:final23520updates,full2048E117expert demos/32layouts,stride1,
10.217214effective epochs,B128. E118reachbest120k FM initialization,ImageNet
ViT-B/16 frozen/online.400RGB+aligned XYZpatches→fusion→target-query hidden→FM;
same hidden supervises128point target-cloud head. Train geometry/fusion/decoder/
projection/head/FM jointly,FM+1.0squared normalized Chamfer,FM LR2e-5/other1e-4,
AdamW(.9,.95),wd1e-4,clip1,BF16,warmup100. Exact training:`../../train/config.json`.

Original rollout:H25/K10/Euler10,noise0,300frames,25Hz; correct-target lift>4cm
and bilateral contacts10frames. Replay saved qpos without new actions or physics.
Success clips end at recorded success; failure clips show full timeout.

Panels:front RGB with orange GT visible target surface and cyan prediction;
oblique robot replay with measured green EE path;25cm-wide target point-cloud
zoom; surface Chamfer/centroid,EE distance,lift and actual contact diagnostics.
Video1344×1100/25FPS,H264,CRF18. Save start/end PNGs and six-clip contact sheet.

Auxiliary cloud is recomputed from the checkpoint at pre-action replans0,10,...
and held between; no extra inference at a terminal state. RGB/depth are rerendered
from saved float32qpos, so minor raster/numerical differences from original online
observations are possible. The predicted cloud is diagnostic only; original FM
uses the decoder latent. GT mask/3Dbody center never enter policy observations.
GT surface uses1pixel mask erosion,60mm median band,10mm3×3median spike filtering.
Metric uses all visible GT surface points,not sampled128labels or object center.

Verify complete video decoding and original trace/checkpoint hashes. Sources:
`config.json`, `verification.json`, per-video metrics/predictions, original
`../results.jsonl` and `../config.json`.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.top_pick.full_video
```

Rendered clips:

- `success_composition.mp4`: `g000/p01/t0`, 5.72s, success.

- `failure_composition.mp4`: `g000/p01/t1`, 12.04s, timeout.

- `success_spatial.mp4`: `g036/p00/t1`, 5.28s, success.

- `failure_spatial.mp4`: `g036/p00/t0`, 12.04s, timeout.

- `success_combined.mp4`: `g036/p01/t1`, 5.16s, success.

- `failure_combined.mp4`: `g036/p01/t0`, 12.04s, timeout.

Visual review passed for initial success and terminal failure panels. All videos include the actual saved physical trajectory. Cyan cloud refreshes every10frames and is held between replans: terminal cloud error can reflect prediction age while a grasped object is moving. For success_combined,last actual replan120has3.549mm Chamfer while terminal128held-prediction metric is51.294mm; do not interpret the latter as a fresh inference error. Failures retain large errors at actual last replans290:82.983/52.330/66.739mm for composition/spatial/combined. These are selected diagnostic cases, not a causal attribution or full failure analysis.
