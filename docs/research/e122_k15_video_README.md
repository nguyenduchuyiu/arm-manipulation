# E122 — K15 failure videos

Status: COMPLETE — three K15failure videos fully decoded and source artifacts unchanged.

Three diagnostic timeout clips: first failure in each of composition,spatial,
combined groups,selected from the complete1408case K15results. Outcome selection
is not a representative failure-rate estimate. Replay actual saved physical qpos,
without new policy actions or physics. Full K15result is1144/1408=81.25%.

Training/checkpoint:final23520,full2048E117expert demos/32layouts,stride1,
10.217214epochs,B128. FM/state initialized from E118reachbest120k;ImageNet
ViT-B/16 frozen/online.400RGB+XYZpatches→fusion→target-query hidden→FM;
same hidden supervises128point cloud head. Geometry/fusion/decoder/projections/
head/FM train jointly. FM+1.0normalized squared Chamfer,AdamW(.9,.95),wd1e-4,
clip1,BF16,warmup100,FM LR2e-5/others1e-4. Exact config:`../../train/config.json`.

Original rollout:H25/K15/Euler10,noise0,300frames at25Hz. Correct-target lift>4cm
and bilateral contacts10frames for success; selected cases all time out.

Panels:front RGB+GT/predicted target surface;oblique robot replay+EEpath;
25cm target-cloud zoom;surface error,EEdistance,lift/contact/prediction-age metrics.
Orange=GTvisible target cloud,cyan=auxiliary prediction,green=measured EE.
Video1344×1100/25FPS,H264,CRF18;save start/end images and contact sheet.

Cloud predictions refreshed at actual K15pre-action replans0,15,...,285,
held between replans;no extra inference at terminal300. Display prediction age
so held cloud error is distinguishable from fresh inference error. Native input
RGB-D is rerendered from saved float32qpos;small raster/numerical differences
from original runtime observations are possible. No GTmask/center enters policy;
GTannotations and body centers are diagnostics only. GTsurface uses1pixel
erosion,60mm median-depth band and10mm3×3local median spike rejection.
Surface Chamfer is unsquared against all visible points,not object-center error.

CUDA initialization is currently unavailable,while EGLrendering was verified.
Compute diagnostic clouds explicitly on CPU/FP32; original policy ran CUDA/BF16.
Saved actions/physical trace remain the source of robot motion. Recomputed cloud
values are approximate diagnostic inference,not the original runtime hidden/output.

Verify complete video decode,expected frame/replan counts and original
trace/checkpoint/results hashes. Sources:`config.json`, `verification.json`,
per-video metrics/predictions;original `../config.json`, `../results.jsonl`.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.top_pick.full_video --k 15 --device cpu
```

COMPLETE:301frames/12.04s per clip,3clips. Verified20cloud replans at0,15,...,285; original traces/checkpoint/results unchanged. CPU/FP32diagnostics with EGLrendering.
