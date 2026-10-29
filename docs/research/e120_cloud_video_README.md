# E120 — Point-cloud diagnosis of successful and failed picks

Status: COMPLETE — six clips fully decoded; physical trace hashes unchanged. Same six cases as `test_k10/videos_1` (one success and one
timeout per composition/spatial/combined group), final23520 after80epochs.
These examples are selected by outcome, not a success-rate estimate.

Training: E117 part1/8 (256demos,37,536stride1chunks),80epochs,B128,FM LR2e-5,
other LR1e-4,warmup100. Frozen RGB ViT; train dense XYZ patch/fusion/query decoder,
cloud head and FM initialized from E118point-cloud reach120k. Loss FM+weight1
auxiliary symmetric squared target-cloud Chamfer. Full training configuration:
`../train/config.json` relative to the parent test directory.

Test: final23520,H25/K10,Euler10,noise0,300frame maximum,25Hz policy/50Hz servo/
500Hz physics. Replay existing saved qpos; no new physics trajectory or action
sampling. Checkpoint SHA and exact selected cases are in `config.json`.

Video panels: native front RGB with projected clouds; full dense RGB-colored
cloud back-projected from actual front depth; target zoom in the same fixed
oblique basis; numeric geometry/control diagnostics. Orange is the visible GT
target surface (eroded instance mask, same depth outlier rejection as training),
cyan is the128point auxiliary prediction, green cross is measured EE. All valid
input depth pixels are used for the scene cloud; no512point downsampling.
GT masks/poses remain diagnostics only, never policy inputs.

The native depth is quantized to2mm and back-projected by the actual model helper
using current camera intrinsics/extrinsics. The predicted cloud is recomputed
from RGB plus the full dense depth/calibration input at pre-action replan frames
0,10,...,held between replans. Terminal post-action frames do not generate an
extra replan. Input geometry and targets update every visualization frame.

Auxiliary prediction recomputation uses float32 saved qpos and freshly rendered
observations, not originally logged hidden states; tiny raster differences from
the original rollout are possible. It does not change the saved action/physics
trajectory. FM attends to decoder hidden, not the predicted point set itself.

Metrics (mm): symmetric mean nearest-neighbor visible-surface Chamfer (unsquared,
all valid GT surface pixels), predicted-vs-GT visible-surface centroid distance,
predicted centroid-to-EE distance and EE-to-GT body center distance. A surface
centroid is not the object center; GT-visible-surface comparison avoids conflating
this bias. Target labels with fewer8pixels are reported N/A for surface metrics.
Target-only zoom is25cm wide and may clip predictions outside that window; full
scene and numeric metrics remain available. Cloud overlays are drawn on top for
diagnosis and do not represent a learned amodal reconstruction.

Each1344×1100clip plays at25FPS. `*_metrics.jsonl` records every frame;
`*_predictions.npz` records predicted clouds at replan frames. Verify complete
video decoding and unchanged source trace hashes in `verification.json`.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.top_pick.fusion_cloud_video
```

Rendering verification passed for all six clips; checkpoint hash verified. Initial and final visible-surface distances (mm; held prediction between replans):

- `success_composition.mp4`: initial 2.279644250869751, final 14.191719055175781, median 2.435798168182373.
- `failure_composition.mp4`: initial 2.341517448425293, final 41.258724212646484, median 7.862515926361084.
- `success_spatial.mp4`: initial 12.22004508972168, final 45.27327346801758, median 7.649771690368652.
- `failure_spatial.mp4`: initial 59.15991973876953, final 63.125633239746094, median 73.28870391845703.
- `success_combined.mp4`: initial 12.464774131774902, final 31.160648345947266, median 11.281900405883789.
- `failure_combined.mp4`: initial 22.527921676635742, final 62.98827362060547, median 22.527921676635742.
