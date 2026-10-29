# Point-prompted object-center tracking demo

Watch [tracking_demo.mp4](tracking_demo.mp4), or open [index.html](index.html) for a video player. Individual clips are [validation](tracking_validation.mp4) and [both unseen](tracking_both_unseen.mp4). The PowerPoint embeds the combined demo; its PDF shows a static poster.

**Green = simulator ground-truth center. Magenta crosshair = model-predicted center. Orange = the initial point prompt.** The initial-prompt thumbnail is always shown as a reference; the model receives that prompt only at frame 0. The main image is a top-camera observation. The plots compare GT and predicted base-frame XYZ in millimeters, revealing depth error as well as image-plane error.

## Model and scope

These videos use the **e100 point-prompted TCOW tracking pilot, best checkpoint step 1,250**, trained for 1,500 updates on a 512-episode subset of the 2,048 training episodes. Sampling used random phase-weighted prefixes with replacement, rather than complete epoch sweeps. The official TCOW RGB/query-channel weights were fine-tuned with an XYZ head; no fixed appearance-reference branch is used.

Inputs are 30 past/current top-camera frames plus a radius-3-pixel dot in the first frame. GT centers, object IDs, and camera calibration are not model inputs. Calibration is used to project GT/predicted XYZ into the video for visualization. XYZ labels supervise the original training.

The two cases were fixed in the original experiment, not selected by prediction quality:

| Clip | Episode | Split | Recording |
|---|---|---|---|
| Validation | `g032/p00/t0` | Validation layout, seen permutation | Reveal → cover → shuffle → expert cover removal → expert target lift |
| Both unseen | `g036/p01/t0` | Unseen layout and permutation | Same task stages |

The robot movements in the recording are expert demonstrations. This is a tracking visualization, not learned robot control. The early pilot has substantial tracking errors, particularly during lifting; the full sequence is retained so that these failures remain visible. These are not the ongoing e105 end-to-end experiment's results. Its result table remains blank.

## Timing and reproducibility

- Output: 1920×1080, H.264 MP4, 25 FPS, with fast-start metadata for browser playback.
- The source RGB is native 320×320, cropped to rows 40:280. Upscaling is for presentation only.
- Actual saved causal predictions are sampled every 3 source frames plus segment-terminal frames. Between outputs, the latest available prediction is held. No future prediction is used for temporal interpolation.
- Dense GT is read from the original simulator labels, and matches every cached GT coordinate to within 1e-7 meters.
- All original RGB MP4s and small coordinate snapshots needed to rebuild are included under `sources/` and `data/`. No checkpoint or GPU is required to re-render these videos.
- Checkpoint identity, original sources, selection, and SHA-256 hashes are in `metadata.json`. Frame counts and full-decode checks are in `verification.json`.

From the parent presentation folder:

```bash
python code/tracking_video.py
```

Dependencies: NumPy, Pillow, PyAV, tqdm, and FFmpeg with libx264. Optional `code/freeze_tracking_video.py` re-extracts the inputs from the original project assets. The rendering script does not run model inference or access CUDA.
