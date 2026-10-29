# e112 — Original iSEE on recorded robot shuffle videos

Status: complete. Zero-shot inference only, no fine-tuning. All five checkpoint
files and all upstream Python model sources have unchanged SHA256 hashes.

Source: https://github.com/insait-institute/iSEE at
`e7100a0a036790372f502f0496a518d61056970e`, unchanged official
`configs/lacater_static.yaml`: 11 slots, 518-square input, TEN tau=.5/rho=.9,
three original static-camera walkers, native walker stride 2, dmax=.35.
All weights frozen. No optimizer, epochs, learning rate, action head, FM,
H/K, flow integration or robot-control rollout applies.

Data: `shuffle_pick_multiview`, top camera, rows 40:280 cropped from 320-square
recordings. Full recorded context followed by recorded expert motion, every
frame at 25 FPS. Four fixed episodes: g000/p00/t0 (train scene), g000/p01/t0
(unseen order), g036/p00/t0 (unseen layout), g036/p01/t0 (both). These split
names refer to our earlier robot training, not to iSEE's LA-CATER training.

The first-frame target center supplies one click. The argmax corrector slot at
that click is selected once and never reassigned using subsequent labels.
iSEE itself receives RGB only and discovers slots automatically: this is not
a native point-prompt interface. Ground-truth masks and center projections are
used only for evaluation and overlays, never for model inference. Native pose
readout follows official infer.py: corrector position when visible, walker
position when held. This is a 2D slot readout, not trained robot XYZ prediction.

Environment: project Python/PyTorch with isolated timm==1.0.30 at
`/workspace/isee_setup/python`; original timm==1.0.12 cannot handle NoPE prefix
tokens correctly. Official repository omits `isee/data/video.py` although its
inference script imports it. Our adapter therefore implements the documented
square resize with bicubic antialiased interpolation, followed by ImageNet
normalization. Exact parity with that unavailable preprocessing helper cannot
be certified. Encoder images are batched in chunks of 8; all original recurrent
slot/TEN/walker code runs unchanged on each complete clip. Float32 with TF32,
seed 0. Recorded-video inference, not a streaming latency benchmark.

Run from project root:
```
PYTHONPATH=/workspace/isee_setup/python:/workspace/iSEE .venv/bin/python -u -m OATFlow.tracking.evaluate_isee --output /workspace/robotics_assets/experiments/e112_isee/test
```

Exact cases, source/weight SHA256, actual timings and metrics are written to
`config.json` / `summary.json`; per-frame arrays to NPZ. Videos show GT green,
native predicted center magenta, and the fixed selected slot on the right.
Body-center pixel error is descriptive: native slot centers need not equal
geometric body centers. Four clips are a diagnostic, not a general success rate.

## Results

| Earlier robot split | Frames | Initial selected-slot IoU | Visible frames with target majority in fixed slot | Mean native center error (px) |
|---|---:|---:|---:|---:|
| Train scene | 643 | .193 | 23.8% | 74.6 |
| Unseen order | 611 | .003 | 4.7% | 105.6 |
| Unseen layout | 548 | .138 | 19.1% | 67.9 |
| Both unseen | 600 | .095 | 24.6% | 109.0 |

Pixel coordinates refer to the 320x240 cropped input, not the enlarged video.
The visible-frame metric only includes frames with >=50% of the target visible
and >=20 visible target pixels. It measures whether the fixed selected slot owns
the majority of visible target pixels; it is not robot grasp success or hidden
localization accuracy. The majority target owner on the final visible frame
differs from the initially selected slot in all four clips.

Visual inspection: the initial query slot often groups multiple object parts,
background or gripper regions. During covering/shuffling it can attach to a
cover or the robot, and the target receives a different slot after reappearance.
This unchanged LA-CATER checkpoint does not provide reliable zero-shot target
tracking in these robot scenes with this click-to-slot adapter. Impure initial
slots already limit the test, so it does not isolate memory quality from object
discovery/domain transfer or native pose calibration. No conclusion about a
fine-tuned model or robot-control success follows from this test.

All 2,402 video frames were decoded after encoding; numerical outputs are finite.
Peak allocated CUDA memory was .60-.68 GiB. Batched image encoding plus the
recurrent slot/TEN/three-walker forward took approximately 8-10 seconds per
548-643-frame clip on RTX 3090. That excludes native mask decoding, NPZ export,
GT evaluation and rendering; it is not streaming end-to-end latency.

Artifacts: `/workspace/robotics_assets/experiments/e112_isee/test`,
live/saved log: `/workspace/robotics_assets/experiments/e112_isee/run.log`,
tmux `ssh_tmux:1` (window e112). Videos: `train.mp4`, `unseen_order.mp4`,
`unseen_layout.mp4`, `unseen_both.mp4`. `verification.json` records full decode
and finiteness checks. Ground-truth overlays show recorded expert motion;
iSEE does not control the robot in these clips.
