# E115: four fixed closed-loop K10 demonstrations

Status: complete. One success in four fixed diagnostic cases; three timeouts.
This is not a population success-rate estimate. Tested the fixed best epoch-10 /
step-9268 checkpoint from E115, without changing it or the frozen encoders.

Training: full 2,048 episodes / 234,434 expert stride-3 chunks per epoch, 10
epochs, batch 256, AdamW LR 2e-5. Freeze E114 epoch-9 mask TCOW and ImageNet
auxiliary ViT-B16; train only FM and its existing joint encoder, initialized from
reach step115000. No XYZ head/loss or context decoder. H25 actions are future
EE delta XYZ and relative rotation 6D anchored at the current EE pose, plus
absolute jaw. Full metadata is in ../train/config.json and test config.json.

Four cases selected before rollout outcomes: g000/p00/t0 (train), g000/p01/t0
(unseen permutation), g036/p00/t0 (unseen layout), g036/p01/t0 (both unseen).
All use RedCube to match prior four-case diagnostics. H25/K10, Euler 10 flow
steps, noise seed 0 reset per case, maximum 900 frames / 36 s at 25 Hz. Start
from recorded expert initial physical state immediately after shuffle. All
subsequent motion is policy-controlled with the existing pose IK/servo.

Read the recorded context, not future expert images/actions. Context frames use
the exact training stride3-plus-terminal cache cadence. Live top frames are
retained at stride3; append the exact current frame temporarily for off-cadence
K10 replans. Uniformly select 30 past/current frames with frame0 retained, as
training. Prompt only the visible target mask at context frame0. Top hidden
uses the same block16 int8 roundtrip and auxiliary features the same FP16
boundary as training. Inputs include current left RGB (static surrogate wrist)
and measured current joints. No later GT masks, task ID or object coordinates
are policy inputs. The reused scene initializer computes a point annotation
for older policies; its point query is replaced with the initial mask before
any TCOW forward and that annotation is not used by this policy.

Success requires the selected cover deposited in its drop zone, target lifted
over 4 cm with bilateral jaw contacts for 10 frames, and other cover displacement
at most 15 mm. A wrong-object lift fails. Otherwise timeout at 900 frames.

Videos show live top RGB with predicted TCOW mask, current auxiliary RGB, and
initial visible mask prompt. Green target center is GT for display only, not a
model prediction or input. Mask updates at K10 replans and its age is labeled.
Video captures every3 frames with causal hold at 25 FPS. Every output MP4 is
fully decoded for verification. Trace NPZ stores states/actions, initial mask,
packed predicted masks, replan frames and selected history timestamps.

Code: OATFlow/shuffle_bc/evaluate_direct.py. Command:

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.shuffle_bc.evaluate_direct --checkpoint /workspace/robotics_assets/experiments/e115_direct/train/best.pt --output /workspace/robotics_assets/experiments/e115_direct/test_k10
```

Fresh output required. Source snapshots, hashes and full configuration accompany
the checkpoint snapshot. stdout/stderr are mirrored to test_k10.log in tmux.

## Actual results

| Case | Result | Correct cover deposited at end | Max target lift | Other cover XY shift |
|---|---|---|---:|---:|
| Train g000/p00/t0 | Timeout, 900 frames | Yes | 5.95 mm | <0.001 mm |
| New permutation g000/p01/t0 | Timeout, 900 frames | No | 6.28 mm | <0.001 mm |
| New layout g036/p00/t0 | Timeout, 900 frames | No | ~0 mm | 296.8 mm |
| Both new g036/p01/t0 | Success, 337 frames | Yes | 117.32 mm | <0.001 mm |

All four targets are RedCube. The three failures never sustain a bilateral
target lift; the success meets the 10-frame contact/lift streak and other-cover
constraint. The new-layout failure raises cover_b by 128.28 mm while its selected
cover_a is not lifted: a wrong-cover action. Train and unseen-permutation cases
raise the correct cover by 128.45/129.50 mm, but fail the subsequent target pick;
only the train case finishes with the cover deposited in the required drop zone.

Qualitative end frames show the target mask near the true target in both the
train failure and new-layout wrong-cover failure. This does not isolate why FM
fails or establish dense segmentation accuracy; it demonstrates that a visibly
localized target does not ensure a correct action in these cases. No claim that
more epochs or additional TCOW heads will necessarily fix these rollouts.

Four individual H264 videos plus comparison.mp4 were fully decoded. Individual
frame counts: 900, 900, 900, 337 (25 FPS); comparison 900 frames. Total rollout,
rendering and video verification time: 149.85 s; peak allocated VRAM 1.442 GiB.
Frozen module fingerprints and source checkpoint hash remained unchanged. Exit
code 0. Config and results: test_k10/{config.json,summary.json,results.jsonl};
local machine-readable copy e115_rollout_results.json. No additional training
or hyperparameter search was performed based on these outcomes.
