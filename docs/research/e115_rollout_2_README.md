# E115 continuation: matched four-case K10 rollout

Status: complete, exit0. Two successes and two timeouts. Select best by fixed validation loss before
rollout: train_2/best.pt, epoch32 / step29420 / samples7501888, validation flow
0.0053752917. Training completed all40epochs; latest is epoch40 / step36748.
The earlier matched test used epoch10 / step9268 and achieved1/4success.

Training: original10epochs plus30additional full-data epochs,2048query episodes,
234434expert stride3chunks per epoch, batch256, AdamW LR2e-5, BF16, no gradient
checkpointing. Freeze E114 target-mask TCOW and auxiliary ImageNet ViT-B16;
train only FM and its existing joint encoder/context K/V projections. FM was
initialized from reach step115000. No XYZ head/loss or context decoder. Actions:
H25future EE deltaXYZ/relative rotation6D anchored at current measured EE pose,
plus absolute jaw. Full training config is embedded in test config.json.

Same four preselected RedCube cases: g000/p00/t0(train), g000/p01/t0(new order),
g036/p00/t0(new layout), g036/p01/t0(both new). Same post-shuffle initial states,
H25/K10, Euler10, noise seed0 reset per case, maximum900frames at25Hz. Existing
pose IK/servo. Only recorded past context plus live RGB and measured joints feed
the policy; no expert future observations/actions. TCOW30frame prefix sampling
retains context frame0 and current frame; query is the initial visible target
mask only. Reproduce the training int8/block16 and FP16 numeric boundaries.
Static left camera substitutes for a wrist camera. Success: correct cover in
drop zone, correct target lifted>4cm with bilateral contacts10frames, other cover
shift<=15mm; wrong-object lift fails.

Video: predicted target mask over live top RGB, live auxiliary RGB and initial
prompt thumbnail; green GT center is display-only. No XYZ prediction. Video
captures every3frames and holds causally at25FPS; mask updates every10frames,
age labeled. Verify all individual MP4s and montage by complete decoding. A
four-case diagnostic does not estimate general success rate.

Only evaluator metadata handling changes from the previous test: accept a
protocol README, derive epoch/title from the chosen checkpoint, remove the
old epoch10-only assertion. Policy computations, histories, noise, control,
success criteria and scene selection remain unchanged; prior exact evaluator
source is retained in the original test source snapshot.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.shuffle_bc.evaluate_direct --checkpoint /workspace/robotics_assets/experiments/e115_direct/train_2/best.pt --output /workspace/robotics_assets/experiments/e115_direct/test_k10_2 --readme docs/research/e115_rollout_2_README.md
```

Fresh output required. Sources/hashes, fixed checkpoint snapshot and config.json
are recorded beside results.jsonl/summary.json/NPZ/MP4. stdout/stderr tee to
e115_direct/test_k10_2.log, ssh_tmux:1(e115_test2). Canonical results/documentation
remain in local docs/research. Prior results and source checkpoints are retained.

## Matched results

| Case | Epoch10 result | Epoch32 best result | Frames | Max target lift |
|---|---|---|---:|---:|
| Train g000/p00/t0 | Timeout | Success | 329 | 120.95 mm |
| New order g000/p01/t0 | Timeout | Success | 324 | 123.22 mm |
| New layout g036/p00/t0 | Timeout | Timeout | 900 | ~0 mm |
| Both new g036/p01/t0 | Success | Timeout | 900 | 19.38 mm |

Initial qpos, qvel, ctrl and initial prompt masks match the epoch10test bit for
bit. H/K, flow steps, noise seed, case rows, history sampling, numeric boundaries,
frozen fingerprints and scoring definitions match exactly. Paired outcome:
two gains, one regression, one persistent failure;1/4to2/4 successes on this
four-case red-target diagnostic. No population success-rate claim.

Both successful cases deposit the selected cover and sustain target bilateral
lift for10frames, with negligible other-cover shift. In the new-layout failure,
selected cover_a is not lifted; cover_b reaches only33.76mm lift and1.736mm XY
shift. Neither cover action completes and the target is not lifted. In the
combined regression the correct cover is deposited, but target lift peaks at
19.38mm with no qualifying bilateral streak. Both failures time out at900frames.
End-frame masks remain near the target; that qualitative observation does not
isolate the cause of the control failures or establish dense mask accuracy.

All four individual videos and the four-panel montage fully decoded; frame
counts329/324/900/900, montage900,25FPS. Additional epoch10_vs32.mp4 shows the
old four-panel montage on the left and new on the right; fully decoded900frames.
Shorter completed clips hold their last frame in the montage. Rollout plus
original video export/verification took126.30s; peak allocated VRAM1.442GiB.
Frozen vision/checkpoint hashes unchanged. Metadata comparison in paired_check.json;
results summary.json/results.jsonl; local copies e115_rollout_2_results.json and
e115_rollout_2_paired_check.json. The epoch40latest checkpoint was not tested;
selection of epoch32best preceded these test outcomes.
