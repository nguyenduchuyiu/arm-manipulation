# E115 full held-out test: 400-frame horizon, no videos

Status: running in ssh_tmux:1(e115_f400), all4workers active with400frame limits.
All130committed prefixes imported; full results pending.
User explicitly reduced the timeout from900to400frames (16s at25Hz). Preserve
all1,408held-out cases and checkpoint selection: best epoch32/step29420, not
latest epoch40. Full test:1024composition,256spatial,128combined;352queries per
target. No train/validation episodes counted. Four independent processes,
single-case inference, no video exports.

Training unchanged: full2048episodes,234434expert stride3chunks/epoch,total40
epochs, batch256, AdamW LR2e-5, BF16. Freeze E114 mask TCOW/ImageNet auxiliary
ViT; train only FM/joint encoder/context K/V, originally initialized from reach
step115000. No context decoder or XYZ loss/head. Full config embedded in test
config.json. H25current-anchor EE deltaXYZ/relative6Drotation/absolute jaw.

Test: H25/K10/Euler10/noise0reset per episode, same initial physical states,
causal30frame top-history sampling/initial visible mask, current auxiliary RGB
and joints. Same original camera render cadence and int8/block16/FP16 feature
boundaries; left camera is a static wrist surrogate. Only horizon changes.
Success still requires correct cover deposited, target lift>4cm with bilateral
contacts10frames and other-cover shift<=15mm. Wrong-object lift fails when it
occurs within400frames. No expert future or GT target coordinates feed policy.

Interrupt the900frame full run test_full_2, retaining its130committed episodes
and original traces/config/source snapshot. Import every committed trajectory's
first min(original_stop,400)steps into fresh test_full_3. Existing success or
wrong-object stops within400retain outcomes. A later stop becomes timeout400:
the original evaluator would have already stopped if a qualifying event had
occurred earlier. Recompute cover-deposit/other-cover displacement at the clipped
endpoint with MuJoCo, and lift maxima/contact streak from the prefix. Trim
qpos/qvel/ctrl to n+1, controls/objects/contacts to n, and replan/mask/history
arrays to replans before n. Preserve original stop/reason and source provenance.
Original900frame results are not overwritten or mixed with endpoint400metrics.

Validate re-scoring against fresh400frame rollouts of four reference cases,
requiring all prefix arrays bit-identical and physical metrics/outcomes equal.
Original source hashes must match for every model/data/physics file; only two
evaluator control files change for horizon/import handling. Scene/manifest and
checkpoint hashes must match. Continue only remaining episodes. Atomic
results.json retains completed cases; --resume restarts only unfinished cases.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.shuffle_bc.evaluate_direct_parallel --output /workspace/robotics_assets/experiments/e115_direct/test_full_3 --workers 4 --frame-budget 400 --import-results /workspace/robotics_assets/experiments/e115_direct/test_full_2
```

Fresh output required. Add --resume to that exact command for an interrupted
400frame run. Files: config.json, checkpoint.pt, source snapshots, results.json/
results.jsonl, per-worker progress/NPZ traces, summary.json. Live log test_full_3.log
in ssh_tmux:1(e115_f400). Canonical research notes remain in docs/research.
Full400frame results pending; earlier900frame diagnostic rates are distinct.

Verification: four fresh400frame reference rollouts reproduce every clipped
900frame trace array bit-for-bit and match all re-scored physical metrics and
outcomes. Their compute wall time was23.572s, no video exports, frozen/source
checkpoint unchanged. Separate audit re-scored all130committed episodes:47
successes remain47, and one wrong-object lift originally occurring at575frames
becomes timeout400. No success had occurred after400in the completed prefix.
All900frame originals remain intact. Additional change cancels unused task-queue
feeder joins on interruption so stopping a partial full test does not hang.

Working ETA for remaining1278episodes is approximately2–3hours at four workers;
live estimate follows actual completion lengths. Evidence:
e115_400_preflight_check.json and e115_400_migration_check.json. Completed
diagnostic folders/checkpoints/trace duplicates are removed after recording
these checks; their outcomes are not imported as new test rows.
