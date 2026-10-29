# E115 full held-out test: independent processes, no videos

Status: the900frame run was interrupted after130committed cases when the user
reduced the horizon to400frames. Active continuation is test_full_3; see
e115_400_README.md. The four-process parity/speed preflight passed. User
requested no video export. Full target is all
1408held-out episodes:1024unseen permutations,256unseen layouts,128both,352queries
per target. Same fixed best epoch32/step29420 as the prior four-case test.

Training: full2048episodes,234434expert stride3chunks/epoch,total40epochs,
batch256,AdamW LR2e-5,BF16,no checkpointing; freeze E114TCOW and auxiliary
ImageNet ViT-B16; train FM/joint encoder/context K/V, originally from reach115000.
No context decoder/XYZ head. Full training metadata in config.json.

Protocol remains H25/K10/Euler10/noise0reset per episode/900frames25Hz, initial
visible mask plus causal30frame top history, current left RGB/joints; original
int8/block16 and FP16 boundaries. Preserve both camera render schedules. Actions
are current-EE-anchored deltaXYZ/relative6Drotation plus absolute jaw. Strict
success: correct cover deposited, target bilateral lift>4cm for10frames, other
cover shift<=15mm; wrong-object lift fails. No GT target state or expert future
feeds the policy. Left is a static wrist surrogate.

Parallelize independent episodes in separate spawned processes, each retaining
the original single-case tensor shapes and seed handling. No batched-policy
numerical change. Four workers fit GPU/RAM/CPU capacity. Benchmark compares all
saved traces bit-for-bit to four known epoch32rollouts before full launch.
Preflight includes the prior train diagnostic solely for parity verification;
full test excludes train/validation. Preview footage and all video export are
disabled. Keep per-case state/action/mask/history NPZ, JSON outcomes and progress.

Reuse committed cases from the interrupted sequential full test only after
checking checkpoint, physics/manifest hashes and protocol. Import traces and
original outcomes without copying videos. Queue remaining cases in deterministic
round-robin split order; completion order reflects independent worker runtimes.
Aggregate JSON rates during execution are incomplete; final summary covers the
exact full manifest set. Results.json updates atomically after each completed
episode. --resume skips committed cases and reruns any unfinished case from its
initial state. Source, model and full protocol must match for resume.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.shuffle_bc.evaluate_direct_parallel --output /workspace/robotics_assets/experiments/e115_direct/test_full_2 --workers 4
```

Preflight uses the same module with --preflight and a fresh diagnostic output.
Main stdout/stderr from parent and workers tee into test_full_2.log in ssh_tmux:1.
Frozen module fingerprints verified in every worker before/after its work.
Final full results pending. Canonical protocol/results: local docs/research;
artifact config.json/results.json/results.jsonl/progress.json/summary.json.

## Measured verification

Four independent-process rollouts match every saved reference trace array
bit-for-bit, including both prior successes (329/324frames) and both900frame
timeouts. All outcomes match. Four-worker benchmark compute42.453s,
including initialization wall50.328s;2453total simulated frames. Sequential
no-video check measured1224frames in42.348s. Aggregate measured throughput
approximately57.8versus28.9simulated frames/s, about2×. Initial full-run estimate
4–6hours, depending on case completion lengths. Peak observed total GPU use
about8.5GiB at100%utilization. CPU quota23.04cores supports4×4torch threads.
Checkpoint/frozen fingerprints unchanged. Essential check recorded locally in
e115_parallel_check.json; completed diagnostic checkpoint/trace folders removed.

The interrupted original sequential run had3committed episodes. Their valid
outcomes/traces are imported after protocol/hash verification; no videos copied.
Only1405remaining episodes are queued. New active artifact directory test_full_2;
original test_full remains as an interrupted-run record. This process exports
no videos at all. Full results remain pending.
