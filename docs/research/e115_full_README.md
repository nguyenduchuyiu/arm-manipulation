# E115 best: full held-out closed-loop test

Status: sequential run interrupted after3committed cases when the user requested
faster execution and no videos. Active full test is the equivalent four-process
no-video run in test_full_2; see e115_parallel_README.md. Its main results are
pending. Test the validation-selected E115
train_2/best.pt, epoch32/step29420, validation flow0.0053752917. This is the same
fixed checkpoint as test_k10_2, not a new selection based on test outcomes.

Training: full2048query episodes,234434expert stride3chunks/epoch, total40epochs
(10initial+30resumed), batch256, AdamW LR2e-5, BF16, no gradient checkpointing.
Freeze E114 epoch9 target-mask TCOW and ImageNet auxiliary ViT-B16. Train only
FM and existing joint encoder/context K/V projections, initialized from FM
trained on reach at step115000. No context decoder or XYZ head/loss. H25future
EE deltaXYZ/relative6Drotation anchored at current measured EE pose plus absolute
jaw. Full training metadata is embedded in config.json.

Evaluate ALL1,408held-out test episodes,352shared contexts,352queries per target:

| Split | Episodes | Shared contexts | Queries per color |
|---|---:|---:|---:|
| Unseen permutation / composition | 1024 | 256 | 256 |
| Unseen layout / spatial | 256 | 64 | 64 |
| Both unseen / combined | 128 | 32 | 32 |

Exclude train and validation. Read every test row directly from the dataset
manifest, not the old128case subset. Freeze the full case list before outcomes.
Run the first shared context of each split, all4targets, first for qualitative
previews, then every remaining row in manifest order. No case or target omitted.

Protocol: H25/K10,10Eulerflow steps, noise seed0 reset independently per episode,
900frame limit /36s at25Hz. Same recorded post-shuffle initial physical state,
pose IK/servo and success criteria as the prior four-case test. Success requires
correct cover deposited in its drop zone; target lift>4cm with bilateral jaw
contact10frames; other cover XY shift<=15mm. Wrong-object lift fails. Sample one
noise seed per test episode; rates characterize this protocol on this finite
held-out set, not robustness over repeated noise seeds or real robots.

Observation: recorded past shuffle context, initial visible target mask only,
then live top RGB history, current left RGB and measured joints. TCOW uses the
same30frame causal prefix sampling, context/live stride3plus terminal, exact
current frame at K10replan. Native BF16vision, int8/block16TCOW roundtrip and
FP16auxiliary boundary match training. Left view remains a static wrist
surrogate. No future expert RGB/actions or later GT masks/object coordinates
feed the policy. GT state is used for scoring and display-only overlays.

Store state/command/action/mask/history NPZ traces and per-case JSON results for
all1,408episodes. Render12videos selected before outcomes (the first context in
each split, all4colors), plus three split-specific four-target montages. All
exported videos fully decode for verification. Most cases omit video export;
both camera render schedules are retained exactly. Top history and policy inputs
are unchanged. The no-video path is
checked against both a prior success and900frame failure, comparing all trace
arrays bit-for-bit. Essential preflight checks: e115_full_preflight_check.json.

Run sequentially to preserve the exact single-case numerical path/noise of the
prior test. Progress JSON includes completed/total cases, success counts, ETA,
and per-split/per-target rates after each completed case; interim rates are
incomplete and affected by deterministic preview-first ordering. Atomic
results.json is the completion record; results.jsonl is a readable journal.
Resume verifies checkpoint, source, scene/manifest hashes and full protocol,
skips completed cases, and reruns an unfinished episode from its initial state.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.shuffle_bc.evaluate_direct --checkpoint /workspace/robotics_assets/experiments/e115_direct/train_2/best.pt --output /workspace/robotics_assets/experiments/e115_direct/test_full --readme docs/research/e115_full_README.md --full-test
```

Fresh output required. To resume the same incomplete output, use the identical
command with --resume; preserve code and checkpoints. The exact checkpoint
snapshot, source snapshots and hashes accompany config.json. Case traces should
require around1GiB and selected videos far less than the33GiB free at launch.
Cache is per-episode history only; no full training feature cache is rebuilt.
stdout/stderr tee to test_full.log in ssh_tmux:1(e115_full). Final summary.json
reports total, split, target, failures and cover deposits; full results pending.

## Verification and timing

No-video rollout matched every saved NPZ array bit-for-bit on g000/p01/t0
(324frames,11.938s) and g036/p00/t0 (900frames,30.409s), including initial state,
commands, qpos/qvel/ctrl, sampled actions, predicted masks and history timestamps.
Both physical outcomes also matched. Frozen fingerprints stayed unchanged.
One earlier attempt skipped some auxiliary camera renders and diverged slightly
(first case completed in323rather than324frames); that optimization was rejected.
Retaining the original rendering schedule restored exact trace agreement.
The failed/preflight trace copies and temporary scripts are removed after
recording these checks. No preflight outcomes are imported into the full test.

Measured upper-envelope estimate is about12hours if most cases time out; with
successful cases finishing earlier, initial working estimate8–12hours. No
large image/feature cache is built; each episode runs new closed-loop physics
and perception. Full per-group/target success rates will be reported only after
all1,408episodes have completed.
