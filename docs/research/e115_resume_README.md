# E115 continuation: 30 additional epochs

Status: complete, process exit0. All30additional epochs ran, reaching total
epoch40 / step36748 /9,377,360samples. Best is epoch32 / step29420, validation
flow0.0053752917. Latest is epoch40; closed-loop best-checkpoint test completed2/4
in test_k10_2, documented in e115_rollout_2_README.md.
Resume the completed 10-epoch run from
`../train/latest.pt` (epoch10, step9268, next_epoch11). Train epochs11–40,
30 additional full-data passes. Expected final step36748 and total samples
9,377,360, including the original run. Prior checkpoints and tests remain intact.

No architecture, loss, data, normalization or optimizer changes. Frozen E114
target-mask TCOW (epoch9/step4608), frozen ImageNet auxiliary ViT-B16; train FM
and its existing joint encoder/context K/V projections. FM originated from reach
step115000. All300TCOW hidden tokens +64auxiliary tokens +1joint token; no context
decoder or XYZ head/loss. Current left camera substitutes for wrist.

Full2048query episodes /512contexts /234434expert stride3chunks per epoch.
Expert-only masked FM loss. TCOW history includes the past shuffle context using
the exact native30frame prefix sampling and initial visible-mask prompt. No
later masks/object coordinates/task IDs feed the policy. H25/25Hz actions are
future EE deltaXYZ and relative rotation6D anchored at current measured EE pose,
plus absolute jaw. Batch256, encoding batch4, AdamW LR2e-5, clip1, BF16, no
gradient checkpointing. Restore optimizer moments, RNG, sample order and global
step; the original100update warmup does not restart.

Rebuild the same78.37GiB RAM cache (top int8/block16scales, side FP16), about
5–6hours. During this rebuild no previously completed FM updates are repeated.
The existing encoder currently processes cache groups without changing FM, then
continues global shuffled training at epoch11. About8.2min per subsequent epoch,
roughly4.1hours for30epochs; estimated total9–10hours. Cache remains RAM-only and
is released after training exits. Available disk is about35GiB, insufficient for
a full feature-cache export; no disk feature cache is created or retained.

Validation: same fixed32held-out episodes /256expert endpoints, fixed seed115000;
select best by masked validation flow loss. Previous best0.0091387402 is copied
forward and replaced only on improvement. Save latest every256updates and each
epoch, retaining optimizer/RNG/next cursor. Checkpoint values remain explicitly
labeled with total epoch/step. Closed-loop tests of this continuation: not run;
planned matched H25/K10/Euler10/noise0/900frame tests only after a checkpoint is
available. Earlier epoch10 four-case diagnostic was1/4success, not a benchmark
success rate. No success guarantee from reducing training loss.

Code unchanged: OATFlow/shuffle_bc/train_direct.py, direct_model.py, direct_data.py.
Exact prior source hashes are required by the existing resume implementation.
Read-only resume verification is recorded in e115_resume_check.json. Command:

```bash
NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python -u -m OATFlow.shuffle_bc.train_direct --epochs 40 --batch 256 --encoding-batch 4 --resume /workspace/robotics_assets/experiments/e115_direct/train/latest.pt --output /workspace/robotics_assets/experiments/e115_direct/train_2
```

Output must be fresh. Live stdout/stderr in ssh_tmux:1 (e115_40), mirrored to
e115_direct/train_2.log. Cache progress/ETA in train_2/cache_progress.json;
epoch/update losses later in metrics.jsonl and progress.json. Source snapshots,
full configuration, validation logs and best/latest checkpoints accompany output.

Startup verified: exact restored model and optimizer values,185optimizer slots,
LR2e-5, matching source hashes. Main process passed its strict resume checks,
copied the previous best checkpoint, and started train-cache encoding (first
108of234434chunks confirmed). No epoch1updates are repeated during rebuild.
`progress.json` and `resume_state.json` explicitly retain epoch10/step9268 and
next_epoch11 during this stage; per-microbatch cache count/ETA is in
`cache_progress.json`. These are startup records; final results follow.

## Completed continuation

Actual elapsed after initial validation:35,474.43s (9h51min14s), including
20,616.76s (5h43min37s) rebuilding the frozen cache. Remaining training/validation
and checkpoints took approximately4h7min38s. Median update0.52853s; peak allocated
VRAM15.029GiB. Frozen vision fingerprints unchanged. The process exited normally,
releasing the RAM feature cache. Each of epochs11–40 visited234434chunks once.

| Total epoch | Mean training flow loss | Fixed validation flow loss |
|---|---:|---:|
| 11 | 0.007684 | 0.009679 |
| 20 | 0.005802 | 0.007897 |
| 30 | 0.004835 | 0.007086 |
| 32 (best) | 0.004641 | 0.005375 |
| 40 (latest) | 0.004253 | 0.006501 |

Previous epoch10 best validation was0.0091387402; the new best is41.18%lower.
Later epochs did not improve on epoch32. These supervised flow losses do not
establish better closed-loop grasping. Best/latest metadata and finite parameters
checked after exit. Full per-epoch results: e115_resume_results.json and artifact
train_2/{summary.json,metrics.jsonl,validation.jsonl}. Subsequent best-checkpoint
rollout uses the same four preselected RedCube states/protocol as epoch10, not a
checkpoint selected by test outcomes. Epoch32best achieved2/4successes versus
epoch10's1/4, with two gains and one regression. This small diagnostic does not
establish a population success rate; see e115_rollout_2_README.md.
