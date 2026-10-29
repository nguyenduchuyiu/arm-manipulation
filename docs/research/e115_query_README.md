# E115 fixed-state target-query probe

Status: complete. Purpose: distinguish query insensitivity from
general trajectory errors at the initial post-shuffle action state. No training
or closed-loop motion occurs in this probe.

Use E115 best epoch10/step9268, with the exact frozen E114 TCOW and auxiliary
ViT. Training: full2048episodes,10epochs,batch256,LR2e-5,expertstride3,FM-only
masked flow loss. Full training metadata is embedded in config.json.

Four contexts: g000/p00, g000/p01, g036/p00, g036/p01, one for each evaluation
split. For each context hold top RGB history, current top/left RGB, measured
joints and FM noise fixed. Assert recorded initial qpos/joints are identical
across all four expert target demonstrations. Decoded RGB can differ slightly
because each video is encoded separately; record those differences and use
the target0 RGB for all four query inputs. Vary
only the initial visible target mask; recompute TCOW hidden. Use native30frame
causal history, training int8/FP16 boundaries, H25, Euler10, noise seeds0–3 shared
across queries. K/frame budget: not applicable, no action execution.

Read expert future actions and GT masks only as diagnostic labels. Compare mask
binding by largest IoU among the four GT masks; compare sampled H25 deltaXYZ
against recorded expert trajectories. Report correct-query versus opposite-cover
query XYZ trajectory RMSE, query-induced separation, and cover agreement of the
nearest expert trajectory. Two targets share each cover, so initial cover actions
cannot distinguish those targets; this is a two-cover action-binding probe.
Expert references are demonstrations, not unique optimal trajectories. Shared
image/state/noise query swaps isolate prompt sensitivity here, not all later
action phases. Four scenes and four seeds do not estimate task success.

Code: OATFlow/shuffle_bc/probe_direct_query.py. Command:

```bash
.venv/bin/python -u -m OATFlow.shuffle_bc.probe_direct_query --output /workspace/robotics_assets/experiments/e115_direct/probe_query_2
```

Fresh output required. Config/checkpoint hash, exact source snapshot, trajectory
arrays and full results are stored beside this README. No large feature cache.

## Results

At the 16 fixed initial query states, TCOW mask binding was correct 16/16;
mean target-mask IoU was 79.32%. These are initial-state probes, not dense-video
tracking results. Across four shared noise seeds, mean sampled H25 XYZ trajectory
RMSE against the correct expert was 8.365 mm using the correct query and 16.746 mm
when substituting an opposite-cover target query. Query information does affect
the policy; it is not entirely ignored.

Nearest-expert cover agreement was 46/64 sampled query/seed combinations.
These 64 samples share only four physical contexts and are not independent task
trials or a rollout success rate. Across opposite-cover target pairs, sampled
trajectory separation was 7.638 mm versus 16.255 mm for expert trajectories.
This is compatible with incomplete target-dependent action binding, not a unique
diagnosis of how attention or training fails.

A specific failure: train g000/p00 target2 (GreenCube) has 94.51% target-mask IoU
and correctly bound mask, but all four sampled trajectories are closer to the
other cover's expert trajectory. For unseen-layout g036/p00 target0 (RedCube),
mask IoU is 77.84% yet all four seeds also favor the wrong cover; this agrees
with the earlier closed-loop wrong-cover action. The correct expert trajectory
is a reference, not the only valid action plan.

The probe supports partial but unreliable query use at initial expert states.
It does not diagnose the later cover-to-object transition or prove that more
training will solve it. Epoch10 remains the lowest validation flow loss, making
additional training a plausible next test rather than an established fix.

Frozen vision fingerprints and checkpoint hash were unchanged. Initial qpos
and joints matched exactly across all query demonstrations. First probe attempt
stopped before model outputs because separately encoded videos do not decode to
bit-identical RGB. Corrected protocol fixes one shared RGB input across queries
and records codec differences (mean absolute pixel differences below0.6 on the
0–255 scale). The failed folder/log was removed after recording this cause.
Completed artifacts: e115_direct/probe_query_2/{config.json,summary.json,traces.npz,
source.py}; local copy e115_query_results.json. Exit0.
