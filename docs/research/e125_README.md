# E125 — Original MobileSAM with one positive point

Status: COMPLETE.

Diagnostic: distinguish visible-object segmentation with known location from
learned task-conditioned localization. Entire model is the original official
MobileSAM `vit_t` checkpoint (TinyViT, prompt encoder, mask decoder unchanged).
No E123/E124 learned weights, task tokens, training, optimizer, epochs or learning
rate. All parameters frozen; inference CPU FP32 in3workers x8threads so the
ongoing E124 CUDA training process is left running.

Input: front RGB and one positive2Dpoint for each sufficiently visible target.
Choose the pixel deepest inside its visible GT instance mask by a distance
transform, guaranteeing the point lies on the requested object. Only that point
is passed to the model; no mask input, box, depth, joints, XYZ or semantic task ID.
No negative points or corrective clicks. This is an oracle prompt independently
provided at every evaluated frame, not tracking or automatic localization.

Data: exactly E124 epoch1 temporal validation records:256original validation
episodes,8uniform frames each,4queries/frame=8192queries/2048images. The same
8160queries with at least8GTpixels contribute to all model comparisons. The
32low-visibility queries are explicitly skipped because a useful visible point
is not defined; they are never counted as successful.

Use original SAM PIL longest-side1024resize, coordinate transform, prompt encoder,
four mask readouts and native240x320postprocessing. Report two predefined modes:
single original mask head0 (same branch E124 trains); original multimask heads1–3
with the highest model-predicted quality. GT IoU never selects a candidate.
The one-pass four-head outputs are checked against both original public
SamPredictor modes on one image per episode. Entire model state hash is checked
unchanged after every episode, along with source checkpoint file hash.

Metrics: mask IoU/Dice, task identity by highest overlap against four GT instances,
home/later frames, per-target confusion, paired correctness against E124 epoch1.
H/K, flow steps, action representation, noise seed and rollout budget are not
applicable; no manipulation rollout or grasp success is measured.

Artifacts: config.json, progress.json, single/{results.jsonl,predictions.npz,summary.json},
multimask/{results.jsonl,predictions.npz,summary.json}, paired.json,
summary.json, verification.json, comparison.png and point_failures.png.
Comparison sheet deliberately selects task-token failures by target/home-later
phase; point_failures selects the worst original quality-selected case per target.
These images illustrate failure modes rather than a random sample.

Run: `.venv/bin/python -u -m OATFlow.task_mask.point_evaluate` from the repository,
inside tmux with stdout/stderr piped through tee. Output:
`/workspace/robotics_assets/experiments/e125_point`; exact checkpoint/code/reference
hashes and settings are in config.json. Reference:
`e124_mask_full/validation/epoch_1/results.jsonl`. Results: complete; see below.


## Results

| Configuration | Mean mask IoU | Identity accuracy | IoU >= 0.5 |
|---|---:|---:|---:|
| E124 learned task token, epoch1 | 71.02% | 82.13% | 77.67% |
| Original MobileSAM + positive point, single mask | 87.44% | 99.88% | 98.06% |
| Original MobileSAM + positive point, quality-selected multimask | 77.31% | 99.60% | 98.64% |

Matched8160visible validation queries. Single-point single-mask correct8150/8160; 10identity failures and158masks below IoU0.5. Home identity1024/1024; later7126/7136. Mean precision88.15%,recall98.95%. Original weights were unchanged; all256episodes verified against both original SamPredictor modes. Point prompting corrected1449/1458E124 epoch1 identity failures, with1reverse regression.

Original MobileSAM can segment these objects well given their visible location. This supports a bottleneck in the learned task-conditioned localization/decoding pipeline at epoch1, rather than a general inability to segment the objects. Both prompt information and decoder weights differ between comparisons, so this does not isolate a single component or establish E124 convergence.

Single-mask output has higher mean IoU than quality-selected multimask; a visible example selects one cube face instead of the full instance. Worst original-point examples involve very small visible fragments near the robot and can select arm/background regions. Identity accuracy alone is insufficient to certify a clean target mask; inspect IoU and analysis.json.

The positive point is independently derived from GT at each frame. This is not autonomous target selection, one-time prompting, tracking through occlusion, or a manipulation rollout. The32queries below8visible pixels are skipped in every matched comparison. E124 CUDA training continued during this CPU evaluation.

Figures: comparison.png (selected E124 failures by phase/color), point_failures.png (worst original quality-selected prediction per color), pilot.png (first E124 home identity error per color). Packed masks and all query-level results retained; no model checkpoint or encoder cache was created. Derived checks: analysis.json.
