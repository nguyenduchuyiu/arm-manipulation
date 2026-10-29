# E126 — Unprompted automatic MobileSAM proposals

Status: COMPLETE.

Question: does original MobileSAM automatically propose complete, distinct masks
for the four pick objects, before a task selector is added? Entire official
MobileSAM checkpoint remains frozen. No E123/E124 learned weights or training.
Input is RGB only; the original SamAutomaticMaskGenerator samples a fixed uniform
32x32point grid. No GT-derived prompts, boxes, masks, task IDs, depth or joints.

Selection fixed before inference:4original validation layouts32–35,4source
trajectory targets,first2available permutations per layout/target=32episodes.
Evaluate frame0 and the last approach/engage/lift frame=128images. All four GT
object instances are assessed in each image; visible instances require8pixels.
This is a balanced diagnostic pilot, not a full-dataset evaluation.

Use official defaults:1024grid prompts/image,64prompts/batch,multimask output,
predicted quality threshold0.88,stability threshold0.95/offset1,box NMS0.7,
no crops or small-region postprocessing. Original PIL1024resize and mask
postprocessing; CUDA FP32. Original checkpoint file and entire model state are
verified unchanged. No task-specific mask filtering or output selection.

GT is used only after generation to measure proposal coverage: best proposal
IoU per object and distinct one-to-one matching recall at IoU0.5/0.75. Matching
maximizes the number above threshold before total IoU. Report how often every
visible object, and separately all four objects, have distinct adequate masks.
Proposal coverage is an oracle upper bound before task selection, not semantic
instance recognition or task accuracy. Retain all generated masks, including
robot/table/background regions, parts and redundant object candidates. Such
non-target masks are not automatically errors for a class-agnostic generator.

Artifacts: selected_jobs.json,config.json,results.jsonl,progress.json,summary.json,
masks/{packed npz,proposal metadata json},sample_{home,approach,engage,lift}/,
worst/,verification.json. Figures: all_masks.png (all proposals with IDs),
proposals.png (proposal gallery),object_coverage.png (oracle best match for each
GT object, orange=GT/cyan=prediction). Matching labels are evaluation only.

Run `.venv/bin/python -u -m OATFlow.task_mask.automatic_evaluate` in repository
tmux, combined stdout/stderr through tee. Output:
`/workspace/robotics_assets/experiments/e126_auto`. Code:
`OATFlow/task_mask/automatic_evaluate.py`. Exact settings and hashes:config.json.
No optimizer, learning rate, training epochs, actions, H/K, flow steps or
manipulation rollout; zero training updates. Results complete; see below.


## Results

| Phase | Images | Visible objects | Distinct recall IoU >=0.5 | Distinct recall IoU >=0.75 |
|---|---:|---:|---:|---:|
| home | 32 | 128 | 100.00% | 78.12% |
| approach | 32 | 128 | 97.66% | 87.50% |
| engage | 32 | 128 | 97.66% | 82.81% |
| lift | 32 | 126 | 95.24% | 84.92% |
| All | 128 | 510 | 97.65% | 83.33% |

Original generator only received RGB. GT matching is evaluation after generation. Of510visible object queries,498have a distinct proposal with IoU>=0.5 and425with IoU>=0.75. Mean best proposal IoU85.50%. Of126images with all4objects visible,117(92.86%)cover all4at0.5 and54(42.86%)at0.75. Two object queries below8visible pixels were excluded explicitly.

Mean31.125proposals/image,range20–39; mean23.73proposals/image have no visible-object match at0.5 (background/robot/parts and other proposals, not all necessarily generator errors). Mean inference1.685s/image on RTX3090,FP32; this includes dense1024point generation/filtering/NMS but excludes RGB decoding and figures.

The generator usually contains useful object candidates, but output is not a clean four-instance partition. Some candidates cover one cube face instead of the full instance; some partially occluded objects are missed. Home coverage at0.5 is128/128objects and32/32complete scenes, but at0.75 only9/32home scenes have all4complete masks. This balanced pilot is not a full-dataset estimate; no semantic task selection or tracking was tested.

All source weights/model-state values verified unchanged. All128packed proposal counts and pixel areas verified against saved metadata. Analysis:analysis.json; explicit figure selections:figure_selection.json. Inference source hash and visualization source hash are separate in config.json; visualization-only changes label <8pixel targets as excluded, regenerated from the same saved predictions without inference.

Examples: sample_home/all_masks.png shows every raw proposal with IDs; sample_home/object_coverage.png shows offline best matches to4GTinstances. worst/object_coverage.png illustrates partial-face proposals; missed_visible/object_coverage.png illustrates missing candidates among visible objects. Best-match color labels are oracle evaluation annotations, not model class predictions.
