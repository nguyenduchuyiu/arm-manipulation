# Code reuse audit after artifact consolidation

Read-only audit of current repository against the curated robotics_assets catalog.
42workspace_analysis Python modules and11tracking Python modules inspected.
This audit identifies obsolete entry points and shared dependencies; no source
files were deleted or implementation changed by the audit.

## Current reusable implementation

- OATFlow/tracking/point.py: both retained PointXYZ variants + official TCOW load.
- OATFlow/tracking/model.py: CausalBackbone/Attention/MLP/PatchEmbedding/SpaceTimeBlock
  are required by PointXYZ. AmodalXYZ/CrossViewFusion are historical RGB3+ID branches.
- OATFlow/tracking/point_data.py,data.py: causal prefix loading,initial dot,projection,
  split selection and evaluation plan. The old three-camera clip is separate.
- OATFlow/tracking/train_point.py,point_video.py,pick_video.py: actual point training,
  evaluation/overlays and corrected live visual-FM rollout.
- workspace_analysis/vendor/e88_policy.py + oatflow_model.py: strict retained10D FM
  inference architecture and underlying flow network. ee_pose.py: pose conversion,
  gripper conventions,measured anchor semantics and bounded IK.
- workspace_analysis/object_xyz.py: retained frozen ViT and XYZ head used by e99.
- collect_shuffle.py,render_shuffle.py,audit_shuffle.py,shuffle_data.py and
  grid_dataset.py,clean_scene.py,ee_chunk_dataset.py,joint_state_dataset.py,audit_dataset.py:
  data creation/export/audit functionality to retain and adapt for canonical paths.
- OATFlow/dataset/expert.py,OATFlow/environment/*,OATFlow/task.py,controllers/*
  provide the physical contact expert and scene/control dependencies for shuffle.

## Candidates to remove or replace after extracting shared code

1. fm_xyz*.py and fm_ee.py,fm_ee_train.py,fm_ee_eval.py: historical e78/e79/e83
   4D deltaXYZ+gripper route. fm_ee_video.py and fm_ee_boxes.py reference that API
   while expecting newer kwargs/10Dmodels and are not reliable retained-model entry
   points. Keep E88Policy and build a10Dtrainer around the actual retained architecture.
2. OATFlow/tracking/train.py,evaluate.py,run.sh and AmodalXYZ/CrossViewFusion branches:
   old3camera+ID experiment. train_point imports save/summary/write_json from train.py;
   extract these before deletion. Keep the common causal backbone/data functions.
3. e2e_reach.py: old e94/e55/e91 static3camera pipeline, deleted dependencies.
   pick_video still imports step/add_marker/path_overlay from it; extract these
   helpers first. replay_ee_pose.py also mixes the required visual_model helper
   with an old one-off replay CLI. render_shuffle imports encoder from rerender_cam3;
   extract the encoder and camera constants before retiring old camera wrappers.
4. multi_ik.py,multimodal_dataset.py,kinematics_encoder.py and old FM/camera comparison
   wrappers belong to deleted experiments; no current curated inference entry point
   depends on them directly. Preserve a relevant utility only if regeneration uses it.
5. Static3view training/evaluation wrappers train_object_xyz*.py,preview_cam3_dataset.py,
   audit_cam3_dataset.py,render_cam3_dataset.py depend on deleted e55/e91/e92/e93/e94
   assets. object_xyz.py remains useful. infer_object_xyz.py currently requires3images
   and defaults to deleted e92; it needs a true retained-top-view inference entry point.
6. Tests of old multiview/ID branches and READMEs/run.sh describing removed paths
   should follow the implementation cleanup. Keep point-query causality/gradient,
   normalization/anchor,actual checkpoint-load and command-replay coverage.

## Confirmed reuse gaps

- collect_shuffle.py initialize_output line299 copies /workspace/datasets/e91_cam3/
  scene.xml,which no longer exists. Use the retained shuffle scene/robot assets
  when adapting regeneration; do not delete the generator wholesale.
- fm_ee_train.py imports EEDeltaFlow from fm_ee.py; its action_dim is4,whereas
  E88Policy has action_dim10. It cannot train/resume the retained e88 checkpoint
  as currently written. This explains why catalog only promises retained optimizer
  state,not a revalidated original FM training entry point.
- fm_ee_video.py calls load_data(joint_states=...,goal_only=...) but the currently
  imported fm_ee_eval.load_data accepts no arguments. That historical wrapper is
  stale independently of the data-path relocation.
- infer_object_xyz.py nargs=3/defaulte92 conflicts with selected top-only e99 model.
- Point training still references e100/e101 research docs and defaults to an existing
  curated model directory. A fresh --output is required; remove experiment-template
  dependencies when creating a reusable training interface.

OATFlow/prior,rl,pick and policy include independent project work and modified user
files; their absence from the selected runtime graph alone does not justify deleting
those entire modules. They are outside this targeted reuse audit's deletion list.
No additional training or tests were run for this read-only code audit.
