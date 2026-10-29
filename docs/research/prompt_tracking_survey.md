# Point-prompt tracking for three-view amodal centers

Survey date:2026-10-15. Research only; no model change or optimizer run. Current
code still uses color-ID queries. The attempted initial-XYZ input was reverted.

## Primary sources and mechanisms

| Method | Prompt mechanism | Application limit |
| --- | --- | --- |
| TAPIR | `get_query_features` interpolates query-frame low/high-resolution feature grids; global matching and local-correlation refinement follow. | Tracks a surface point, not object center. Official Online TAPIR is causal; full-video TAPIR is a separate mode. |
| CoTracker3 | `get_track_feat` samples query features and support neighborhoods at multiple feature-pyramid levels. | Local support is useful for constructing appearance prompts. Window-based online processing alone does not guarantee strict per-output causality. |
| SAM2 | Click coordinates plus positive/negative embeddings become sparse prompt tokens; decoder attention obtains visual evidence. Memory and object pointers support videos. | Segmentation model with mask-based training/memory; not an XYZ-only tracker. |
| SAM-PT | Propagates first-frame points and uses their trajectories to prompt SAM masks. | Demonstrates point-prompted object segmentation but adds a mask pipeline. |
| TAPNext/TAPNext++ | Query-coordinate token at query time, learned token otherwise; spatial attention exchanges image/track information and recurrent temporal processing carries state. | Causal architecture precedent. TAPNext++ improves long sequences and re-detection through training changes, including occluded-point supervision. Surface-point results do not establish hidden object-center performance. |
| MV-TAP | Camera-aware view attention joins multi-view point tracks. | Uses camera geometry and primarily queries per view. Appendix A's feature-based cross-view initialization searches all video frames; this cannot initialize our causal rollout unchanged. |
| MVTracker | Camera poses and depth lift features into a shared3D point cloud, with kNN correlations and iterative tracking updates. | Requires more than the present fixed RGB-only rig; tracks physical points. |

Sources inspected:

- https://github.com/google-deepmind/tapnet/blob/main/tapnet/models/tapir_model.py
- https://github.com/google-deepmind/tapnet
- https://github.com/facebookresearch/co-tracker/blob/main/cotracker/models/core/cotracker/cotracker3_online.py
- https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam/prompt_encoder.py
- https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py
- https://arxiv.org/abs/2408.00714
- https://github.com/SysCV/sam-pt
- https://arxiv.org/html/2504.05579v2
- https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapnext/tapnext_torch.py
- https://arxiv.org/abs/2604.10582
- https://arxiv.org/pdf/2512.02006 (problem formulation, Section3.3, Appendix A)
- https://github.com/cvlab-kaist/MV-TAP
- https://github.com/ethz-vlg/mvtracker
- https://github.com/basilevh/tcow#evaluation (actual first-frame query is a mask)

## Proposed adaptation, not implemented

Input is RGB history, timestamps and `(camera_index,u,v)` on frame0, where the
click selects a visible object. Output remains that object's amodal base-frame
XYZ center. The click need not be the projected center or supply depth.

The current ViT-B/16 maps240x320 RGB to15x20 patch features of width768. Reshape
tokens to `[B,768,15,20]` and bilinearly sample at the click. This combines
neighboring learned descriptors, not raw RGB values. Pixel resize and feature
cell-center conventions must be handled explicitly.

Retain a small local neighborhood plus the center feature, add coordinate/view
and local-offset embeddings, and project to fusion width256. A neighborhood is
not a GT segmentation mask. Small objects can occupy less than a16px patch, so
coarse descriptors may mix background; high-resolution features or local crops
are possible ablations, not demonstrated solutions.

Keep initial reference descriptors as an identity anchor and give target tokens
a causal temporal state. Current fusion latents are independent per timestamp;
temporal processing currently lives in patch features. Let the target state read
spatially encoded evidence from all three views and feed the existing XYZ head.
An alternative inspired by TAPNext injects a coordinate-only prompt token into
frame0 spatial attention and learns appearance extraction there. Neither option
has been compared on e96.

One click does not automatically locate the same object in the other cameras.
Wide-baseline appearance association is an additional learned problem; MV-TAP
explicitly documents unreliable single-view query transfer. For the fixed rig,
joint XYZ-supervised attention is a plausible experiment, not verified geometry.
One object-selection click per view is a useful easier comparison. Such clicks
need not represent the same surface point for object-center prediction, unlike
multi-view physical-point triangulation. New-camera generalization is unproven
without camera geometry.

If all cameras lose the object, appearance matching supplies no current center
measurement. History and learned object/container motion must support prediction.
Identical observations with different hidden motions cannot be uniquely resolved
without additional assumptions. Evaluate shuffle/hidden and reappearance phases,
not only opened-cover localization.

## Existing data and checks

e96 supplies shared contexts for four queries and per-frame base-center labels.
Add explicit frame0 point-query annotations. Synthetic frame0 geometry may
generate2D query annotations offline; only2D coordinates enter the network, and
GT XYZ remains loss/evaluation supervision. No GT mask inputs/loss are required.
Do not use future labels or images to choose prompts or cross-view matches.

The present opening-T1 curriculum must change: include frame0 plus the opening
frame, or provide an explicitly cached frame0 prompt feature. Keep frame0 when
truncating/resampling history. Current whole-prefix sampling has about1.46Hz
median cadence at opening; a new prompt cannot restore missed swaps.

Validate same RGB/different object clicks, different clicks on one object,
coordinate conversion, and zero future RGB/time/label influence. Prompt-only
and frame0-only baselines expose shortcuts. Keep existing layout/permutation
splits and report hidden/shuffle, opening and reappearance errors separately.
No existing ID-query or mask-assisted checkpoint validates this proposed design.
