# Real-time trackers for the shuffle manipulation pipeline

Source check: 2026-10-18. Survey of original papers, project pages and official repositories. No alternative model was installed, trained or benchmarked on this instance. Earlier architectural survey: [tracking_architecture_survey.md](tracking_architecture_survey.md). Its installation/status statements describe that earlier point in the project; the current e111 pipeline already uses frozen Track-On-R.

## Recommendation

Evaluate **LiteTracker** and **TAPNext++** first with frozen weights on the existing recorded shuffle clips at raw stride1. LiteTracker has directly relevant published RTX3090 latency; TAPNext++ has a simple recurrent interface and stronger long-duration/re-detection training. Neither has demonstrated reliable identification of the target while it remains under a moving cover in our task. Published inference speed does not predict fine-tuning time.

| Method | Causal interface | Published speed and conditions | Practical fit |
|---|---|---|---|
| LiteTracker | One incoming frame; cached temporal features; reuses CoTracker3 online weights | Official repo: p95 latency29.67ms on RTX3090,1024points; roughly34Hz equivalent. Paper input512x384(WxH). Natural-scene table is marked unpublished by the authors. | Best immediate latency baseline on our hardware class; code/checkpoint available. |
| TAPNext++ | One frame and recurrent state, no future frame needed | Paper:193FPS/5.18ms latency,256points,256square,H100. Headline562FPS is a different32frame-window throughput configuration. | Strong candidate for persistent online representations; benchmark Ampere implementation rather than assuming H100 speed. |
| TAPNext / BootsTAPNext | One frame and recurrent state | Original project:14.2ms on V100 and5.05ms on H100,256points. | Useful speed reference; prefer ++ for long sequences because the update explicitly addresses state degradation. |
| Online TAPIR / BootsTAPIR | Causal per-frame feature extraction and temporal context | Original TAPIR project reports roughly40FPS for256points/256square; not an RTX3090 BootsTAPIR measurement. | Mature causal baseline, public PyTorch causal BootsTAPIR checkpoint; CNN-based alternative to interleaved DINO adapters. |
| CoTracker3 online | Overlapping temporal chunks, predictor step8 for window16 | TAPNext++ paper reports102FPS throughput but80ms worst-case output latency on H100/256points. These are different metrics. | Useful baseline and training codebase; vanilla chunk wrapper needs explicit latency handling for control. |
| Point2Pose | Causal RGB-D multi-object6D pose tracking | Project reports30Hz on RTX4090. Uses2Dtracker correspondences, depth and geometric maps. | Relevant if depth is available and metric geometry is desired; not a direct RGB-only hidden-object tracker replacement. |

Sources for the table: [LiteTracker official repo](https://github.com/ImFusionGmbH/lite-tracker), [LiteTracker paper](https://papers.miccai.org/miccai-2025/paper/2578_paper.pdf), [TAPNext++ paper, Table2](https://tap-next-plus-plus.github.io/static/papers/paper.pdf), [TAPNext project](https://tap-next.github.io/), [TAPIR project](https://deepmind-tapir.github.io/), [CoTracker3 official predictor](https://github.com/facebookresearch/co-tracker/blob/main/cotracker/predictor.py), [Point2Pose project](https://point2pose.github.io/).

## LiteTracker: strongest directly comparable speed evidence

LiteTracker introduces inference optimizations over CoTracker3: cache past frame features and initialize current estimates from past motion. It accepts the standard CoTracker3 online scaled checkpoint without extra training. Its main validation is endoscopic tissue tracking, and the authors additionally report natural-scene benchmarks including RoboTAP. This does not establish cup-containment reasoning. Speed alone is not grounds to assume better hidden-object identity than Track-On-R.

The published p95 result is valuable because it uses the same RTX3090 GPU family as our instance. It still differs from our setting in point count, image preprocessing, precision, support tracks and whether another training job shares the GPU. Measure complete per-frame preprocessing, transfer and forward latency with1and4target queries, and report actual support-point count. CoTracker3 supplies official training and fine-tuning scripts; adapting a tracker and deploying LiteTracker with the resulting weights remains an experiment, not a proven transfer result.

Sources: [LiteTracker README and latency table](https://github.com/ImFusionGmbH/lite-tracker), [MICCAI paper](https://papers.miccai.org/miccai-2025/paper/2578_paper.pdf), [CoTracker training/fine-tuning instructions](https://github.com/facebookresearch/co-tracker#training).

## TAPNext++: low-latency recurrent alternative, with a relevant failure

TAPNext uses a recurrent video transformer with temporal state-space processing and spatial attention; point tokens directly produce coordinate/visibility outputs. TAPNext++ changes training, including1024frame sequences and re-detection augmentation. Public PyTorch checkpoints at256square and512square and a per-frame state-carrying demo are available in the official TAP repository. This architecture avoids Track-On's explicit K-candidate reranking design, but that alone does not establish faster fine-tuning on our dataset.

The current official TAP README documents a training framework for TAP-Net/TAPIR, rather than an equivalent turnkey TAPNext++ fine-tuning recipe. Starting frozen inference is straightforward compared with promising a ready-made training pipeline. The H100 speed comparison uses FlashAttention3; RTX3090/Ampere must use a compatible implementation and be timed separately.

Most relevant to our goal, the authors explicitly show a cups-and-balls sequence where TAPNext++ loses the ball while hidden and only recovers it after the cup is lifted. Re-detection after exposure is useful, but choosing the correct cover before opening requires hidden-target association. We must score those stages separately.

Sources: [TAPNext++ project, cups-and-balls example](https://tap-next-plus-plus.github.io/), [official checkpoints/training scope](https://github.com/google-deepmind/tapnet), [official PyTorch model](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapnext/tapnext_torch.py), [TAPNext++ paper](https://tap-next-plus-plus.github.io/static/papers/paper.pdf).

## Integration and evaluation

The current Track-On2/Track-On-R family itself targets real-time inference; its training cost cannot be compared directly with another method's inference FPS. Our current401query/M72 setup and concurrent training also differ from the published Track-On2 efficiency configuration. [Track-On2 efficiency](https://kuis-ai.github.io/track_on2/).

For a controlled replacement test, keep the same initial point, causal frame order, top crop and held-out scenes. First compare native2Dtracks/visibility independently of our XYZ decoder and FM. Score visible frames, fully covered/shuffling intervals, correct-cover binding before opening and recovery after exposure. A3Dbody-center projection is not necessarily the same physical point as a clicked surface landmark after rotation; keep that distinction in the metric. Do not treat low visibility or a tracker intentionally refusing a hidden point as a precise amodal center estimate.

Record median/p95 end-to-end latency, FPS, actual query/support count, resolution and VRAM. At25Hz, one frame arrives every40ms; a tracker near30ms leaves limited time for extra camera encoding/action generation if all modules run at that same cadence. FM currently replans every10frames, so schedule and amortization matter. This is an integration budget, not a measured system latency.

If metricXYZ is desired without a state-driven readout shortcut, evaluate calibrated depth/multiview geometry or a visual-only XYZ head. Depth under total occlusion measures the cover surface, and Point2Pose advertises recovery once objects reappear; it does not supply the trajectory of an unseen object moved under a cover. Our diagnosed joint sensitivity in e111 remains a separate confound when judging a tracker replacement. [Point2Pose](https://point2pose.github.io/), [e111 diagnostic](e111_bc.md).

No training changes were made by this survey. Next proposed evaluation: LiteTracker and TAPNext++ on a small fixed set of recorded train/new-order/new-layout/both clips before any expensive fine-tuning or FM integration.
