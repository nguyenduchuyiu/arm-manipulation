# E124 — Full-frame task-conditioned MobileSAM

Status: RUNNING.

Continue E123 best epoch5/step1280 task-prompt and mask-decoder weights on every
frame of the original full pick training split. The original MobileSAM TinyViT
and prompt encoder stay frozen; only the four learned256D task embeddings and
original single-mask decoder train. No FM is attached.

Input: front RGB and requested target ID (Red0,Blue1,Green2,Yellow3).
Visible instance masks are training labels only; no input masks, points, boxes,
depth, joints or XYZ. Supervise all4task IDs per image to separate object choice
from the arm's current manipulation target. Absent visible targets receive empty
mask labels; this remains single-frame visible segmentation.

Data: all2048original train episodes,296688frames including final frames,
stride1,1186752mask queries/epoch. Preserve the E117/E122 train/val/test split.
Every frame is globally shuffled and visited exactly once per epoch. Five
additional full-frame epochs:4636updates/epoch,23180total updates.

Training: RTX3090, CUDA BF16 autocast with FP32 trainable weights and loss;
batch64images/256queries; AdamW freshly initialized, task LR1e-3, decoder LR1e-4,
weight decay1e-4,20update warmup, gradient norm limit1, seed123 (frame-order seed124).
Loss=BCE+softDice+0.1IoU-quality MSE. Original SAM PIL resize240x320->768x1024,
normalize and bottom pad to1024square. No gradient checkpointing or augmentation.

Cache:84.883GiB native RGB/instance labels in CPU RAM,4loader workers. Frozen
TinyViT embeddings are computed online: a full FP16 feature cache would require
579.469GiB. Validation embeddings stay in RAM (approximately4.6GiB); no disk
feature cache is created. A four-update real-data preflight checks gradients,
frozen encoder hash, resize equivalence, and paired decoder versus original SAM.

Validation: all256original val episodes,8uniform frames/episode including first
and last,4queries/frame=8192queries. Highest validation visible-target mean IoU
selects best.pt. Record visible-mask IoU/Dice, identity accuracy, confusion and
low-visibility counts. Report absent cases separately; validation selects no
training samples. Per-epoch metrics, packed masks and sample sheet are retained.

Testing after training: best validation checkpoint; same E123 protocol with all
1408held-out home queries,72later expert images x4queries,16retained XY images
x4queries. Later/XY are diagnostic subsets, not full-trajectory estimates.
No manipulation rollout; action representation/H/K/flow/noise/frame budget are
not applicable. Test case details and exact checkpoint hash are in test/configs.

Artifacts: config.json, original split manifests, preflight.json, cache_info.json,
baseline.json, loss.jsonl, progress.json, history.json, validation/epoch_*,
best.pt (weights+metadata), latest.pt (optimizer/RNG/frame order/cursor),
verification.json, test/{home,dynamic,xy}. No redundant final checkpoint.

Run: `.venv/bin/python -u -m OATFlow.task_mask.full_train` from the repository,
inside tmux with combined stdout/stderr piped to tee. Implemented in
`OATFlow/task_mask/{full_train,full_data}.py`; exact source hashes/configuration
are stored in config.json. Output: `/workspace/robotics_assets/experiments/e124_mask_full`.

Results: pending.

Preparation: loader validation preserves non-target instance IDs (robot/background labels5,7,8); only IDs1–4 define object masks. Initial strict maximum-ID check was corrected before training. Failed preflight metadata directory removed; no learned checkpoint existed.

Launch verified: actual step25/23180, real CUDA updates with finite losses; resumable checkpoint step1 has correct optimizer counters,296688unique global frame indices and correct cursor. Native cache64.43s,84.883GiB; container RAM about98.7GiB; peak allocated VRAM20.12GiB. Initial measured speed gives about8.3hours for5epochs plus validation/testing (about1.65h/epoch). Evidence: launch_verification.json. Window:ssh_tmux:1 e124_mask.

Interruption/resume: original process disappeared around epoch2/step5925;
no traceback and current cgroup OOM counter0, cause unconfirmed. Resume latest
step5750, epoch2 cursor71296, preserving original optimizer/RNG/order and5epoch
budget. RAM-only caches are rebuilt. Trainer now supports `--resume`, saves on
its first resumed update, and retains partial-epoch loss sums in future saves.
The original checkpoint lacks an epoch2 loss sum, so epoch2's overall training
mean will be null; report the trained suffix mean explicitly. Epoch1 validation
IoU71.0173%,identity82.1324%. Resume audit: resume_1.json/resume_verification.json.

Resume launch verified: checkpoint step5751 has advanced beyond5750, optimizer counters and cursor match; first new loss0.0429238 is finite. GPU uses approximately20.1GiB allocated VRAM; remaining training estimate approximately6hours. See resume_1.json for replay checks.

Numerical replay check: the first replay loss matches the original log at its
4decimal display precision; repeated step5775 differs(loss delta-0.00763449).
State restoration is verified, but bitwise reproduction of subsequent CUDA/BF16
updates is not established; cause has not been isolated. Original training
configuration and shuffled sample order are unchanged. Exact comparison is
recorded in resume_1.json/resume_verification.json.

Second interruption observed before E126 automatic-mask evaluation: no E124
process, GPU empty, current cgroup OOM count0, cause unconfirmed. Last logged
step7900; latest resumable checkpoint7750(epoch2,cursor199296). Original5epoch
budget remains unchanged. Resume queued after E126; audit:resume_2.json. Previous
resume verification/cache metadata copied with resume_1_ prefixes.
