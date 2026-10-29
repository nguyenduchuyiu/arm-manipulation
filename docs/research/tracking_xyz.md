# Native three-view amodal XYZ

Implemented in `OATFlow/tracking`:model,data,weights,train,evaluate. No third-party
tracker runtime, legacy converters, mask/action heads or hidden-state inputs.
The old TCOW source/pipeline and obsolete experiment artifacts were removed.
Shared FM and prior/reach code remain; retained FM outputs match original exactly.

RGB `[B,3,T,3,240,320]`, target ID and timestamps map to normalized XYZ `[B,T,3]`;
`predict` returns current base metres. Twelve width768 divided space/time blocks,
strict causal temporal SDPA, first-frame-only CLS. Fusion after4/8/12 exchanges
camera evidence through per-timestamp target/context states,width256. Prefix
positions are stable and actual elapsed time is encoded without episode duration.

Native backbone artifact:/workspace/checkpoints/vision_init/tracking_backbone.pt,
245tensors/121,353,984parameters,485,493,091bytes. Original source tensor values
verified exactly before deleting its full training checkpoint. The unused final
norm was removed. Full model125,313,539parameters. Source weights alone do not
provide learned XYZ prediction; fusion/ID/head require training.

Verification:11focused tests pass, including zero future-image/time gradients,
future-value perturbation, prefix extension, active camera feedback, batch
isolation, XYZ normalization/restore, native tensor loading, parameter gradients,
and data clips unaffected by future cached images/timestamps/labels. Fresh import
and inference succeed with external tracker imports blocked. Both frozen and
upper4 optimizer steps pass on a real e96 clip (10.26/11.35GiB GPU peak).

User-authorized subset training/evaluation is e97_xyz; exact data split, settings,
status and results in [e97_xyz.md](e97_xyz.md). [cleanup manifest](tracking_cleanup.json).

Full finetuning and opened-cover evaluation completed; localization unsuccessful.
Actual settings/results in [e97_xyz_full.md](e97_xyz_full.md).

Debugging added native RGB ImageNet initialization and835parameter spatial
readout. The historical mask-assisted diagnostic reached11.857mm opening error;
the matched XYZ-only control reached20.736mm. These are separate results.
Following the user's mask-free requirement, the current trainer loads no training
masks and uses only XYZ loss. Visibility labels are evaluation-only. ID is a
256D target query read at blocks4/8/12, with feedback to later backbone blocks;
blocks1--4 are unconditioned. The next full run is planned, not launched.
Full diagnostic config and hidden-tracking limits in [e97_debug.md](e97_debug.md).
