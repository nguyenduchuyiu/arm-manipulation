# e97_xyz debug

Status: complete. The old model predicts almost the same location for different
IDs on identical images. Old Red/Blue prediction separation0.270mm versus true
82.614mm. Corrected model separation94.269mm on that input, errors17.714/4.964mm.
Data audit512train episodes passes entity order, exact world/base conversion,
label/state timestamp alignment and opened-target phase checks. Normalization
round trip and backbone gradients pass. This is an optimization/representation
failure, not a proven label-index/normalization bug.

## Chosen changes and evidence

Native causal backbone/fusion retained. RGB ImageNet ViT spatial initialization
with zero temporal output residuals replaces the problematic RGB-only TCOW
initialization for new training. TCOW T1opening warmup41.230mm becomes106.791mm
with actual30frame history; same experiment using ImageNet gives38.951/39.261mm.
Native spatial weight import is functionally checked against a reference ViT.
RGB normalization is persisted in model_config. No upstream runtime dependency.

Readout now optionally combines target-token features with a small9->64->3
geometric residual from attention-weighted patch XY and camera probability mass.
Only835extra parameters; total125,314,374. Existing amodal masks supervise query
attention at fusion layers as an auxiliary CE loss. Inference remains RGB+ID+
past/current timestamps -> XYZ only. No GT masks/centers/calibration/cover states
as model inputs. Masks at future frames never enter earlier clip supervision.

Matched geometry-head controls: same ImageNet fixed features,1000head updates,
batch128, same sampling, then50full-finetune updates (all parameters enabled),
T1batch16, backbone3e-6/head1e-4. Aux weight0 produces20.736mm opening T30val;
aux weight1 produces11.857mm. The result is a combined init/readout/training
change; it does not isolate every contribution or prove a single coding bug.
Full-finetune run loads the best head checkpoint, step600, before50updates.
Saved full.pt step1050 is the job counter; its model ancestry is600head+50full.
This diagnostic is not1500fresh full-finetune updates. Frozen feature stages were
controlled head tests; the corrected main trainer enables all parameters from
the start. New full1500update e98recipe remains planned, not launched.

## Opening evaluation, ordinary30past/current frames

| Group | Old mean mm | Corrected mean mm | Episodes |
| --- | ---: | ---: | ---: |
| Validation |106.233|11.857|32|
| Unseen compositions |101.168|7.927|64|
| Unseen layouts |107.162|18.156|32|
| Both unseen |107.133|18.401|64|

Select head by32opening validation episodes, never held-out test. Each current
endpoint is first cached frame>=t_obj before grasp (stride3, <=2raw frame delay).
Actual source/evaluation paths in subset manifests; exact thresholds in test.json.
Checkpoint:/workspace/checkpoints/e97_xyz/test_debug/mask/full.pt.
Same-image ID binding in mask/id_binding.json; before/after4query comparison in
mask/opening_compare.png. Camera projection is only used for visualization.

15focused tests pass with active feedback and geometric readout. Trained model
has exactly zero future-image gradients and zero prefix changes when future
RGB/timestamps are replaced. A real30frame training backward with XYZ+attention
loss reaches all125,314,374parameters with finite gradients, including temporal
QKV. Test configuration in mask/config.json; metrics/causality/preflight JSON.

## Negative controls and remaining limits

T1tiny16image overfit is0.331mm from native TCOW init and0.480mm from collapsed
checkpoint after300updates/B16. Same-update B2 gives18.208mm/T1 and~20mm/T30;
B16has8xexample exposure, so batch size alone is not isolated. The original full
1500update run saw3000clips/512episodes,5.859clips per episode,65.27%hidden
endpoints. Gate/visible-only-loss/extra positional memory/shared-reveal curriculum
alone did not fix collapse and are not retained as claimed fixes. Gate and
positional-memory experimental code removed. Failed/superseded checkpoint weights
removed, metrics retained; paths/reasons in cleanup.json. Original e97 preserved.

Hidden tracking remains untested by the corrected checkpoint. Whole-prefix
resampling to30frames gives median1.456Hz at opening (range1.142–1.893Hz);
22.75%ofswap motion cores have at most1sample. This limits fast hidden tracking
and is not corrected by the new readout. Do not equate opened-target coordinate
accuracy with success under full occlusion or grasp success.

## Code and reproduction

Changed OATFlow/tracking/model.py,data.py,weights.py,train.py,run.sh and
OATFlow/tests/test_tracking_xyz.py. No edits to user prior/RL/reach work in this
debugging turn. Temporary diagnostic scripts removed after completion. Canonical
research report is local docs/research/REPORT.md; output folders contain README.

Tests:PYTHONPYCACHEPREFIX=/tmp/tracking_xyz_cache OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1 .venv/bin/python -m unittest OATFlow.tests.test_tracking_xyz -v

Future full run:OATFlow/tracking/run.sh --output /workspace/checkpoints/e98_hist
Actual recipe/limits in docs/research/e98_hist.md. It uses ImageNet RGB init,
full-finetune throughout, shared-ID reveal curriculum, XYZ+attention supervision,
opening T30checkpoint selection. Debug log:workspace_analysis/logs/e97_debug.log.

Same-history30frame ID intervention confirms old separation0.356mm, corrected92.179mm, true82.614mm; id_binding.json. Earlier T1control in id_binding_t1.json.

### Mask-free requirement

User rejected mask supervision after the diagnostic. The11.857mm opening result
is historical and mask-assisted. Current training loads no mask files, uses XYZ
loss only and samples by phase/ID without occlusion labels. Evaluation alone
loads occlusion fractions to report hidden/visible metrics. ID is embedded256D
and becomes the target query at fusion blocks4/8/12; blocks1--4 are not conditioned
on ID. No architecture change or new optimizer run was made for this correction.
The historical matched XYZ-only control reached20.736mm validation opening error;
its weights were deleted previously. The new full mask-free run is not launched.
