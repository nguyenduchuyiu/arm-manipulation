# e98_hist

Status: planned; full corrected run not launched during e97 debugging.
Native causal three-camera amodal XYZ,125,314,374trainable parameters, no freeze.
Official ImageNet ViT-B/16 spatial initialization, RGB ImageNet mean/std,
cloned temporal attention/normalization with temporal output residuals and time
embeddings zero initially. Native code only, no third-party tracker runtime.

XYZ readout combines a target feature MLP with a small9->64->3geometric residual
from per-view attention-weighted pixel XY and camera probability mass. No camera
calibration input. Inputs are3RGBviews+target ID+past/current timestamps, output
XYZ. ID is embedded into the target query used at fusion blocks4/8/12; blocks1--4
are not ID-conditioned. Loss is train-normalized XYZ0.5endpoint+0.5prefix MSE
only. No masks are loaded for training, used in the loss or used to select training
examples. Evaluation alone loads occlusion fractions for hidden/visible metrics.

Fresh training first300updates:4same-RGB reveal contexts x4IDs, T1/batch16.
Next300:2reveal contexts x4IDs plus8opening examples,2per ID, T1/batch16.
Then mixed causal T30, logicalbatch4/micro1, one episode per ID, original phase
sampling[.15,.35,.2,.3]. Default1500updates total, all parameters trainable
throughout. Head LR3e-4; backbone3e-6visible/3e-5temporal. AdamW,WD1e-4,
clip1,BF16,seed97. Each stage cosine LR0.1floor and20visible/50temporal warmup.
Only train XYZ mean/std,10mm std floor. Source RGB320square resized240x320.
Approved subset512train/32val/64composition/32spatial/64combined e96episodes.

Validation uses ordinary30frame history at cover-open t_obj before grasp; every
250updates and curriculum boundaries, choose lowest exposed mean error. Tests
run after fit on saved best across all evaluation phases, exposing phase scores.
Clip30always contains first context frame and selected current endpoint. Optional
--init restores exact saved model configuration/weights, skips visible curriculum,
and creates fresh optimizer. No actions/H/K/flow/robot rollout applies.
Artifacts actualconfig/sourcehashes/subset/normalization/cache/metrics/progress/
best/latest/test and predictions. Run:OATFlow/tracking/run.sh --output
/workspace/checkpoints/e98_hist; log workspace_analysis/logs/e98_hist.log.

Evidence and limitations: the previous mask-assisted e97debug run used1000head-only control
updates then50full-finetune updates, not this fresh1500update recipe. Opening
T30val11.857mm and held-out means7.927/18.156/18.401mm. Those results do not apply
to this mask-free recipe. The matched XYZ-only debug control achieved20.736mm
validation opening error; its checkpoint was deleted, metrics retained. Full1500run and hidden
tracking quality remain untested. Whole-prefix temporal resampling is only
~1.46Hz median at opening and can undersample fast swaps; this has not been fixed
by the readout changes. Do not claim success under full occlusion.
