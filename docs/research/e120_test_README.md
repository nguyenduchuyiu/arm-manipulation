# E120 — Full held-out pick evaluation

Status: planned; no result yet.

Policy: frozen RGB ViT plus full-resolution calibrated XYZ-map patch embedding,
aligned spatial fusion, target-query decoder and FM initialized from E118 reach
step120,000. Only ViT frozen; all other policy modules train for80epochs on E117
part1/8 (256 demonstrations,37,536 stride1chunks), batch128, FM LR2e-5 and other
trainable LR1e-4, warmup100, loss FM + weight1 target-cloud auxiliary. H25 outputs
anchor-relative EE deltaXYZ, relative rotation6D and absolute jaw command.

Evaluate immutable final exact80epoch checkpoint step23,520; select no settings
using test outcomes. All3,200 held-out episodes: spatial2,048, unseen permutation128,
combined1,024. H25/K10, Euler10, noise0,300frames, policy25Hz/servo50Hz/physics500Hz.
Inputs: native front RGB240×320, full front depth2mm, intrinsics/camera-to-base,
target ID and measured joints. GT target mask/center is never a policy input.
Actual online depth is quantized and back-projected through the same geometry path
as training. Success requires correct-object lift>4cm and bilateral jaw contact
for10 consecutive frames; wrong-object lift fails. Save physical trajectories,
no video rendering.

Exact checkpoint SHA/protocol in `config.json`; per-case outcomes in
`results.jsonl`, partial aggregate in `progress.json`, final in `summary.json`.
Training details and architecture: parent `README.md` and `train/config.json`.
