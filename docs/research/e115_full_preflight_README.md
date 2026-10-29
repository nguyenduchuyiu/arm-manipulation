# E115 full-test evaluator preflight

Status: passed. Reproduce two previously tested held-out RedCube cases with
the same best epoch32/step29420, H25/K10/Euler10/noise0/900frame policy protocol,
using the no-video path planned for most of the full test. Cases: g000/p01/t0
(success) and g036/p00/t0 (900frame timeout). Compare every saved state, command,
sampled action, mask and history timestamp against test_k10_2, bit for bit.

The no-video path skips only video capture/export; both camera render schedules,
top-history cadence and all policy computations stay unchanged. Source/frozen
weights are the same E115 model used in the four-case
test. Training: full2048episodes, total40epochs, batch256, LR2e-5, freezeTCOW/ViT,
trainFM/joint encoder, H25 EE deltaXYZ/relative6Drotation/absolutejaw.

Record essential numerical/timing results locally and remove this completed
preflight folder, temporary script and generated trace duplicates. Full-test
output will be fresh and will not reuse these diagnostic outcomes as test rows.

Results: success case324frames in11.938s; failure case900frames in30.409s.
Every saved array and physical result matches test_k10_2exactly. Frozen weights
unchanged. An initial attempt that skipped some camera renders caused trajectory
differences and was rejected; both original render schedules are retained.
Essential evidence: e115_full_preflight_check.json. Duplicate trace folders and
temporary execution scripts removed after this record.
