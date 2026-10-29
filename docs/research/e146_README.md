# E146 — Local kNN encoder +128-query cross-attention

Status: preparation/benchmark planned; no training/results yet.

Replace two global point self-attention layers with XYZ-only kNN16 local geometry +two sparse mean-neighbor residual blocks.17D descriptor: absoluteXYZ, mean/std/min/max neighbor deltaXYZ, mean/max neighbor radius; learned17→64→64 geometry MLP. Fuse geometry64 with SAM3 features256 andXYZ3 via323→128→128 MLP. Decoder unchanged in form: task embedding,128 learned queries, two width128/4-head cross-attention decoder layers,128×3 output.

No N×N point attention; local aggregation uses sparse block-diagonal adjacency O(kN), decoder cross-attention O(128N), query self-attention O(128²). Geometry/graph use XYZ only, no GT masks/centers/sizes. Fixed static two-example graphs are cached; graph build cost must be reported separately from steady training/inference.

Same E142/E14420/80mm frames, full scene robot/chair/mat/table/floor, same center/task0. Prepare8,192 and full74,356 points; original16k prefix/features bit-exact. Frozen SAM3,128 GT/output points, no augmentation, one Chamfer loss. FP32/TF32 off for fair benchmark. Compare old/new step time and memory, train selected configuration after benchmark; final counts/budget/results pending.

Artifacts `/workspace/robotics_assets/experiments/e146_local`; code `OATFlow/local_cloud`. This is a new architecture trained from scratch, not a dropped-layer reuse of the old checkpoint.
