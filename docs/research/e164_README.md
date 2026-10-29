# E164 — Self-attention giữa 8 surface patches

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Thêm8patchSA chưa cải thiện reconstruction tổng thể18k; attention không tự bảo đảm topology/không giao nhau. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành18k/evaluation; best18000.

## Thay đổi kiến trúc

Giữ E162 original encoder +128taskdecoderqueries → cross-attention8patchqueries.
Thêm đúng một full self-attention8×8 giữa patchlatents, width128,4heads,dropout0,
residual z+MHA(z,z,z) và LayerNorm. Không FFN/positionalencoding/mask mới.
Sau đó giữ nguyên8×16UV folding: c+u*a+v*b+delta,128outputXYZ.
Mỗi patch đọc cả8latents, không hard grouping các decoderqueries.
1,030,540trainableparams, +66,304(+6.88%) so E162964,236params.
Tất cả module E162 gồm cả foldinghead khởi tạo seed135 giống hệt; thêm block SA
sau khi khởi tạo modules cũ. Train từ đầu, không warm-start E16340k.
SAM3 frozen454,038,784params, không backbone3Dpretrained.

## Training giữ nguyên E162

18kupdates, seed135; batch8scenes/32queries,4096fullsceneXYZ,4taskcolors,
GT/pred128. Data/cache immutable E157 symlink:1024train,128validation;
RGB320×240 với robot/chair/mat/table/floor/camera như cũ. Physical cube sizes
đã renderRGB/depth/GT và encodeSAM3 nhất quán; SAM3vision_features256D frozen.
HaarSO3+translation±0.5m, SE3scene/GT trướcXYZ×2, giữfeaturecorrespondence;
không coordinate scaling, không sensor-noise augmentation.
AdamW3e-4→3e-5 cosine, warmup100,weightdecay0.01,gradientclip1,
BF16learnedoperations,FP32foldingcomposition/parameters/optimizer/loss,TF32off.
Loss y hệt E161/E162: exactEMD +1.0matched allpairs,16256orderedpairs,
meters, cùng EMDmatching; không thêm overlap/repulsion/normal regularizer.
GTmasks chỉ supervision/eval, không input GTcenter/pose/shapeparams/masks.
Action/flow/rolloutN/A. Budget≈140.625scenepasses.
Best minimum fixedSE3validationEMD mỗi500steps, canonicaldiagnostic,
không feedback test. Existing E16218k/E16340k artifacts giữ nguyên.

## Evaluation đã chạy

Primary matched-budget: E16418k vs E16218k, khác SAblock và6.88%parameters;
không coi E16340k là matchedbudgetbaseline.
128scenes×4tasks/split IID/positions/rotations/sizes/combinations,
IID thêm4fixedE137SE3views, K128,1RGBDobservation/case.
Same E157physicalresize seeds/cases:target-only/allobjects localaxis×0.6/1.4,
freshRGBD/GT/SAM3 mỗi case, ghi visibility/clearance exclusions.
Metrics EMD/pair,directionalCD,centroid-alignedCD,coverage2/5mm,robustXYZextent,
selection và pairedextent/robustextentresponsegain. BestvsGTviewer rawXYZ,
có tùy chọn8patchcolors. Không dùng test để chọn checkpoint.

Run: `.venv-concerto/bin/python -m OATFlow.patch_attention.benchmark`, rồi
`.venv-concerto/bin/python -m OATFlow.patch_attention.train` (auto evaluate/report).
Config/checks: config.json,train/config.json,benchmark.json,comparison_audit.json;
data_config.json vàfeatures/config.json trỏ nguồn immutable.
Log tmux `ssh_tmux:e164-log`, file `../e164_train.log`.
Giới hạn: self-attention tạo kênh phối hợp nhưng không bảo đảm patches không
giao/chồng nhau. Một seed, camera cố định, visibleGT; cần đánh giá thực nghiệm.

## Kiểm tra và khởi chạy

Pass: mọi E162parameter/buffer bao gồm foldinghead khởi tạo giống hệt; decoderhidden không đổi. Attentionweights cóshape[B,4,8,8], không maskedkey; outputpatch0 có nonzerogradient tới cả7latents khác. Patchpermutationequivariance,128hiddentokenpermutationinvariance vàfoldingformula/128output pass. SixGPUupdates cóloss/gradientfinite; self-attentionvànormalization mới đều nhận gradient, cùng cácmodulecũ.

Loss functionobject identicalE161/E162. comparison_audit.json xác nhận data/features và seed/budget/LR/SE3/loss/foldingsettings giữ nguyên. Benchmarkmedian0.1486s/update; fulltrain ban đầu peakallocated2.99GiB. Lúc khởi chạy đãqua50updates finite; nay18k/evaluation đã hoàn thành. Không giữ preflightcheckpoint/cache mới.

## Kết quả hoàn thành

Train31.76phút, eval7.35phút; peak allocated2.99GiB. Best18000, validation EMD12.412mm.

| Test | Model | EMD mm | Pair mm | Aligned CD mm | Pred→surface mm | Surface→pred mm | Coverage5mm | Selection |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iid | E162 | 11.598 | 6.900 | 6.476 | 7.642 | 6.114 | 50.76% | 100.00% |
| iid | E164 | 11.678 | 6.951 | 6.591 | 7.811 | 6.201 | 50.71% | 100.00% |
| SE3 | E162 | 12.182 | 7.063 | 6.561 | 7.896 | 6.278 | 48.78% | 100.00% |
| SE3 | E164 | 12.328 | 7.038 | 6.510 | 7.972 | 6.308 | 49.31% | 100.00% |
| positions | E162 | 12.587 | 7.045 | 6.471 | 8.078 | 6.427 | 48.43% | 100.00% |
| positions | E164 | 13.189 | 7.177 | 6.632 | 8.359 | 6.646 | 47.31% | 100.00% |
| rotations | E162 | 12.989 | 6.983 | 6.568 | 8.225 | 6.549 | 47.94% | 100.00% |
| rotations | E164 | 13.124 | 7.001 | 6.543 | 8.466 | 6.522 | 47.69% | 100.00% |
| sizes | E162 | 14.746 | 7.884 | 6.793 | 9.852 | 7.804 | 44.53% | 100.00% |
| sizes | E164 | 15.692 | 7.838 | 6.940 | 10.600 | 8.848 | 40.79% | 99.80% |
| combinations | E162 | 11.607 | 7.020 | 6.519 | 7.696 | 6.208 | 50.08% | 100.00% |
| combinations | E164 | 11.820 | 7.071 | 6.656 | 7.855 | 6.263 | 50.24% | 100.00% |

| Physical test | Model | EMD mm | Pair mm | Coverage5mm | Extent gain | Robust extent gain |
|---|---|---:|---:|---:|---:|---:|
| target_only | E162 | 11.511 | 6.888 | 51.10% | 0.236 | 0.277 |
| target_only | E164 | 11.521 | 6.956 | 51.37% | 0.308 | 0.286 |
| all_objects | E162 | 11.447 | 6.869 | 51.31% | 0.242 | 0.275 |
| all_objects | E164 | 11.466 | 6.914 | 51.54% | 0.308 | 0.283 |

iid: E162−E164 EMD = -0.080mm, scene-bootstrap95%CI [-0.232,+0.055].

SE3: E162−E164 EMD = -0.146mm, scene-bootstrap95%CI [-0.304,+0.000].

Physical attempts1280, scored1271, exclusions9; all cases/rejections in physical_scale/cases.jsonl. Gains use full visibleGT paired deltas; ideal1. Raw metrics except explicitly named centroid_aligned_CD. Robust extents/Q95−Q5 and coverage2mm saved in archives/summary. No test selection.

Surface patches add geometric coupling without extra losses; no GT face assignment. This does not establish independent XYZ grounding or guarantee coverage. One seed,6.88%extra parameters and changing visibility remain limitations.

[Best vs GT](http://localhost:17070/patchattn/viewer.html) · [Learning curves](http://localhost:17070/patchattn/learning_results.png).

## Nhận xét và audit sau hoàn thành

Ở budget18k, thêm SA8patch chưa cải thiện reconstruction tổng thể: IID EMD11.598→11.678mm, SE3 12.182→12.328mm; pairedscene-bootstrapCI95 đều gồm0 (SE3 upper0.000373mm). IIDalignedCD6.476→6.591mm, coverage50.76→50.71%; OODsizesEMD14.746→15.692mm vàcoverage44.53→40.79%. Target-onlyextentgain0.236→0.308, nhưng robustgain0.277→0.286 chỉ tăng nhẹ; không thể coi đây là sửa xong patch arrangement. Không có metric intersection/overlap mới nên không kết luận định lượng mức patchgiao nhau giảm/tăng. Viewer giữ rawXYZ và8patchcolors để kiểm tra trực tiếp.

Best18000, valEMD12.412mm; trainmean14k–16k→16k–18k EMD13.077→12.281mm,pair7.127→7.048mm; valmeanEMD13.524→12.899mm,pair7.081→7.016mm. Chưa plateau rõ ở18k; kết quả hiện tại chưa kiểm tra40k cho kiến trúcSA. E16340k làmodelkhôngpatchSA, không primary matched-budgetcontrol.

Completionaudit pass: training-source/checkpoint hashes đúng; hoàn thành18000, best minimum fixedSE3validationEMD ở18000; logtotal=EMD+pair. E162baseline canonical/SE3 khớp bit-for-bit; physicalcase metadata/GT vàbeforepredictions khớp. One physicalaftercase (scoredindex1005) có baseline rerun variation ảnh hưởng1/2021scoredqueryEMD0.02541mm (meanabs0.00001257mm); giữrawresults. Tính lại EMD/pair/extentgains độc lập từarchives khớp. Xem completion_audit.json.

Viewer browsercheck pass: HTTP200, image/3D load, best18000 đúng,8patchcolors×16points, visibility/defaultcolor toggles, không pageerrors. Xem best_viewer_browser_check.json. Train31.76min,eval7.35min; GPUallocated2.99GiB.
