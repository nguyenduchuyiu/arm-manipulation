# E162 — Target-conditioned surface folding

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Các patch phẳng nhưng chưa ghép đúng bề mặt cube; không có cải thiện reconstruction tổng thể18k. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành18k/evaluation; best18000.

## Data

Giữ nguyên E161/E157:1024 train scenes,128 validation,4 task màu/scene;
robot/chair/mat/table/floor và camera như cũ. Physical local XYZ dimensions đã
đổi trong simulator0.6–1.4, render/encode lại RGBD và frozen SAM3 nhất quán.
Actual camera RGB320×240, không đổi lên512×512 trong ablation decoder này.
Frozen SAM3 vision_features256×72×72, sample cùng4096 full-scene XYZ.
Data/features dùng symlink E157, không tạo cache mới.

## Kiến trúc

Original point encoder và TransformerDecoder hai lớp giữ nguyên:128 learned
queries → target hidden[batch_tasks,128,128]. Upstream weights khởi tạo seed135
giống hệt E161, nhưng mọi module trainable được train từ đầu; không warm start.
SAM3 frozen454,038,784 parameters, chỉ cache features.

Thay shared XYZ MLP bằng:

- 8 learned patch queries cross-attend toàn bộ128 hidden qua một MHA4heads;
 residual query+attention và LayerNorm →8 patch latents128D, không hard grouping.
- Shared Linear128→9 dự đoán c,a,b (3D/patch).
- Fixed16UV từ lưới4×4 [-1,1]², không trainable/không liên quan RGB UV.
- Shared delta MLP130→128→128→3, GELU; input[patch latent,u,v].
- p=c+u*a+v*b+delta(z,u,v), [B,8,16,3]→[B,128,3].

964,236 trainable parameters vs E161878,979 (+85,257,9.70%).
Frame weights c std0.01,a/b std0.001, bias0; delta last layer weight/bias0,
khởi đầu patches phẳng. Không constrain orientation/scale, không GT center/pose.
BF16 learned operations, FP32 geometry composition/loss/parameters/optimizer;
TF32 off. UV grid là surface coordinates, không phải image projection.

## Training

18,000 updates từ đầu, seed135, batch8scenes/32tasks, ≈140.625 scene passes.
AdamW3e-4→3e-5 cosine, warmup100, weight decay0.01, gradient clip1.
SE3 Haar SO(3)+translation±0.5m: cùng transform scene và GT trước XYZ×2;
features gắn đúng point, không coordinate scaling, không sensor-noise augmentation.
Loss giữ nguyên E161: exact EMD +1.0 matched pair distance, mét,128GT/pred,
16,256 ordered off-diagonal pairs; reuse cùng EMD matching, no added regularizer.
GT masks chỉ supervision/eval; không input GT segmentation/center/pose/shape params.
Best checkpoint minimum fixed SE3 validation EMD mỗi500 updates; canonical
validation diagnostic. Không test feedback. Action/flow/rollout N/A.

## Evaluation đã chạy

Best E162 vs fixed E161best18000: cùng data/SE3/seed/batch/LR/budget/loss;
chỉ thay output architecture và tăng số tham số9.7%, không matched-capacity control.
128scenes×4tasks cho IID/positions/rotations/sizes/combinations;
IID thêm4 fixed E137 SE3 views, một RGBD observation/case, K128.
Physical target-only/all-object one-local-axis×0.6/1.4: same E157 seeds/cases,
fresh RGB/depth/GT/SAM3 features, visibility/clearance exclusions được ghi.
Metrics: EMD/pair, raw directional Chamfer, centroid-alignedCD, robustXYZ extents,
coverage2/5mm, target selection, paired extent/robustextent response gain và SE3.
Viewer best vs GT dùng raw baseXYZ, tùy chọn tô màu8 patches.

Run: `.venv-concerto/bin/python -m OATFlow.surface_cloud.benchmark`, rồi
`.venv-concerto/bin/python -m OATFlow.surface_cloud.train` (tự evaluate/report).
Exact configs/hashes: config.json,train/config.json,data_config.json,features/config.json;
benchmark.json ghi checks/thời gian/VRAM. Kết quả evaluation hoàn thành ở bên dưới.

Giới hạn: một seed/camera, visible-surface GT, patches không được gán mặt trước;
không có loss ngăn overlap/collapse giữa patches. Surface bias không bảo đảm
coverage tốt hơn hay SE3 equivariance chính xác; phải đo kết quả độc lập.

## Kiểm tra trước train

Passed: upstream parameters và targethidden khớp bit-for-bit baseline seed135;128 decoderqueries giữ nguyên, aggregator invariant với permutation128hidden tokens; UV16fixed, đúng folding formula,8×16output, khởi đầu planar. Sáu GPUupdates finite; pointencoder, TransformerDecoder, patchqueries, attention, frame và các lớpdelta đều nhận gradient sau warmup preflight. Loss function đúng cùng function object E161, không thêm penalty.

Matched-budget audit passed: data/features resolve cùng files E161, seed/steps/batch/LR/warmup/SE3/normalization/validation transforms/loss/matching giống hệt. comparison_audit.json và benchmark.json ghi chi tiết. Benchmarkmedian0.143s/update (~43min18k chưa gồm validation/eval), fulltrain peak ban đầu khoảng3GiB. Không giữ preflight checkpoint/cache mới.

Log live `ssh_tmux:e162-log`, file `../e162_train.log`. Chạy từ đầu, giữ dữ liệu/checkpoints E161 nguyên vẹn. Tự evaluate/report sau18k, với best-vs-GT viewer và tùy chọn màu8patches.

## Kết quả hoàn thành

Train31.29phút, eval7.27phút; peak allocated2.99GiB. Best18000, validation EMD12.241mm.

| Test | Model | EMD mm | Pair mm | Aligned CD mm | Pred→surface mm | Surface→pred mm | Coverage5mm | Selection |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iid | E161 | 11.574 | 6.555 | 6.210 | 7.652 | 6.153 | 51.01% | 100.00% |
| iid | E162 | 11.598 | 6.900 | 6.476 | 7.642 | 6.114 | 50.76% | 100.00% |
| SE3 | E161 | 12.073 | 6.749 | 6.323 | 7.696 | 6.219 | 49.96% | 100.00% |
| SE3 | E162 | 12.182 | 7.063 | 6.561 | 7.896 | 6.278 | 48.78% | 100.00% |
| positions | E161 | 12.999 | 6.799 | 6.302 | 8.171 | 6.646 | 46.89% | 100.00% |
| positions | E162 | 12.587 | 7.045 | 6.471 | 8.078 | 6.427 | 48.43% | 100.00% |
| rotations | E161 | 12.386 | 6.585 | 6.194 | 8.010 | 6.187 | 51.02% | 100.00% |
| rotations | E162 | 12.989 | 6.983 | 6.568 | 8.225 | 6.549 | 47.94% | 100.00% |
| sizes | E161 | 15.316 | 7.406 | 6.397 | 10.337 | 8.078 | 42.94% | 99.80% |
| sizes | E162 | 14.746 | 7.884 | 6.793 | 9.852 | 7.804 | 44.53% | 100.00% |
| combinations | E161 | 11.618 | 6.686 | 6.298 | 7.606 | 6.246 | 50.48% | 100.00% |
| combinations | E162 | 11.607 | 7.020 | 6.519 | 7.696 | 6.208 | 50.08% | 100.00% |

| Physical test | Model | EMD mm | Pair mm | Coverage5mm | Extent gain | Robust extent gain |
|---|---|---:|---:|---:|---:|---:|
| target_only | E161 | 11.446 | 6.558 | 50.92% | 0.321 | 0.306 |
| target_only | E162 | 11.511 | 6.888 | 51.10% | 0.236 | 0.277 |
| all_objects | E161 | 11.422 | 6.531 | 51.14% | 0.327 | 0.313 |
| all_objects | E162 | 11.447 | 6.869 | 51.31% | 0.242 | 0.275 |

iid: E161−E162 EMD = -0.023mm, scene-bootstrap95%CI [-0.221,+0.190].

SE3: E161−E162 EMD = -0.109mm, scene-bootstrap95%CI [-0.311,+0.101].

Physical attempts1280, scored1271, exclusions9; all cases/rejections in physical_scale/cases.jsonl. Gains use full visibleGT paired deltas; ideal1. Raw metrics except explicitly named centroid_aligned_CD. Robust extents/Q95−Q5 and coverage2mm saved in archives/summary. No test selection.

Surface patches add geometric coupling without extra losses; no GT face assignment. This does not establish independent XYZ grounding or guarantee coverage. One seed,9.7%extra parameters and changing visibility remain limitations.

[Best vs GT](http://localhost:17070/surface/viewer.html) · [Learning curves](http://localhost:17070/surface/learning_results.png).

## Nhận xét sau hoàn thành

Ở cùng18k budget, chưa có bằng chứng surface folding cải thiện E161 tổng thể: IID/SE3 EMD11.598/12.182mm vs11.574/12.073mm, bootstrapCI cho chênh lệch đều gồm0. IID pair6.555→6.900mm, centroid-alignedCD6.210→6.476mm và coverage5mm51.01→50.76%. Target-only extentgain0.321→0.236, robustgain0.306→0.277. Unseen sizes cải thiện EMD15.316→14.746mm và coverage42.94→44.53%, nhưng orientation split EMD12.386→12.989mm. Selection100% ở mọi split. Một seed, tăng9.7%parameters; không kết luận surface architectures nói chung không hiệu quả.

Train vẫn giảm cuối run: mean14k–16k→16k–18k EMD13.061→12.050mm, pair7.103→6.992mm; validationmeanEMD14.311→12.562mm, pair7.063→6.949mm. Best18000 là cuối budget, chưa có bằng chứng plateau rõ.

Completion audit: checkpoint/source hashes đúng; best chọn đúng validationEMD; total=EMD+pair. E161 baseline predictions bit-identical canonical/SE3 và physical-before; physicalcase metadata/GT128 bit-identical. Một trong1271 physical-after cases có baseline rerun khác nhỏ:1/2021 scoredqueries thay đổiEMD tối đa0.1334mm, meanabsolutechange toànquery0.000066mm. Nguồn rerender/forward variation chưa tách riêng; giữ rawoutputs, không ép kết quả thành giống nhau. Tính lại EMD/pair/extent gains từ archives độc lập khớp. Xem completion_audit.json.

Browser check: viewer trả200, hình và3D load được; nút màu8patches đúng8màu×16điểm, GT giữ màu; toggles/camera sync pass, không pageerror. Xem best_viewer_browser_check.json.

Patch diagnostic trên512IIDqueries×8patches: medianRMS tới best-fitplane0.01494mm, medianprincipalRMSaxes0.01494/8.448/11.432mm; mean covariancefractions0.0000013/0.341/0.659. Chỉ0.098%patches có second-eigenfraction<1%; đa số là miếng2D gần phẳng, không chỉ linecollapse. Surface bias hoạt động ở mức patch nhưng chưa khiến tổng cloud khớp bề mặt cube tốt hơn E161. Các con số này chỉ mô tả output, không có supervision surface nào thêm vào training.
