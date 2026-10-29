# E153 — radius1,2m + FPS16k, PTv3 + EMD, coordinate RSP

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: PTv3+RSP18k không học grounding tốt: IIDEMD116.34mm, selection27.73%. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành18k và evaluation, best17500.

User đã duyệt preview E152 và yêu cầu train. **Scale là augmentation của tọa độ XYZ nhưSO(3), không scale vật thật:** dữ liệu simulator/RGB/depth/GT gốc không đổi, không rerender hoặc chỉnh cube size. Toàn scene point input và GT cùng `P′=R diag(s)P+t`, `G′=R diag(s)G+t`; SAM3 feature-point association/task ID giữ nguyên.

## Data và preprocessing

Giữ đúng dataset E140/E135: train1.024 scene, val128, IID128; positions/rotations/sizes/combinations128 mỗi split. Cam_front320×240,4 màu/task0:red/1:blue/2:green/3:yellow; robot và clutter gần còn trong point cloud. GT128 points/task lấy nguyên nhãn cũ, có replacement khi visible points<128 như dataset gốc.

Canonical full74.356 XYZ base-frame → `||XYZ||₂≤1,2m` →69.155 points (audit toàn1.792 scene giữ0 target points bị loại) → exact greedy EuclideanFPS16.384, start first retained row. Chọn điểm chỉ bằng XYZ, không GT/RGB/task/object center/size. FPS cache chỉ original row indicesuint32, không copy RGBD hoặc feature-perpoint dataset. Uniform full-scene đã thay bằng radius+FPS theo yêu cầu user.

**Thứ tự:** radius filter → canonical FPS → original-UV SAM3 feature sampling → scale → rotation → translation → model XYZ×2. Giữ indices/UV để feature correspondence không đổi; tuyệt đối không reproject augmented XYZ bằng camera cũ. Crop/FPS không chạy lại sau RSP.

## Model và train dự kiến

Kiến trúc giữ nguyên E149: official PTv3 Mode1 U-Net cls_mode=False, grid1mm, patch64, non-Flash, per-point64D→projection128D→128-query task decoder. Backbone47.178.272, tổng47.684.771 trainable; **PTv3/head từ đầu seed135**, không pretrained PTv3, không warm-start các checkpoint2-example/fullcloud. Frozen pretrained SAM3 vision encoder+neck454.038.784; reuse E150 FP16 maps256×72×72, same weights hash6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a; no prompt/GT-feature input.

18.000 updates, batch dự kiến2 scenes/8 tasks, encode mỗi scene một lần, decode4tasks. Chỉ **exact EMD128↔128 mean unsquared Euclidean**, assignment1–1 và gradient FP32, không auxiliary loss. AdamW3e-4→3e-5 cosine/warmup100, weight_decay.01, clip1. BF16 attention/MLP; FP32 sparse conv, XYZ head, params/optimizer/loss; TF32 off. Batch/runtime sẽ được xác nhận bằng benchmark16k. Khác E150 microbatch1/accumulation2, E153 dự kiếnbatch2 nếu fitsGPU; BatchNorm official giữ nguyên.

Augmentation seed140, Haar-uniformSO(3), independent sx/sy/szUniform[0,6;1,4], independent tXYZUniform[−0,5;0,5]m; per scene/update, cùng phép cho cả4GT. Voxel metadata tính lại sau RSP; không GT center/size/mask/token vào model. E140 gốc có878.979 params,4k points,batch8scene32task,Chamfer; comparison không matched capacity/compute/batch.

Best theo EMD trên128val scenes với chính fixedRSP transforms E140, mỗi500updates; cleanval diagnostic only. Không dùng test chọn best. Giữ best và1 resume checkpoint model+optimizer/RNG; không weight history.

## Evaluation sau train

HeldoutIID128scene/4task. Canonical, global scale, combinedRSP, SE3 dùng đúng4 transforms/scene lưu E140. Target-only shape dùng18.944 cases E140/E139, masks lấy theo FPS original indices; GT chỉ dựng/scoring counterfactual, không model input. Scope staleRGB/feature OOD, không physical rerender. Legacy shape helper recenter input subset, GT128 và full-visible surface riêng theo empirical centroid như E140; không khẳng định cùng continuous physical deformation.

Đo exactEMD, directional CD riêng hai chiều vớiGT128 và full-visibleGT, Q95−Q5XYZ/max−min, voxel4mm surface coverage tại2/5mm cùngGT128 reference; target selection, equivarianceCD, global/local extent-response gain và other-object drift. Reference surface là toàn bộ visible target points gốc, không lấy FPS subset làm GT coverage. Không diễn giải alignedCD là shape error thuần.

Artifacts `/workspace/robotics_assets/experiments/e153_fps_rsp`, source `OATFlow/ptv3_fps_rsp`; cached SAM3 maps dùng chung E150, datasetRGBD/GT dùng chungE135. Log trong window riêng `ssh_tmux`, remain-on-exit on, stdout/stderr tee file. Exact FPS index audit, benchmark, train/config.json và evaluation/summary.json cập nhật theo tiến độ thực tế; hiện chưa có kết quả train.

## Launch và verification thực tế

FPS index cache1792scene hoàn tất336,25s, first-scene indices khớp exact E152 đã duyệt; train1024scene không mất object, min10/median70 target points sauFPS, IIDmin19. Cache indices ~112MiB, SAM3 maps dùng chungE150 không tạo lại. Benchmark batch2scene/8task/khôngaccum: randomRSP0,419s/update peak7,21GiB; scale1,4 upper-bound sample0,505s peak8,73GiB. Ước tính18k train~2,09h trước validation/eval overhead. Actual first400updates~0,43s/update, peak8,3GiB.

Đã chạy training từ đầu trong `ssh_tmux:5:e153-log`, Ctrl+B rồi5; `remain-on-exit on`, stdout/stderr qua tee `train.log`. Chưa có kết quả18k tại thời điểm launch. Model source/params giữ nguyên, pretrained PTv3 không dùng; source hashes trong train/config.json đã xác nhận không đổi. Preflight: RSP giữ nguyên tensorfeatures, scene/4GT shared transform và inverse; full forward/backward finite gradients; FPSXYZ/UV đúng original indices, GT128 byte-identical, coverage reference fullvisibleGT thay vìFPSsubset, localcounterfactual chỉ thaytargetXYZ/giữcentroid. File `pipeline_verification.json`, `benchmark.json`, `fps/config.json`. **Không thay size/pose simulator hoặc render lạiRGB/depth: S chỉ coordinate augmentation nhưR.**

## Kết quả hoàn thành

**Kết luận:** cấu hình này chưa học được grounding tốt: IID EMD116,34mm, target selection27,73% (4choices), coverage5mm0,10% soGT128reference92,21%; combinedRSP EMD131,43mm/selection28,81%. Global-scale extentgain0,031 và target-onlyaxisgain≈−0,00011. Canonical baseline đã sai nhiều nên không dùng riêng local-shape failure để kết luận sâu về geometry branch. Loss giảm không đồng nghĩa giải quyết được RSP.

Thời gian train2.40h, eval0.57h; peak allocated8.50GiB. Best chỉ chọn bằng fixed RSP validation EMD, không test-driven.

| IID test | Queries | EMD(mm) | Pred→visibleGT(mm) | visibleGT→pred(mm) | Coverage5mm | GT128 reference5mm | Target accuracy | EqCD(mm) | Extent gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | 512 | 116.337 | 91.027 | 105.104 | 0.10% | 92.21% | 27.73% | — | — |
| global_scale | 2048 | 118.198 | 91.990 | 107.030 | 0.13% | 91.37% | 28.81% | 12.677 | 0.031 |
| combined_RSP | 2048 | 131.433 | 105.149 | 119.727 | 0.14% | 91.47% | 28.81% | 57.708 | 0.156 |
| SE3 | 2048 | 127.992 | 102.619 | 116.445 | 0.16% | 92.43% | 30.13% | 53.890 | 0.538 |

Robust extents Q95−Q5 XYZ trong JSON theo từng case và mean trong summary; không dùng alignedCD làm shape error thuần. Coverage chỉ surface nhìn thấy, voxel4mm/tolerance2/5mm, GT128 reference cùng point budget.

| Local variant | Cases | Pred→visibleGT(mm) | VisibleGT→pred(mm) | Stationary visibleGT→pred(mm) | Coverage5mm | Extent gain | Robust extent gain | Other centroid drift(mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sham | 512 | 91.027 | 105.104 | 105.104 | 0.10% | — | — | 0.000 |
| axis_scale | 6144 | 91.134 | 105.157 | 105.152 | 0.10% | -0.000 | -0.000 | 0.600 |
| mixed_scale | 6144 | 90.649 | 105.257 | 105.249 | 0.11% | -0.000 | -0.000 | 0.839 |
| bend | 6144 | 90.911 | 105.109 | 105.109 | 0.10% | -0.001 | -0.001 | 0.209 |

Exact E140 stale-feature OOD variants. Input subset, GT128 and visible-GT surface each preserve their own empirical centroid, as the legacy protocol; not one shared continuous physical deformation.

E140 cũ dùng4096 input và878979params, Chamfer, batch8scene; E15316k FPS/radius/model/loss/batch đều khác nên không quy thay đổi cho riêngEMD hoặc kiến trúc. Một seed, synthetic scenes/camera cố định, chưa đánh giá real robot. GlobalRSP pass không tự chứng minh object-local metric geometry grounding; đọc target-only extent/coverage.

[Viewer](http://localhost:17070/fps-rsp/viewer.html), `evaluation/summary.json`, `evaluation/target_shape.npz`, `train/config.json`, `fps/config.json`. Source `OATFlow/ptv3_fps_rsp`; pretrained SAM3/fixedRGB và indices giữ nguyên.

## Kiểm tra hoàn thành

CPU float64/SciPy xác nhận exactEMD và hai chiềuCD trên15case thuộc4suite trong0,001mm; mean archive khớp summary, target-selectionaccuracy tính lại khớp, checkpoint SHA256 và toàn bộ training-source hashes khớp. Local18.944cases giữ nguyên XYZ distractor, centroid sai tối đa5,22e-9m. `completion_audit.json`. Raw cases.jsonl giữ metadata moved_input_points từE1404k để bảo toàn case hash; actualFPS masks/counts dùng khi test nằm trong `evaluation/actual_FPS_target_counts.json`. Lỗi thiếu Path import ởreport-only đã sửa và chạy lại report; không chạy lại model/train/test. Viewer/PNG/summary trảHTTP200.
