# E157 — physical simulator scaling + SE(3), original encoder + EMD

> **Code giữ lại:** [code/README.md](/workspace/robotics_assets/experiments/e157_sim_emd/code/README.md). data: consistent physical resize/RGBD/SAM3 generation and loaders. [Danh mục/bài học](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành18k và evaluation; best16500.

Mục tiêu: train với RGB/XYZ/features nhất quán khi vật thay đổi kích thước thật. Giữ encoder gốc878.979params,4096inputpoints,128outputpoints/task, EMD duy nhất như E156; thay coordinate-RSP bằng **physical per-object geometry scaling trong simulator + coordinate SE(3) augmentation**. Thất bại ở scale với stale features chỉ chứng minh phản ứng yếu trên OOD counterfactual, không đủ kết luận encoder không encode geometry.

## Data generation

Render mới1792scenes: train1024,val128,IID/positions/rotations/sizes/combinations mỗi128, cùng source scene IDs/poses/colors của E135. Camera cam_front320×240, K/T, robot/ghế/thảm/bàn/sàn giữ nguyên. Dùng renderer đã kiểm tra E141; **tạo bộ dữ liệu mới với seed157000000+split×100000+index**, không reuse RGB/features cũ.

Scale độc lập localXYZ từng cube: sx/sy/sz Uniform[0,6;1,4], sửa **visual và collision geom_size** trước mj_forward/render. Cube body center/quaternion cố định. Resample scales khi vi phạm4mm table clearance,8mm OBB clearance hoặc visibility<16pixels/object; phân phối sau rejection không còn uniform độc lập tuyệt đối. Static render, không stepping dynamics hoặc chứng nhận mass/inertia mới.

Mỗi scene render mới RGB, optical-Z depth và GT instance labels. Backproject depth về robot base, lấy4096point full-scene uniform không masks/crop/FPS; GT128visiblepoints/object. Compact lưudepth/calibration thay duplicatefullcloud; full visibleGT dựng lại khi chấm. Audit originalposes/dimensions/depth→input/GT/projection. GT masks chỉ làm nhãn hoặc thiết kế/chấm physical test, không đưa vào model.

Trích lại frozen pretrained SAM3 vision_features256×72×72 trên **RGB mới**, sample tại pixel tương ứng depth mới; same pretrained weights SHA6d06f0a5… và same model revision như E156. Cache per-point FP16 toàn1792scene (~3,5GiB). Không đưa cache feature cũ E150 vào model lượt này.

## Architecture/training

Unchanged `OATFlow.task_cloud.model.PointCloudGrounder`: XYZ×2+LayerNorm(SAM3feat256), MLP259→128→128, sceneTransformer2layers/4heads/FF384, taskembedding4×128,128learnedqueries, decoder2layers và XYZMLP128→128→3. 878.979trainable từ đầu seed135, không warm-start/pretrained3D; chỉ SAM3 pretrained454.038.784 đóng băng. BF16 autocast như encoder gốc; FP32 weights/optimizer/EMD. Không GT center/size/masks/object tokens vào forward.

18kupdates, batch8scene/32queries (~140,625train-scene passes), AdamW3e-4→3e-5 cosine/warmup100, wd0,01, clip1. **Chỉ exact equal-mass EMD128×128**, mean unsquaredEuclidean optimal1-to-1 assignment; no auxiliary loss/Sinkhorn. SE3 augmentation như E137: HaarSO(3), tXYZUniform[−0,5;0,5]m, CUDAaugRNG137; P′=RP+t/GT y hệt trướcXYZ×2, feature-point association giữ nguyên. **Không scale XYZ trong training augmentation.** Physical sizes đã nằm trong fresh render; sampling mỗi scene cố định khi data generation, không simulator rescale on-the-fly mỗi batch.

Fixed128scene SE3validation lấy nguyên R/t E137, s=1; best theo validationEMD mỗi500updates, canonicalval diagnostic. DataRNG135, log50steps, giữ best+resume1000steps. Test không chọn checkpoint. Window riêng trong attachedssh_tmux, remain-on-exit on, main stdout/stderr tee log; không đổi window người dùng.

## Evaluation dự kiến

Canonical heldout IID/positions/rotations/sizes/combinations trên fresh-render data. SE3 heldoutIID4transforms×128scene×4task giữ t/R E137, không scaleXYZ. E156 best17500 làm baseline, chấm cùng fresh pointcloud/features khi phù hợp; báo rõ dataset/augmentation thay đổi nên không coi là loss-only ablation.

**Scale test riêng dùng simulator:** trên128IIDsource poses, fresh baseline render với scales=1; target-only scale một localaxis cân bằng theo(scene+task)%3, factor0,6/1,4, các vật khác giữgeometry; thêm all-object scale một localaxis cân bằng, cùng factors. Tất cả absolute scales trong training range[0,6;1,4]. Giữ bodycenters/orientations/camera, render RGB/depth/GT mới và chạy frozenSAM3 mới cho mỗi variant. Ghi mọi physical/visibility rejection, không tự di chuyển vật. Visible-cloud centroids/input rows/occlusion có thể đổi tự nhiên. Baseline E156 và model E157 dùng cùng observations, không stalefeature test làm primary.

Metrics rawcoordinate EMD, directionalCD pred→GT/GT→pred (GT128 và fullvisibleGT), robust Q95−Q5 XYZ, extentresponse và coverage visibleGT voxel4mm/tolerance2/5mm với GT128reference, nearestGT selectionaccuracy. Physical paired extent gains dùng mean/matched deltas, tách vị trí vàshape bằng cả rawcentroid error và robustextent; không căn GTcentroid để trợ giúp prediction. Không quy CD thành angle accuracy, không suy ra geometry branch đã học độc lập SAM3 chỉ từ freshRGB success. Không noise augmentation; action horizon/flow steps không áp dụng.

Artifacts `/workspace/robotics_assets/experiments/e157_sim_emd`, code `OATFlow/simscale_emd`; immutable data_config.json, features/config.json, train/config.json, stage logs/progress. Canonical documentation tại docs/research/e157_README.md. Tại thời điểm chuẩn bị chưa có kết quả cuối; xem phần hoàn thành và diễn giải bên dưới.

## Launch thực tế

Đã render mới1792scenes trong51,03s, data_audit.json pass (sourceposes giữ nguyên, size metadata đúng, depth→input/GT/projection error<1e-3px). Render-change audit21scenes xác nhậnRGB/depth đổi, GTsurface residual tới box đãresize mean0,493mm/p950,695mm, so oldunscaledbox3,243/7,418mm; gồm sai số calibration/rendering gốc, không khẳng định residual0. FrozenSAM3 trích lại toàn bộ1792freshRGB,323,53s; sameweightsSHA vàfrozenparameters, cache3,5GiB. Feature-change audit ởtrain/val/IID xác nhận freshfeatures khác source-imagefeatures tại cùng pixels.

Đã chạy18k fromscratch trong `ssh_tmux:4:e157-log` (Ctrl+B rồi4); stdout/stderr tee train.log. Benchmark6updates finite vàsharedSE3/features/inverse/distance checks pass; 0,138s/update, batch8scene/32queries; estimate~41phút trước validation/eval. Model878.979params source/state shapes giữ nguyênE156; coordinate scale tắt. Training qua250updates, last50EMD210,10mm, peak2,99GiB; đây là snapshot ban đầu, chưa kết luận hội tụ hoặc có test cuối.

Original uniform4096sampling giữ nguyên: toàn1792scenes/7168tasks có2queries không có point input thuộc target, không bị lọc/resample bằngGT. Thống kê từngsplit tronginput_point_counts.json. FullvisibleGT/GT128 vẫn có đủ target; đây là giới hạn của sparse full-scene sampling, không giấu trongdatafilter.

## Kết quả hoàn thành

Train29.67phút, eval6.89phút; peakallocated2.99GiB. Best theoSE3validationEMD, checkpoint16500. Full exact configs/hashes trong train/config.json, data_config.json và features/config.json.

| Test | Model | EMD(mm) | Pred→visibleGT(mm) | VisibleGT→pred(mm) | Coverage5mm | GT128 reference5mm | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| iid | E156 | 12.099 | 7.389 | 6.972 | 50.65% | 91.96% | 100.00% |
| iid | E157 | 11.631 | 7.218 | 6.768 | 52.47% | 91.96% | 100.00% |
| SE3 | E156 | 12.703 | 7.780 | 7.118 | 48.33% | 92.13% | 100.00% |
| SE3 | E157 | 12.798 | 7.976 | 7.248 | 47.69% | 92.13% | 100.00% |
| positions | E156 | 13.759 | 8.201 | 7.819 | 45.02% | 90.61% | 100.00% |
| positions | E157 | 13.407 | 7.933 | 7.564 | 47.01% | 90.61% | 100.00% |
| rotations | E156 | 13.861 | 8.505 | 7.375 | 47.12% | 91.20% | 100.00% |
| rotations | E157 | 13.191 | 8.091 | 7.018 | 50.19% | 91.20% | 100.00% |
| sizes | E156 | 17.584 | 11.504 | 9.901 | 47.25% | 87.87% | 99.80% |
| sizes | E157 | 16.621 | 10.654 | 9.363 | 51.81% | 87.87% | 100.00% |
| combinations | E156 | 12.560 | 7.522 | 7.419 | 48.14% | 91.32% | 100.00% |
| combinations | E157 | 11.759 | 7.233 | 7.023 | 51.36% | 91.32% | 100.00% |

SE3 làwhole-scene coordinate transform sau fresh render, không scaleXYZ. Rawmetrics, không centroid alignment. RobustextentXYZ và coverage2mm có trong JSON/archives.

Physicalscale attempted1280, scored1271, excluded9; mọi exclusion được ghi trong cases.jsonl.

| Physical resize | Model | Queries | EMD(mm) | Coverage5mm | Extent gain | Robust extent gain |
|---|---|---:|---:|---:|---:|---:|
| target_only | E156 | 1021 | 11.866 | 51.64% | 0.007 | 0.005 |
| target_only | E157 | 1021 | 11.485 | 52.78% | 0.008 | 0.007 |
| all_objects | E156 | 1000 | 12.015 | 51.13% | 0.009 | 0.008 |
| all_objects | E157 | 1000 | 11.466 | 53.08% | 0.008 | 0.007 |

E156 target-only: visibleGT→prediction improvement so giữ prediction trước resize-0.105mm, scene-bootstrapCI95[-0.175,-0.032].

E157 target-only: visibleGT→prediction improvement so giữ prediction trước resize+0.025mm, scene-bootstrapCI95[-0.043,+0.106].

Extent/robustextent response gains chấm paired deltas tới fullvisibleGT, không dùng amodal size làm modelinput. Fresh views có thể đổi visibility/centroid/input sampling; clearance/visibility rejection làm tập scored có selectionbias. Một seed, syntheticcam cố định. FreshRGB success chứng minh mô hình theo observations nhất quán; chưa tách riêng đóng góp XYZ geometry khỏi SAM3 hình ảnh. E156 baseline có trainingdata/augmentation khác E157.

[Viewer](http://localhost:17070/sim-emd/viewer.html). Checkpoints/data/features cũ giữ nguyên; không còn stale-coordinate-scale test trong primary physicalscale evaluation.

## Diễn giải sau completion audit

**Render thật và refresh features chưa giải quyết được scale response trong lượt18k này.** E157 chọn đúng target100% trên freshIID vàSE3; IIDEMD11,63mm, SE3EMD12,80mm. Cùng freshIID observations, E15612,10mm/50,65%coverage5mm; E15711,63mm/52,47%. Chất lượng grounding giữ tốt và cải thiện nhẹ trên canonicalfreshdata, nhưng shape vẫn gần kích thước cố định. GT128reference coverage5mm91,96%.

Physical target-only1.021queries, cùng observations mới cho hai model: E156 extentgain0,00689/robustgain0,00504; E1570,00796/0,00710 (ideal1). All-object250renders×4tasks=1.000queries: E157 extentgain0,00755/robustgain0,00730. Scale observations đã nhất quán RGB/XYZ/GT/features nên không quy failure này cho staleSAM3features. Chưa phân định lỗi encoder/fusion/decoder hoặc thiếu training chỉ từ kết quả một seed này.

Để tránh cancellation giữa shrink/expand, riêng physicalscale×1,4:

| Local axis được kéo | Cases | Δ robust visibleGT extent XYZ(mm) | Δ robust prediction extent XYZ(mm), E157 |
|---|---:|---|---|
| X |171|12,292 /1,171 /0,398|0,243 /0,182 /0,150|
| Y |171|1,364 /13,117 /−0,350|0,186 /0,254 /0,175|
| Z |167|0,606 /−0,004 /13,172|0,118 /0,067 /0,069|

Các delta đo trongbaseXYZ, trong khi can thiệp theo localaxis từngbody; không đồng nhất chúng thành cùng trục tuyệt đối. Rawpredictions giữ nguyên, không centroid alignment. E157 paired improvement visibleGT→prediction so giữ prediction trước resize chỉ+0,025mm, CI95[−0,043;+0,106]mm: chưa có cải thiện rõ về metric này.

Train29,67phút, eval6,89phút, best16500/18000 theoSE3validationEMD12,920mm; finalval13,043mm. Bộphysicaltest attempted1.280, scored1.271, excluded9clearance/visibility failures ghi rõ trongcases.jsonl. Sizes đều trongtrainingrange; đây làmatchedphysical-observation test, không phải coordinate-scale augmentation.

Completionaudit pass: checkpoint/source hashes khớp, beststep đúngvalidation, s=1 vàsameE137validationR/t. Recompute toàn bộglobal/physicalsummarymeans vàextentgains;24CPUfloat64globalqueries maxmetricerror1,21e-6mm. Independent simulator rerender4physicalcases tạo GT128 bit-identical, fullvisibleGTdelta khớp archive; exampleRGB tái tạo bit-identical. Viewer HTTP200. Xem completion_audit.json; primary metrics và rawarrays lưu đầy đủ.
