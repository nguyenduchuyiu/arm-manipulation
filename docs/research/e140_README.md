# E140 — Full SE(3) + anisotropic global scaling

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Coordinate scaling với feature ảnh cũ không kiểm tra đúng physical shape; giữ SE(3), render lại khi đổi size. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Hoàn thành 18.000 updates; chọn checkpoint bước **18000** chỉ bằng validation RSP. Global extent-response gain **0.093**, target-only single-axis gain **-0.0017** (ideal1, không đổi0). Xem các metrics bên dưới trước khi diễn giải equivariance thành shape grounding.

## Data và kiến trúc

Dùng nguyên E135/E137: train1.024 scene/4.096 task samples, val128 scene, IID test128 scene/512 queries. Bốn cube đỏ/xanh dương/xanh lá/vàng random XYZ, orientation, size; robot, ghế, thảm, bàn và sàn vẫn có trong full scene. Camera front cố định như test trước; RGB320×240, scene4.096 points base frame, mỗi GT target128 visible points. Không tạo ảnh hay feature mới. SAM3 vision_features cache 256D/point giữ nguyên correspondence.

Đúng PointCloudGrounder cũ: concat XYZ×2 với LayerNorm(features), MLP259→128→128, TransformerEncoder2 layers/4heads, task embedding4×128 và128 learned point queries, TransformerDecoder2 layers, MLP128→128→3. **878.979 trainable**, frozen SAM3 encoder+neck **454.038.784**. Không GT center/size/mask, object tokens hay extrinsics vào forward. GT masks chỉ tạo nhãn và can thiệp khi test.

## Training

Từ đầu seed135, không warm start; cùng budget18k với E137. AdamW batch8scene/32query, LR3e-4→3e-5 cosine/warmup100, weight decay0,01, clip1; ~140,625 training-scene passes. BF16 autocast, FP32 weights và khoảng cách. Thời gian 9.88 phút; peak GPU 3.05 GiB.

Mỗi scene/step lấy R Haar-SO(3), sx/sy/sz độc lập Uniform(0,6;1,4), t độc lập Uniform(−0,5;0,5)m mỗi trục. **P′=R diag(s) P+t**, GT bốn target biến đổi y hệt; scale trước rotation, trước normalization XYZ×2. Features/task IDs cố định. Khoảng scale bị thiếu trong tin nhắn hiển thị nên 0,6–1,4 được chọn và thông báo trước khi chạy. Duy nhất unsquared symmetric Euclidean Chamfer; không auxiliary loss, không object-local augmentation, không đổi kiến trúc.

Validation128 scene dùng fixed RSP seed140100, mỗi500 updates; canonical validation chỉ theo dõi. Test không dùng chọn checkpoint. Scale làm camera-to-new là affine với linear R diag(s), không phải rigid extrinsic; model dùng feature cache nên không cần camera input hay reprojection.

## Test 1 và 3 — Global scaling, RSP và SE(3)

Mỗi bài:128 IID scene×4 transform×4 tasks =2.048 queries/model. Global scale dùng S riêng mỗi scene, R=I,t=0; RSP dùng cùng S cộng randomR/t. SE3 dùng chính4 bộ R/t của E137 để so trực tiếp, S=I. Features giữ nguyên. Equivariance CD=CD(f(F,AP),A f(F,P)); aligned equivariance CD chỉ trừ centroid, không align rotation/scale.

| Test | Model | GT CD(mm) | GT aligned CD(mm) | Eq CD(mm) | Eq aligned CD(mm) | Accuracy nearest-GT |
|---|---|---:|---:|---:|---:|---:|
| canonical | E137 SE3-only | 6.580 | 6.843 | — | — | 100.00% |
| canonical | E140 RSP | 6.530 | 6.909 | — | — | 100.00% |
| global_scale | E137 SE3-only | 6.834 | 6.904 | 2.702 | 2.088 | 100.00% |
| global_scale | E140 RSP | 6.671 | 6.949 | 2.330 | 1.961 | 100.00% |
| combined_RSP | E137 SE3-only | 7.093 | 6.981 | 5.167 | 3.324 | 100.00% |
| combined_RSP | E140 RSP | 6.912 | 6.959 | 4.878 | 3.122 | 100.00% |
| SE3 | E137 SE3-only | 6.833 | 6.901 | 4.836 | 3.155 | 99.95% |
| SE3 | E140 RSP | 6.721 | 6.902 | 4.710 | 2.964 | 100.00% |

Global extent gain so với kích thước canonical prediction sau scale (pooled dot-product gain, ideal1):

- E137: gain **0.021**, MAE extent X/Y/Z tới transformed reference 8.356/8.054/8.192 mm.
- E140: gain **0.093**, MAE extent X/Y/Z tới transformed reference 7.756/7.623/8.730 mm.

## Test 2 — Target-only scaling, centroid cố định

Lặp **đúng18.944 case E139**:18.432 shape changes và512 no-op, query cả4 tasks; cùng masks/case SHA256. Single-axis±20%/±40%, mixed-axis1,2/1,4 với reciprocal axis, quadratic bends5%/10% hệ số. Input subset và GT giữ centroid riêng, mọi distractor XYZ/feature giữ nguyên. Đây là OOD với RGB/features stale, không rerender/refresh occlusion hay chứng nhận physics.

| Can thiệp | Model | Aligned CD(mm) | Giữ prediction cũ(mm) | Extent gain | Target centroid drift(mm) |
|---|---|---:|---:|---:|---:|
| axis_scale | E137 | 6.8768 | 6.8763 | 0.00058 | 0.731 |
| axis_scale | E140 | 6.9463 | 6.9449 | -0.00172 | 0.684 |
| mixed_scale | E137 | 7.0100 | 7.0104 | 0.00155 | 0.945 |
| mixed_scale | E140 | 7.0840 | 7.0829 | -0.00089 | 0.901 |
| bend | E137 | 6.8366 | 6.8358 | 0.01010 | 0.324 |
| bend | E140 | 6.9043 | 6.9032 | 0.00286 | 0.320 |

Paired improvement=stationary aligned CD−actual aligned CD; CI95% bootstrap theo scene, không coi mọi can thiệp là mẫu độc lập:

- axis_scale: -0.00133 mm, CI [-0.00432,+0.00180].
- mixed_scale: -0.00101 mm, CI [-0.00379,+0.00198].
- bend: -0.00110 mm, CI [-0.00453,+0.00221].

### Kích thước từng trục, chỉ kéo +40%

512 target/axis, trung bình mm; chỉ ghi trục được kéo:

| Trục | GT trước→sau | E140 prediction trước→sau | ΔGT | Δprediction |
|---|---:|---:|---:|---:|
| X | 41.77→58.48 | 41.47→41.45 | +16.708 | -0.023 |
| Y | 39.99→55.98 | 45.42→45.31 | +15.995 | -0.111 |
| Z | 40.69→56.96 | 43.96→44.00 | +16.275 | +0.040 |

Extent MAE X/Y/Z tới GT mới (mm):

- axis_scale: 9.346/10.326/9.242 mm.
- mixed_scale: 10.455/11.281/10.452 mm.
- bend: 7.360/8.347/7.080 mm.

FP32 control (TF32/autocast tắt), 1152 strong changes trên16 scene đầu: single-axis/mixed/bend extent gains -0.00037/-0.00029/0.00011; kết quả và arrays đầy đủ trong `target_shape/precision_FP32.json`.

## Diễn giải và kiểm tra

**Global shape scaling cũng chưa đạt:** extent chỉ đáp ứng khoảng 9.3% thay đổi yêu cầu, so với 2.1% ở E137. CD/accuracy tốt không đồng nghĩa kích thước scale đúng. Target-only extent response gần 0 và aligned CD không cải thiện so với giữ prediction cũ. SE(3) theo các metrics cũ vẫn được giữ, nhưng E140 chưa học global shape equivariance tốt sau18k updates; vì vậy chưa thể xem global augmentation là giải pháp đã thành công rồi chỉ thiếu local generalization.

GT extents đo visible point clouds (max−min), không amodal box. Bend field áp quanh centroid riêng từng sampling như E139 nên không chứng nhận cùng một continuous warped surface. Metrics nearest-GT accuracy không đủ kết luận shape đúng. Bộ dữ liệu synthetic cố định, không thêm sensor noise trong training; không mở rộng phát biểu sang real robot.

Kiểm tra:2 transform unit tests, shared scale-before-rotation/target transform/features pointer; state keys/shapes và878.979 params trùng E137; SAM3 weight hash giữ nguyên. CPU NumPy float64/SciPy xác nhận CD/centroid/extents/equivariance trên96 global queries và37 local operations; protocol SE3 và target-only giữ đúng bản cũ. Xem `verification.json`.

Code `OATFlow/scale_cloud/`; chạy `python -m OATFlow.scale_cloud.train`, rồi `evaluate`, `verify`, `report`. Main artifacts `/workspace/robotics_assets/experiments/e140_rsp`; exact `train/config.json`, `evaluation/summary.json`, `target_shape/summary.json`. Frozen SAM3/data/features dùng lại E135. [Viewer](http://localhost:17070/rsp/viewer.html), `results.png`, `learning_curve.png`.
