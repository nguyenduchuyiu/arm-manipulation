# E141 — Scale vật thật trong simulator, chỉ augment SE(3)

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Pipeline physical-size cũ, được bản data nhất quán E157 thay thế. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Hoàn thành18.000 updates; chọn checkpoint bước **17500**. Không còn scaling trong coordinate augmentation. Physical target-only extent gain **0.013**; stale-feature target-only gain **-0.0000**.

**Ở budget18k này, resize thật trong simulator vẫn chưa dạy được shape tracking rõ rệt.** Physical target-only extent gain chỉ 0.013, baseline E137 0.011, so với ideal1. Paired aligned-CD improvement toàn bộ có CI dương nhưng độ lớn chỉ 0.0075mm; subset trong training support chưa cho cải thiện rõ. Không diễn giải accuracy100% thành geometry đúng.

## Data

Giữ nguyên1.792 source scene/split/index của E135: train1.024, val128, IID128, positions/rotations/sizes/combinations mỗi128; bốn task queries/scene. Scale độc lập localXYZ của **từng cube thật** Uniform(0,6;1,4), cập nhật cả visual/collision geom_size rồi render lại RGB/depth/GT và frozen SAM3 features. Body XYZ/orientation/màu, camera, robot, ghế, thảm, bàn và sàn giữ nguyên. Scale được lấy lại khi vi phạm4mm table clearance,8mm cube OBB clearance hoặc visibility<16 pixels; không đổi pose. Vì vậy scale sau rejection không còn uniform độc lập tuyệt đối. Đây là static render, không physics stepping/mass dynamics.

Render 1792 scene trong 50.8s. Camera cam_front gốc, RGB320×240; full-scene uniform4.096 points trong robot base frame, GT128 visible points/object. Full cloud reconstruct từ depth/calibration; không lưu bản trùng để tiết kiệm dung lượng. Frozen SAM3 vision_features256D/point được bilinear sample theo calibration mới kiểm tra; ảnh/point/features khớp shape thật, không giữ feature của cube cũ.

## Kiến trúc và training

PointCloudGrounder nguyên bản878.979 trainable params: XYZ×2 + LayerNorm256D features → MLP259→128→128 → TransformerEncoder2 layers/4heads → task embedding và128 learned queries → TransformerDecoder2 layers → MLP128→128→3. SAM3 encoder+neck454.038.784 frozen. GT masks/centers/sizes/metadata không vào model.

Từ đầu seed135,18k AdamW updates (~140,625 train-scene passes), batch8scenes/32queries, LR3e-4→3e-5 cosine warmup100, weight decay0,01, clip1. Sole loss: symmetric mean nearest Euclidean Chamfer, unsquared meters. BF16 autocast/FP32 weights/distances. **Augmentation chỉ Haar SO(3)+translation Uniform(−0,5;0,5)m**: P′=RP+t, G′=RG+t; features/task IDs cố định trong phép đổi frame. Không S, không auxiliary loss, normalization XYZ×2 cũ. Chọn minimum fixed-SE3 val seed137100 mỗi500 updates; clean val chỉ diagnostic. Train 9.92 phút; peak GPU 3.05GiB.

## IID/OOD và SE(3)

Cùng scene/feature mới cho model E141 và baseline E137 (chỉ train cubes gốc), hai model đều budget18k. Mỗi split128 scene/512queries. CD unsquared mm; aligned CD chỉ trừ centroid, không align rotation/scale.

| Split | E137 CD | E141 CD | E141 aligned CD | E141 centroid error | E141 accuracy |
|---|---:|---:|---:|---:|---:|
| iid | 6.925 | 6.840 | 6.897 | 9.036 | 100.00% |
| positions | 7.895 | 7.698 | 6.735 | 10.902 | 100.00% |
| rotations | 8.100 | 7.390 | 6.374 | 11.378 | 100.00% |
| sizes | 9.922 | 9.476 | 7.503 | 12.919 | 100.00% |
| combinations | 6.974 | 6.998 | 6.926 | 9.341 | 100.00% |

Size split giữ nguyên **base-side** heldout của dataset gốc; sau physical scaling, các chiều thực có thể overlap train, nên không gọi đây là heldout dimension support tuyệt đối.

E137 trên4 SE3 transforms gốc ×128 scenes×4tasks: CD 7.234mm, equivariance CD 5.031mm, accuracy 100.00%.

E141 trên4 SE3 transforms gốc ×128 scenes×4tasks: CD 7.254mm, equivariance CD 4.926mm, accuracy 100.00%.

Coordinate-global scaling chỉ là diagnostic khi test (không dùng train):

- E137: CD7.163mm, extent response gain0.018.
- E141: CD7.046mm, extent response gain0.025.

## Resize thật, fresh RGB-D/features

128 fixed IID cuboid scenes, mỗi target chọn một local axis cân bằng theo(scene+task)%3 và nhân0,6/1,4; thêm resize cả4 objects theo một local axis cân bằng. **1280 attempted renders, 1261 scored, 19 loại trước prediction do clearance/visibility**. Không thay body pose. All-objects row chấm cả4 tasks; target-only row chấm task object được resize. Full-scene point sample mới uniform. Các pixel khác có thể đổi do bóng/occlusion dù geometry distractor giữ nguyên. Visible-cloud centroid có thể đổi dù body center không đổi.

Extent=max−min visible GT/prediction theo base X/Y/Z. Gain=ΣΔE_pred·ΔE_GT/Σ||ΔE_GT||² (ideal1, stationary0; chỉ ||ΔE_GT||≥0,5mm). ACD improvement=stationary old prediction vs new GT − actual new prediction vs new GT. Baseline/reference đều lấy từ same physical IID scene trước resize.

| Test | Model | Aligned CD | Stationary aligned CD | Extent gain | Extent MAE X/Y/Z(mm) |
|---|---|---:|---:|---:|---|
| target_only (1016) | E137 | 6.7363 | 6.7366 | 0.0114 | 11.55/12.16/11.51 |
| target_only (1016) | E141 | 6.8175 | 6.8249 | 0.0129 | 11.38/12.02/11.54 |
| all_objects_all_tasks (980) | E137 | 6.6938 | 6.6991 | 0.0094 | 11.56/12.21/11.41 |
| all_objects_all_tasks (980) | E141 | 6.7815 | 6.7857 | 0.0125 | 11.47/11.99/11.45 |
| supported_target_only (512) | E137 | 6.5368 | 6.5307 | 0.0074 | 9.46/10.18/9.84 |
| supported_target_only (512) | E141 | 6.6356 | 6.6349 | 0.0128 | 9.30/9.92/9.93 |

Target-only improvement +0.00749mm, CI95% scene-cluster bootstrap [+0.00076,+0.01396]. Support row chỉ giữ pairs có tất cả actual local scale components0,6–1,4; các pair khác có thể extrapolate ra ngoài training range.

Per-axis GT/prediction changes và MAE, per-case GT/predictions đầy đủ trong `physical_pairs/*.npz`; axis strata trong `physical_pairs/summary.json`.

### +40% theo local axis: thay đổi extent base X/Y/Z

Vật có rotation giữ nguyên, nên local-axis resize có thể đổi nhiều base-axis extents. Giá trị trung bình mm, đo visible clouds:

| Local axis ×1,4 | n | ΔGT base X/Y/Z | Δprediction base X/Y/Z |
|---|---:|---|---|
| X | 170 | +13.089/+1.856/+0.484 | +0.262/+0.144/+0.241 |
| Y | 171 | +2.244/+13.968/+0.530 | +0.168/+0.268/+0.206 |
| Z | 163 | +0.832/+0.386/+13.544 | +0.167/+0.114/+0.303 |

## Control cũ: giữ feature, chỉ scale XYZ target

Lặp nguyên18.944 E139 cases trên original cube RGB/features:18.432 shape changes+512 no-op, hỏi cả4task. Đây vẫn là OOD appearance–geometry mismatch; không phải bài physical resize ở trên.

| Can thiệp | Aligned CD | Stationary aligned CD | Extent gain |
|---|---:|---:|---:|
| axis_scale | 6.9865 | 6.9873 | -0.00005 |
| mixed_scale | 7.1050 | 7.1066 | 0.00056 |
| bend | 6.9629 | 6.9629 | -0.00839 |

## Rotation-only audit

Chạy thêm2.048 queries: full-scene P′=RP quanh base origin, t=0, F giữ nguyên. Canonical và rotated inference đều FP32, TF32/autocast tắt. Centroid equivariance error **9.151mm**. Sau khi trừ centroid, CD tới rotated canonical prediction **2.312mm**, tới unrotated canonical prediction **1.470mm**; nếu giữ shape hoàn toàn đứng yên, CD tới rotated reference là 2.917mm. Chỉ 8.98% query có shape gần rotated reference hơn unrotated reference. Có một phần cải thiện so với control đứng yên, nhưng chủ yếu học quay vị trí; chưa rotation-equivariant tốt ở geometry/orientation. Đây là coordinate rotation, không phải rerender vật thật xoay trong camera. Không quy các CD thành angle error hay orientation accuracy. Artifacts `rotation_only_FP32.json/.npz`.


## Giới hạn và verification

Fresh-RGB performance không tách được đóng góp shape từ SAM3 image features và từ target XYZ. Nếu fresh test tốt nhưng stale-feature test yếu, chỉ kết luận model theo shape trong quan sát nhất quán; chưa chứng minh riêng XYZ geometry grounding. Một seed/checkpoint/budget18k, synthetic data, ảnh cố định camera và cảnh clutter gốc; không mở rộng sang real-world robustness. Accuracy nearest-GT không đủ xác nhận geometry.

2 geometry unit tests pass; audit toàn bộ1.792 source poses/orientations và lineage depth→scene/GT. SAM3 weight hash không đổi, RGB/depth/features thực sự khác ảnh gốc. CPU float64/SciPy xác nhận64 paired metrics và đo GT residual so với các mặt box đã resize thật. Pipeline gốc và mới đều có box-plane residual khoảng0,5–0,7mm ở các mẫu kiểm tra; không khẳng định analytic surface membership chính xác, giữ nguyên calibration/render pipeline để so sánh. Scale augmentation=False và state keys/shapes878.979 params được kiểm tra; SE3 test dùng đúng transform arrays E137. Xem `data_audit.json`, `features/config.json`, `verification.json`.

Code `OATFlow/physical_cloud/`; entry points `generate`, `features`, `audit`, `train`, `evaluate`, `verify`, `report`. Main artifacts `/workspace/robotics_assets/experiments/e141_simscale`; exact train/data/eval configs và source hashes lưu cùng artifacts. [Viewer before/after](http://localhost:17070/simscale/viewer.html), `results.png`, `data_samples.png`, `learning_curve.png`.
