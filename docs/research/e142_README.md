# E142 — Overfit hai object cùng tâm, cùng task, khác kích thước

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Overfit custom encoder20/80mm không đạt shape criterion; tăng số bước không đủ chứng minh shape tốt. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Hoàn thành18000 updates, best step **17000**, predeclared shape criterion **NOT MET**. Small→large extent gain **0.857**, ideal1.

## Data và cấu hình

Hai scene chỉ khác cạnh cube đỏ20mm/80mm (4× chiều dài,64× thể tích), cùng body center world[0.54,−0.32,0.12]m, orientationidentity, màu/task ID0. Camera gốc, robot/ghế/thảm/bàn/sàn và ba cube distractor không đổi. Render RGB320×240, depth, GT mới; frozen SAM3 vision_features256D/point mới. Uniform4.096 full-scene XYZ input; target128 visible points. Cùng seed142 sampling, không chọn input theo mask. Small/large có lần lượt 4/158 target input points; chỉ mask/metadata để tạo GT/audit, không vào forward. Cùng physical center không có nghĩa visible-cloud centroid bằng nhau.

Exact PointCloudGrounder878.979 trainable params, SAM3 encoder+neck454.038.784 frozen. From scratch seed135; fullbatch2 fixed examples mỗi update; AdamW LR3e-4→3e-5 cosine/warmup100, weight decay0,01, clip1. **Không augmentation**. Head FP32, autocast/TF32 tắt; frozen features giữ precision trước đây. Duy nhất symmetric unsquared Euclidean Chamfer, meters, original XYZ×2; không GT centering/size/mask hay auxiliary loss. No action/flow representation. Không validation/heldout: chọn best mean training CD mỗi250 steps; diagnostic memorization, không phải generalization.

## Kết quả

| Sample | Raw CD(mm) | Aligned CD(mm) | Centroid error(mm) | GT extent X/Y/Z(mm) | Pred extent X/Y/Z(mm) |
|---|---:|---:|---:|---|---|
| 20mm | 0.7133 | 3.5611 | 4.7837 | 17.05/18.24/20.30 | 16.88/17.76/20.27 |
| 80mm | 2.1130 | 11.9126 | 14.8124 | 78.02/76.65/79.34 | 72.06/74.12/61.55 |

GT/pred extents=max−min visible samples, không amodal side length. Aligned CD chỉ trừ centroid từng cloud, không align rotation/scale. Predeclared strict pass: cả2 aligned CD<2mm và mọi axis extent relative error<20%. Encoder/decoder flow và density details trong `summary.json`. Unsquared Chamfer có thể thấp khi point density không khớp; chênh lệch centroid/ACD phải được đọc cùng raw-CD và coverage/extent, không chỉ dùng một số.

### Coverage và mật độ point

| Sample | Pred→GT mean(mm) | GT→pred mean(mm) | GT within2mm | Unique nearest-GT assignments |
|---|---:|---:|---:|---:|
| 20mm | 0.0896 | 1.3370 | 78.91% | 56/128 |
| 80mm | 0.1070 | 4.1191 | 45.31% | 54/128 |

Diagnostic bổ sung (không thay metric đã chọn): bbox-center aligned CD small/large 0.8327/8.9389 mm; đối chiếu với centroid-aligned CD để xem độ nhạy với mật độ point.

## Encoder→decoder và giới hạn

Encoder memory relative L2 difference 0.473869; decoder output L2 difference 0.400271m; swap memory giữa2 samples đổi prediction với max residual 0.000000000m. Task/query parameters giống nhau, nên khác biệt output đi qua scene memory. Full XYZ/F swaps chỉ là OOD diagnostics; không dùng làm tiêu chí overfit.

### Cross-modal controls (OOD)

Hai sample có cùng pixel_uv input rows; thay modality vẫn tạo RGB–geometry mismatch. Đây chỉ là probe trên hai mẫu, không phải proof không dùng XYZ nói chung.

| XYZ source | Feature source | Pred extent X/Y/Z(mm) |
|---|---|---|
| small | small | 16.876/17.760/20.267 |
| small | large | 71.984/74.061/61.540 |
| large | small | 16.876/17.759/20.263 |
| large | large | 72.061/74.117/61.546 |

Relative L2 change theo fused input / point MLP / encoder memory: XYZ-only fixed F0 0.004372/0.003121/0.002272; F-only fixed XYZ0 0.370237/0.441518/0.473489. Norm changes chỉ mô tả sensitivity, không đo trực tiếp lượng thông tin geometry.

Trong các control này size gần như theo F, dù training RGB/XYZ nhất quán. Encoder truyền được thông tin phân biệt mẫu, nhưng bài2-mẫu cho phép memorization từ RGB; chưa chứng minh metric shape của XYZ được dùng. Coverage/point-density vẫn là vấn đề riêng ở decoder/Chamfer optimization, không quy toàn bộ lỗi chỉ cho encoder.

Hai shape prediction đã khác rõ: kiến trúc có thể truyền thông tin phân biệt hai mẫu tới decoder và điều chỉnh kích thước. Điều này không chứng minh model dùng metric XYZ thay vì ghi nhớ RGB, cũng không chứng minh fitting density/orientation/generalization đã tốt.

Training 12.70 phút, peak GPU 0.43GiB. CPU float64/SciPy xác nhận raw/aligned CD; same center/task và architecture source hash được kiểm tra. Dataset siêu nhỏ và small target chỉ4 input points: overfit không phải bằng chứng geometry-only grounding.

Artifacts `/workspace/robotics_assets/experiments/e142_overfit2`, exact `config.json`, `data_config.json`, `summary.json`; code `OATFlow/overfit_cloud`. Chạy `python -m OATFlow.overfit_cloud.run`, sau đó `python -m OATFlow.overfit_cloud.finish`. [Viewer](http://localhost:17070/overfit2/viewer.html), `learning_curve.png`.
