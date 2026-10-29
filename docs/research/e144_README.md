# E144 — Overfit lại với full-scene point cloud dày hơn

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Tăng dense input/global attention chậm và không giải quyết shape; không giữ run thử dở. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái run **stopped_at_user_request**, last logged step15750 / planned18000; chọn checkpoint **15000**, input **8192 points**, output128 không đổi. Predeclared E142 shape criterion: **NOT MET**. Actual target input counts small/large **17/294** so với4/158 ở E142.

## Data, kiến trúc, training

Dùng đúng hai E142 red cube20/80mm, cùng body center/orientation/task0, cùng RGB320×240/depth/camera/robot/ghế/thảm/bàn/sàn. GT target128 points giữ bit-exact. Reconstruct full74.356 depth points, giữ nguyên4.096 input rows E142 rồi lấy thêm uniform không hoàn lại từ các pixel depth chưa dùng, seed144. Không GT mask/ROI/object-aware input sampling, không duplicate để làm dày. Masks chỉ audit target counts/supervision. Features từ cùng frozen SAM3 map, sample đúng các pixel mới;4.096 feature rows cũ bit-exact.

Chuẩn bị cache tối đa16.384 points nhưng actual train dùng prefix8192. Người dùng chọn rõ8.192 điểm/18.000 updates FP32 để giữ update budget E142 sau khi xem benchmark chi phí. Actual config là `config.json`, không suy ra N train từ full prepared cache.

Kiến trúc PointCloudGrounder878.979 trainable params nguyên bản; SAM3 encoder+neck454.038.784 frozen. From scratch seed135, fullbatch2, task0 cả hai, **không augmentation**, một symmetric unsquared Euclidean Chamfer loss. Head FP32, TF32/autocast off, original XYZ×2; AdamW LR3e-4→3e-5 cosine/warmup100, weight decay0,01, clip1. Chọn minimum mean training CD mỗi250 updates, không heldout/generalization claim. No action/flow representation.

Last logged training time 35.90 phút. Peak GPU chỉ ghi khi trainer kết thúc bình thường; run dừng theo yêu cầu nên không khẳng định giá trị peak. Old-model dense-input control chấm E142 checkpoint bằng cùng input/features8192 của model mới để tách inference density khỏi retraining.

## Kết quả

| Model / input | Sample | Raw CD(mm) | Aligned CD(mm) | Pred extent X/Y/Z(mm) |
|---|---|---:|---:|---|
| E142 /4096 | 20mm | 0.7133 | 3.5611 | 16.88/17.76/20.27 |
| E142 /4096 | 80mm | 2.1130 | 11.9126 | 72.06/74.12/61.55 |
| E142 /8192 | 20mm | 0.7725 | 3.5703 | 16.99/17.98/20.54 |
| E142 /8192 | 80mm | 2.2604 | 11.8979 | 71.37/73.51/61.64 |
| E144 /8192 | 20mm | 0.7777 | 3.3610 | 15.30/16.51/20.30 |
| E144 /8192 | 80mm | 2.0374 | 12.0914 | 71.99/74.20/61.61 |

GT extents giữ nguyên small 17.05/18.24/20.30 mm, large 78.02/76.65/79.34 mm. CD không bình phương; aligned chỉ trừ centroid mỗi cloud, không align rotation/scale. Pass criterion giữ như E142: cả2 aligned CD<2mm, mọi axis extent error<20%.

### Density và coverage

| Model | Sample | Pred→GT(mm) | GT→pred(mm) | GT within2mm | Unique nearest-GT assignments |
|---|---|---:|---:|---:|---:|
| old_model_dense_input | 0 | 0.1741 | 1.3709 | 77.34% | 56/128 |
| old_model_dense_input | 1 | 0.2961 | 4.2247 | 44.53% | 54/128 |
| dense_retrained | 0 | 0.1939 | 1.3614 | 73.44% | 57/128 |
| dense_retrained | 1 | 0.1629 | 3.9119 | 50.00% | 59/128 |

## Four-way correspondence test

Cùng canonical128 visible target locations E143 +common-background observations, tổnginputN đúng training. Target points ghép theo normalized surface q, feature lấy tạiProj(center+source_side·q), không random-token permutation; background XYZ chung, feature tensors bit-exact khi chỉ đổi geometry. GT masks/center/size chỉ dựng control, không vào model. Target-aware resampling chỉ dùng diagnostic, không training. Vẫn là RGB/geometry conflict OOD.

| F | XYZ | Pred extent X/Y/Z(mm) | Distance tới F baseline(mm) | Distance tới XYZ baseline(mm) |
|---|---|---|---:|---:|
| 0 | 0 | 15.742/17.179/20.853 | 0.0000 | 0.0000 |
| 1 | 1 | 68.065/71.999/59.144 | 0.0000 | 0.0000 |
| 0 | 1 | 15.746/17.185/20.856 | 0.0078 | 84.8988 |
| 1 | 0 | 68.058/71.996/59.140 | 0.0089 | 84.8982 |

0=small,1=large; old checkpoint được chấm trên cùng control inputs, extents lưu trong `correspondence.json`.

## Verification và giới hạn

Architecture state keys/shapes/source hash không đổi. CPU float64/SciPy xác nhận raw/centroid-aligned CD; exact RGB/depth/GT và4.096 old XYZ/features preserved. Data target counts được đo sau sampling, không dùng chọn điểm. Hai mẫu có cùng physical center nhưng visible-cloud centroid khác; aligned CD nhạy cả point density. Overfit/cross-modal controls không chứng minh generalization hay geometry-only grounding. Output K128 và loss không đổi nên input dày hơn tự nó không bảo đảm decoder coverage/density tốt.

Code `OATFlow/dense_overfit`: `prepare`, `train --points N --steps S`, `finish`; artifacts `/workspace/robotics_assets/experiments/e144_dense2`, exact `data_config.json`, `config.json`, `summary.json`, `correspondence.json`. [Viewer](http://localhost:17070/dense2/viewer.html), `learning_curve.png`.
