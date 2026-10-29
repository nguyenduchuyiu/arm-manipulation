# E148 — PTv3 BF16, 18k bước, full cloud

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Chamfer dễ tụ điểm, thiếu coverage/Z; giữ comparison và metrics để đối chiếu EMD. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành 18.000 bước; best tại 16600. Thời gian 2.91 giờ.

- Data: đúng hai frame E142, cube đỏ 20/80 mm cùng tâm và orientation, giữ robot, ghế, thảm và toàn cảnh. Dùng `e146_local/data.npz` và `features.npy`: 74.356 điểm/frame, target hiện diện 150/2.624 điểm; GT output 128 điểm. Không crop, không augmentation, không dùng mask/center/size làm input. Hai mẫu dùng để overfit, không có held-out test.
- RGB: camera `cam_front`, 320×240 như E147. SAM3 `vision_features` 256D đóng băng, cache FP16 không đổi; chỉ backbone 3D và head được train từ đầu, seed 135.
- Kiến trúc giữ nguyên E147: official PTv3 Mode1 U-Net `cls_mode=False`, grid 1 mm, patch 64, non-Flash; output feature 64D cho từng input point qua inverse voxel map; projection 128D, hai decoder layers với 128 queries. Tổng 47.684.771 tham số trainable. Official commit `3229e9b7de1770c8ad17c316f8e349982de509f8`.
- Train: 18.000 updates từ đầu, batch 2 scene, AdamW 3e-4 → 3e-5 cosine, warmup 100, weight decay .01, gradient clip 1. Chỉ một symmetric unsquared Euclidean Chamfer loss, đơn vị m. Best chọn theo mean training CD mỗi 100 bước; kiểm tra shape bằng centroid-aligned CD và extents XYZ.
- Precision: BF16 autocast cho attention/MLP. FP32 master parameters, optimizer, sparse convolution, XYZ head và Chamfer. `spconv` hiện tại tự ép FP16 dưới autocast, vì vậy sparse conv được chạy FP32 tường minh. Head FP32 tránh lượng tử hóa tọa độ BF16 ở mức mm. Không GradScaler; TF32 tắt. Không sửa source official PTv3 hoặc số tham số.
- Benchmark: 6 bước full cloud, bỏ warmup bước đầu khi lấy median; kiểm tra loss/gradient hữu hạn, dtype thực tế và so sánh FP32 trước/sau precision wrapper. Đây là kiểm tra ngắn, không phải convergence test.
- Checkpoints: giữ `best.pt` và một `resume.pt` có optimizer/RNG để tiếp tục nếu ngắt; không lưu lịch sử hàng loạt. `resume.pt` ghi mỗi 1.000 bước; best mỗi 100 bước nếu CD giảm.
- Test dự kiến: evaluate best trên đúng hai mẫu ở BF16 và FP32; so raw CD, centroid error, aligned CD, extents với E147 FP32 1k. Hai run khác số bước nên không dùng để kết luận BF16 tốt hơn FP32.

Artifacts: `/workspace/robotics_assets/experiments/e148_bf16`; benchmark `benchmark.json`, cấu hình train `train/config.json`, tiến độ `train/progress.json`, metric history `train/metrics.jsonl`. Chưa chạy lại các phép can thiệp chéo feature/XYZ.

Chạy từ `/workspace/arm-manipulation`:

```
.venv-concerto/bin/python -m OATFlow.ptv3_cloud.bf16
.venv-concerto/bin/python -m OATFlow.ptv3_cloud.train_bf16
```

## Dọn ổ đĩa trước run

Đã giải phóng 36.186.574.848 bytes (33,70 GiB), từ khoảng 231 MiB lên 33,93 GiB trống: cache feature/noise của E135/E136/E141, checkpoint `latest/final` khi có `best`, cache kNN và weights của custom model E146 đã hủy, pip download cache. Giữ RGB/depth gốc, pretrained SAM3/Concerto, best các baseline còn dùng, nguồn feature đang train và báo cáo/config. Danh sách chính xác: `/workspace/robotics_assets/cleanup_20261025.json`. Xóa cache trên đĩa không làm giảm VRAM activation khi train.

## Cập nhật khởi chạy

Đã khởi chạy training trong tmux `e148`, log `train.log`. Benchmark full cloud: 0.578 s/bước, peak 15.52 GiB; ước tính 18k ~2.89 giờ, chưa tính toàn bộ overhead eval/checkpoint. Dtype audit và FP32 wrapper parity đã pass. Trạng thái hiện tại nằm trong `train/config.json` và `train/progress.json`; chỉ `summary.json` được tạo sau khi hoàn thành/evaluate checkpoint best.

## Kết quả checkpoint best

| Precision inference | Cube | CD (mm) | Centroid error (mm) | Aligned CD (mm) | Extents XYZ (mm) |
|---|---|---:|---:|---:|---|
| BF16 | 20 mm | 0.678 | 4.063 | 3.415 | 16.84/16.52/20.24 |
| BF16 | 80 mm | 2.030 | 15.512 | 12.226 | 74.88/76.95/61.50 |
| FP32 | 20 mm | 0.677 | 4.064 | 3.416 | 16.86/16.50/20.25 |
| FP32 | 80 mm | 2.025 | 15.510 | 12.229 | 74.92/76.78/61.53 |

Đây chỉ là kiểm tra overfit hai mẫu, chưa chứng minh generalization hoặc shape grounding bằng metric XYZ. Các phép chéo feature/XYZ và phản thực chưa chạy lại.

## Nhận xét sau hoàn thành

18.000 bước hoàn tất trong 2 giờ 54 phút, best 16.600, peak allocated 15,52 GiB. Raw CD BF16 0,678/2,030 mm; centroid-aligned CD 3,415/12,226 mm. Cube lớn vẫn thiếu chiều Z: 61,50 mm so với GT visible extent 79,34 mm (thiếu 22,5%). Full-cloud PTv3 chưa giải quyết triệt để shape/density bias trong lượt overfit này. BF16/FP32 inference của cùng checkpoint gần nhau: chênh raw CD lớn nhất 0,0052 mm; không phải so sánh hai lượt training khác precision. CPU float64/SciPy kiểm lại cả raw/aligned CD trong 0,001 mm và checkpoint hash khớp (`completion_audit.json`).

## Vì sao centroid alignment làm CD tăng

Centroid ở đây là trung bình các điểm, không phải tâm hình học của cube. Prediction phân bố không đều: small có 64/128 predicted points cùng nearest GT point; large 58/128. Đây là nearest-neighbor assignment, không khẳng định tọa độ trùng tuyệt đối. Large prediction chỉ sử dụng 59/128 nearest GT indices. Raw pred→GT 0,141 mm nhưng GT→pred 3,920 mm: predicted points nằm sát một phần GT, coverage không đều. Centroid lệch 15,512 mm do phân bố điểm; trừ lệch centroid này dịch cả prediction khỏi bề mặt đang gần đúng, làm CD tăng từ 2,030 lên 12,226 mm. Small tương ứng centroid lệch 4,063 mm, CD 0,678→3,415 mm. Centroid matching không tối ưu Chamfer. Vì thế aligned CD trong trường hợp sampling/collapse này không phải lỗi shape thuần tách biệt vị trí; phải đọc cùng directed CD, coverage, phân bố điểm và extents. Không diễn giải centroid error trực tiếp thành sai tâm vật lý.

## Hình centroid alignment

`alignment.png` và [interactive view](http://localhost:17070/ptv3-bf16/alignment.html): hai mẫu, đúng saved BF16 prediction best16.600 và GT128, không inference/resample. Cùng origin GT/cùng tỷ lệ cho raw và aligned trong mỗi hàng. Đánh dấu centroid/cụm cùng nearest GT và đường CD theo fraction dịch centroid. Slider interactive 0→1 dịch prediction bên phải, raw bên trái giữ nguyên. Code `OATFlow/ptv3_cloud/alignment_plot.py`; số liệu `alignment_metrics.json`.

## Ba metric trên cloud gốc: directional CD, robust extent, coverage

Dùng đúng saved BF16 best16600, prediction không căn tâm/rotate/scale/resample. `GT128` là training target cũ; `visible_GT` là toàn bộ150/2624 target points nhìn thấy trong full-depth cloud, GT label1 chỉ dùng để đo. Directional CD là mean Euclidean unsquared mm.

| Cube | Pred→GT128 | GT128→pred | Pred→visibleGT | visibleGT→pred |
|---|---:|---:|---:|---:|
| 20 mm | 0.102 | 1.253 | 0.102 | 1.518 |
| 80 mm | 0.141 | 3.920 | 0.135 | 7.550 |

Robust extent Q95−Q5, NumPy linear quantile, cùng tọa độ base. Đây là extent theo phân bố điểm; tụ điểm cũng làm quantile co lại, nên không đồng nhất với kích thước vật lý.

| Cube | Cloud | Q95−Q5 XYZ (mm) | Z max−min (mm) |
|---|---|---|---:|
| 20 mm | prediction | 14.100/13.465/6.347 | 20.241 |
| 20 mm | GT128 | 16.809/18.236/20.299 | 20.299 |
| 20 mm | visible_GT | 16.823/18.236/20.299 | 20.299 |
| 80 mm | prediction | 57.551/60.700/18.160 | 61.502 |
| 80 mm | GT128 | 70.258/70.875/61.347 | 79.342 |
| 80 mm | visible_GT | 71.832/69.292/68.446 | 79.342 |

Coverage: GT occupied voxels4 mm, origin=visible_GT.min XYZ, mỗi voxel chọn GT sample gần mean trong ô nhất làm đại diện. Covered nếu đại diện cách ít nhất1 predicted point ≤τ. Mỗi vùng trọng số1, không dùng số predicted points trong vùng. Chỉ surface nhìn thấy, không đánh giá mặt bị che. GT128 tham chiếu đo cùng quy trình để nhận biết giới hạn128 output points; không phải cận trên tối ưu. Có sensitivity voxel2/8 mm trong JSON, và5 binsZ độ rộng bằng nhau (bin trống null).

| Cube | τ (mm) | Prediction coverage | GT128 reference coverage | GT regions |
|---|---:|---:|---:|---:|
| 20 mm | 1 | 40.00% | 82.00% | 50 |
| 20 mm | 2 | 60.00% | 98.00% | 50 |
| 20 mm | 5 | 100.00% | 100.00% | 50 |
| 80 mm | 1 | 1.79% | 3.84% | 781 |
| 80 mm | 2 | 5.76% | 14.21% | 781 |
| 80 mm | 5 | 30.09% | 55.31% | 781 |

Pred→GT nhỏ đồng thời GT→pred lớn và region coverage thấp cho thấy coverage không đều/thiếu; không suy diễn từ aligned CD. Robust quantile cũng phản ánh tụ điểm, không đủ một mình để kết luận mất cả một mặt. Với128 outputs, ngưỡng2 mm rất chặt cho cube80 mm: mốc GT128 cũng thấp; đọc thêmτ5 mm và Z bins. Check: directional mean khớp CD cũ trong0,001 mm; coverage bất biến khi duplicate toàn bộ prediction.

Hình `coverage.png`; [3D coverage viewer](http://localhost:17070/ptv3-bf16/coverage.html), sliderτ1/2/5 mm; `coverage_metrics.json` chứa toàn bộ số liệu/định nghĩa. Code `OATFlow/ptv3_cloud/coverage.py`, chạy `.venv-concerto/bin/python -m OATFlow.ptv3_cloud.coverage`.
