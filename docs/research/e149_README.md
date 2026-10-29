# E149 — PTv3 với exact EMD thay Chamfer

> **Code giữ lại:** [code/README.md](/workspace/robotics_assets/experiments/e149_emd/code/README.md). core: PTv3 + SAM3 + exact EMD, successful GT128 overfit. [Danh mục/bài học](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành18.000 updates, best17200, 2.93 giờ.

Chỉ đổi training loss từ Chamfer E148 sang **exact Earth Mover's Distance** với128 prediction và128 GT điểm đồng trọng số:

`EMD(P,G) = min_permutation mean_i ||P_i − G_permutation(i)||₂`.

Cost Euclidean không bình phương, đơn vị m; báo cáo mm. Tìm assignment1–1 tối ưu bằng SciPy `linear_sum_assignment` trên128×128, sau đó tính norm từng cặp bằng PyTorch và backprop qua tọa độ. Assignment rời rạc không lấy gradient; lấy gradient của nhánh tối ưu hiện tại (a.e., tie có thể không duy nhất). Không Sinkhorn/entropic regularization, không auxiliary loss. Mỗi GT point nhận đúng một prediction, tránh nhiều prediction cùng được thưởng bởi một nearest GT như Chamfer. Không bảo đảm phủ cả surface camera vì GT supervision vẫn chỉ128 samples cũ.

- Data/camera/features giữ nguyên E148: hai frame cube đỏ20/80 mm cùng tâm/pose với robot/ghế/thảm/toàn cảnh; cam_front320×240; full74.356 input points/scene, target hiện diện150/2624. Dataset/cache dùng chung `e146_local/data.npz`, `features.npy`, không tạo bản sao cache. SAM3 frozen vision_features256D, cùng pretrained weights/point-feature correspondence. GT label/size/center không vào model.
- Kiến trúc giữ nguyên official PTv3 Mode1 U-Net `cls_mode=False`, grid1 mm, patch64, non-Flash,64D feature từng input point và128-query decoder;47.684.771 trainable params. Official commit `3229e9b7de1770c8ad17c316f8e349982de509f8`, source không sửa.
- Train từ đầu seed135, **18.000 updates**, batch2, không augmentation, không warm start E148. AdamW LR3e-4→3e-5 cosine/warmup100, weight_decay.01, clip1. Same initialization/config as E148 ngoại trừ loss và tiêu chí chọn best theo mean training EMD mỗi100 bước.
- Precision giữ nguyên E148: BF16 attention/MLP, FP32 sparse conv/XYZ head/master weights/optimizer/loss; TF32 off, không GradScaler.
- Lưu best và một resume checkpoint mỗi1000 updates; không lưu checkpoint history hoặc copy pretrained/data caches. Chọn best theo EMD; log CD chỉ để đối chiếu, không tham gia loss.
- Verification trước train: exact optimum so với exhaustive permutation5 points; float64 finite-difference gradient check; permutation invariance; known translation; zero-distance finite gradients; collapsed predictions vẫn phải transport toàn bộ GT mass. Full-cloud6-step forward/backward benchmark, median bỏ warmup1.
- Sau train: cùng checkpoint best evaluate BF16/FP32; ba metric trên prediction gốc, không centroid alignment: directional unsquared CD vớiGT128 và full visibleGT150/2624, Q95−Q5 XYZ và Z max−min, equal-voxel surface coverage4 mm tạiτ1/2/5 mm cùng5 Z bins. Dùng cùng GT128 reference để kiểm soát point budget, cùng code `coverage.py` như E148. Coverage sensitivity voxel2/8 mm. Chỉ surface nhìn thấy, không mặt bị che.
- Đây là overfit2 mẫu, không held-out set, một seed; không kết luận generalization. So sánh chính E148 Chamfer18k và E149 EMD18k, cùng model/data/seed/schedule; best của mỗi objective được chọn theo objective tương ứng.

Artifacts `/workspace/robotics_assets/experiments/e149_emd`; `benchmark.json`, `train/config.json`, `train/progress.json`, `train/metrics.jsonl`, `summary.json`, `coverage_metrics.json`.

Code `OATFlow/ptv3_cloud/emd.py`, `train_emd.py`, `compare_emd.py`; environment `.venv-concerto` sẵn có, không thêm dependencies. Chạy từ `/workspace/arm-manipulation`:

```
.venv-concerto/bin/python -m OATFlow.ptv3_cloud.emd
.venv-concerto/bin/python -m OATFlow.ptv3_cloud.train_emd
```

Nguồn thuật toán assignment: [SciPy linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html), tìm minimum-cost complete matching; implementation modified Jonker–Volgenant.

## Benchmark và launch

Exact EMD solver/gradient checks pass; full-cloud6-step benchmark median0,570 s/update (loss forward khoảng8 ms/batch), peak15,52 GiB, dự kiến18k ~2,85 giờ trước eval/checkpoint overhead. E148 Chamfer benchmark0,578 s/update; chênh nhỏ trên benchmark ngắn, không kết luận EMD nhanh hơn. Đã khởi chạy tmux `e149`, log `train.log`. Đã đối chiếu24 config fields và model source hash với E148; code metric sau refactor cho kết quả bit-for-bit giống E148, comparison plot smoke-render pass; các artifact kiểm tra tạm đã xóa. Xem `verification.json`, `source_hashes.json`.

## Snapshot visualization best15800

Saved-prediction preview trong khi train: `preview_15800/preview.png`, [3D viewer](http://localhost:17070/ptv3-emd/preview_15800/preview.html). GT/Chamfer/EMD cùng trụcbase XYZ mm và tỷ lệ mỗi hàng, không alignment/resample/inference. Đánh dấu rõ đây là interim best15800, không phải kết quả18k. Snapshot SHA và ba metric trong `preview_15800/metrics.json`; code `visualize_emd.py`.

## Cửa sổ log

Training window đã chuyển nguyên trạng sang attached session `ssh_tmux`, window `1:e149-log` (Ctrl+B rồi1), không restart hoặc gián đoạn training. Window bật `remain-on-exit on` để giữ stdout/stderr sau khi hoàn thành; full log vẫn ở `train.log`. Window0 là phiên làm việc chính. Các lượt sau mở window riêng cùng session theo `AGENT.md`.

## Ba metric trên cloud gốc: directional CD, robust extent, coverage

Dùng đúng saved BF16 best17200, prediction không căn tâm/rotate/scale/resample. `GT128` là training target cũ; `visible_GT` là toàn bộ150/2624 target points nhìn thấy trong full-depth cloud, GT label1 chỉ dùng để đo. Directional CD là mean Euclidean unsquared mm.

| Cube | Pred→GT128 | GT128→pred | Pred→visibleGT | visibleGT→pred |
|---|---:|---:|---:|---:|
| 20 mm | 0.099 | 0.099 | 0.099 | 0.370 |
| 80 mm | 0.100 | 0.100 | 0.100 | 4.868 |

Robust extent Q95−Q5, NumPy linear quantile, cùng tọa độ base. Đây là extent theo phân bố điểm; tụ điểm cũng làm quantile co lại, nên không đồng nhất với kích thước vật lý.

| Cube | Cloud | Q95−Q5 XYZ (mm) | Z max−min (mm) |
|---|---|---|---:|
| 20 mm | prediction | 16.868/18.241/20.334 | 20.394 |
| 20 mm | GT128 | 16.809/18.236/20.299 | 20.299 |
| 20 mm | visible_GT | 16.823/18.236/20.299 | 20.299 |
| 80 mm | prediction | 70.326/70.877/61.358 | 79.389 |
| 80 mm | GT128 | 70.258/70.875/61.347 | 79.342 |
| 80 mm | visible_GT | 71.832/69.292/68.446 | 79.342 |

Coverage: GT occupied voxels4 mm, origin=visible_GT.min XYZ, mỗi voxel chọn GT sample gần mean trong ô nhất làm đại diện. Covered nếu đại diện cách ít nhất1 predicted point ≤τ. Mỗi vùng trọng số1, không dùng số predicted points trong vùng. Chỉ surface nhìn thấy, không đánh giá mặt bị che. GT128 tham chiếu đo cùng quy trình để nhận biết giới hạn128 output points; không phải cận trên tối ưu. Có sensitivity voxel2/8 mm trong JSON, và5 binsZ độ rộng bằng nhau (bin trống null).

| Cube | τ (mm) | Prediction coverage | GT128 reference coverage | GT regions |
|---|---:|---:|---:|---:|
| 20 mm | 1 | 82.00% | 82.00% | 50 |
| 20 mm | 2 | 98.00% | 98.00% | 50 |
| 20 mm | 5 | 100.00% | 100.00% | 50 |
| 80 mm | 1 | 3.84% | 3.84% | 781 |
| 80 mm | 2 | 14.21% | 14.21% | 781 |
| 80 mm | 5 | 55.06% | 55.31% | 781 |

Pred→GT nhỏ đồng thời GT→pred lớn và region coverage thấp cho thấy coverage không đều/thiếu; không suy diễn từ aligned CD. Robust quantile cũng phản ánh tụ điểm, không đủ một mình để kết luận mất cả một mặt. Với128 outputs, ngưỡng2 mm rất chặt cho cube80 mm: mốc GT128 cũng thấp; đọc thêmτ5 mm và Z bins. Check: directional mean khớp CD cũ trong0,001 mm; coverage bất biến khi duplicate toàn bộ prediction.

Hình `coverage.png`; [3D coverage viewer](http://localhost:17070/ptv3-emd/coverage.html), sliderτ1/2/5 mm; `coverage_metrics.json` chứa toàn bộ số liệu/định nghĩa. Code `OATFlow/ptv3_cloud/coverage.py`, chạy `.venv-concerto/bin/python -m OATFlow.ptv3_cloud.coverage`.

## So sánh với E148 Chamfer18k

Best theo objective riêng: Chamfer step16600; EMD step17200. Tổng thời gian EMD 2.93 giờ, peak allocated 15.52 GiB.

| Cube | Loss train | EMD (mm) | Pred→visibleGT (mm) | visibleGT→pred (mm) | Robust Z (mm) | Coverage2mm | Coverage5mm | Lowest Z-bin coverage5mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20 mm | Chamfer | 6.632 | 0.102 | 1.518 | 6.347 | 60.00% | 100.00% | 100.00% |
| 20 mm | EMD | 0.099 | 0.099 | 0.370 | 20.334 | 98.00% | 100.00% | 100.00% |
| 80 mm | Chamfer | 25.627 | 0.135 | 7.550 | 18.160 | 5.76% | 30.09% | 3.70% |
| 80 mm | EMD | 0.100 | 0.100 | 4.868 | 61.358 | 14.21% | 55.06% | 24.69% |

GT128 reference, thresholds và voxel/bin settings giống nhau; xem toàn bộ bảng ba metric phía trên và `comparison.json`. Quantile còn phụ thuộc sampling, coverage là vùng GT nhìn thấy với đại diện voxel, không phải tỷ lệ toàn bộ mesh. Đây chỉ là overfit2 mẫu, chưa kiểm chứng generalization.

[Comparison viewer](http://localhost:17070/ptv3-emd/comparison.html), `comparison.png`; code `compare_emd.py`.

## Completion audit

18k hoàn tất trong10.549,27 s (~2 giờ56 phút), best17200. CPU float64/SciPy xác nhận exact EMD, raw/aligned Chamfer cảBF16 vàFP32 trong0,001 mm, checkpoint SHA256 khớp. Cả hai cloud BF16 có128 nearest-GT indices riêng biệt, tối đa1 predicted point mỗi nearest GT (so với Chamfer64/58 cùng1 nearestGT). Đây là đối chiếu với GT128, không chứng minh full-surface completeness hoặc geometry grounding/generalization. File `completion_audit.json`. Comparison/coverage HTML và PNG trảHTTP200; training window `ssh_tmux:1:e149-log` kết thúc exit0, giữ log bằngremain-on-exit.

Viewer hiện tại là bản Plotly nguyên bản: `comparison.html` khoảng5 MB với4 khung3D, `coverage.html` có slider ngưỡng. Đã gỡ viewer nhẹ theo yêu cầu người dùng. Checkpoints/data/metric JSON giữ nguyên.
