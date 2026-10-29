# E147 — Official PTv3 U-Net, feature tại từng point

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: PTv3 FP32 initial1k chưa overfit chính xác; giữ implementation PTv3 dùng được ở E149. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Đã bỏ custom `OATFlow/local_cloud` và chuyển sang official PTv3 Mode1, **cls_mode=False**. Full-cloud1.000 updates FP32 hoàn thành, best checkpoint **1000**. Dense feature export **[2,74.356,64]**,128 target output points. Đây là initial1k-update overfit diagnostic, không cùng budget18k của E142/E144.

## Source và architecture

[Official PTv3 repository](https://github.com/Pointcept/PointTransformerV3), pinned commit `3229e9b7de1770c8ad17c316f8e349982de509f8`; model.py/serialization không sửa. Dùng5 encoder stages channels32/64/128/256/512, depths2/2/2/6/2;4 U-Net decoder stages channels64/64/128/256, depths2/2/2/2. SerializedPooling/SerializedUnpooling và skip connections official; không dùng encoder-only/classification output.

Input259D=[XYZ×2,LayerNorm(SAM3 frozen256D)], grid1mm. Unique finest voxels nhận mean input features, official U-Net trả feature64D trên finest voxels; exact voxel inverse map về mỗi input row. Với hai frame này cả8k/full không có voxel merges:74.356 unique voxels và74.356 restored original points mỗi scene. Native voxel/input ordering được kiểm tra qua output grid_coord/coord của backbone; duplicate-input inverse được test riêng. Không GT center/size/mask/tokens vào model.

PTv3 backbone **47,178,272 params**, tổng trainable **47,684,771**; frozen SAM3 encoder+neck454.038.784 từ cache cũ. Memory projection64→128 và task embedding/128 learned queries/2-layer width128 decoder/XYZ MLP nối với backbone. Custom kNN architecture đã xóa khỏi repository; cancelled E146 chỉ giữ logs/data/partial checkpoint history.

## Precision và memory configuration

Dùng FP32/TF32 off, FlashAttention disabled; official non-Flash attention implementation. Patch128 full-cloud FP32 đã OOM trên GPU24GB, lưu `benchmark_patch128.json`. Patch64 dùng cùng official class và U-Net, giảm memory attention; không tự viết lại backbone hay kernel. Drop path và order shuffle tắt cho fixed2-example diagnostic. Dependency env `.venv-concerto` có spconv/torch_scatter; không cài/đổi driver.

## Benchmark cùng FP32

Training step gồm forward+Chamfer+backward+clip+AdamW, batch2;1 warmup rồi2 timed iterations. Static voxel preprocessing và SAM3 extraction không tính vào steady step; thời gian preprocess được ghi riêng.

| Input | Old global step(s) | PTv3 step(s) | Old/PTv3 speed ratio | PTv3 train peak(GiB) | Dense output shape |
|---|---:|---:|---:|---:|---|
| 8,192 | 0.1358 | 0.2787 | 0.49× | 4.97 | [2, 8192, 64] |
| 74,356 | 9.4978 | 1.0596 | 8.96× | 20.23 | [2, 74356, 64] |

PTv3 base không nhanh hơn tiny baseline ở8k, nhưng nhanh hơn ởfull74k. Số parameters/depth khác lớn nên không coi đây là matched-capacity comparison. PTv3 vẫn có serialized patch attention, không phải chỉ128×N cross-attention. FlashAttention chưa dùng nên đây không phải peak-speed PTv3 configuration.

## Data và training

Dùng đúng hai red cube20/80mm, cùng body center/pose/task0/camera và clutter robot/ghế/thảm/bàn/sàn. RGB320×240, full visible-depth74.356 points/scene, target observed points150/2.624, GT target128 unchanged. Same cached SAM3 vision_features, no prompt; same16k input prefix/features đã kiểm tra bit-exact. Full cloud là toàn bộ surface nhìn thấy bởi một camera, không phải mesh/amodal360°.

Train from scratch seed135, fullbatch2,1.000 updates, một symmetric unsquared nearest Euclidean Chamfer, meters. Không augmentation, không auxiliary loss. AdamW LR3e-4→3e-5 cosine/warmup100, weight decay0,01, clip1; best mean training CD mỗi50 updates, không heldout set. Actual time **18.13 phút**, peak **20.24GiB**. Benchmark old models không phải pretrained PTv3 initializer; không load old custom architecture weights.

## Kết quả checkpoint tốt nhất

CD mm, aligned chỉ trừ centroid, không align rotation/scale.

| Inference N | Sample | Raw CD | Aligned CD | Centroid error | Pred extent X/Y/Z(mm) |
|---|---|---:|---:|---:|---|
| 8192 | 20mm | 13.8370 | 3.9714 | 19.6771 | 22.19/27.87/24.47 |
| 8192 | 80mm | 40.1900 | 18.9180 | 34.5635 | 24.84/29.31/28.34 |
| 74356 | 20mm | 1.9873 | 3.8738 | 4.7475 | 32.64/31.30/34.03 |
| 74356 | 80mm | 5.7348 | 13.4338 | 15.6361 | 69.60/71.81/66.64 |

GT visible extents small 17.05/18.24/20.30mm, large 78.02/76.65/79.34mm. Kết quả1k updates chưa phải overfit chính xác; không dùng accuracy chọn object để thay shape metrics.

## Correspondence-preserving cross controls

Reuse exact audited128 canonical surface locations/features E143; background dùng common physical surfaces với correct cached SAM3 feature rows. Chỉ xây dựng probe bằng GT, không cấp masks/centers/sizes cho model. Full-cloud input count giữ nguyên, matched baselines dưới cùng sampling. Nhưng target density large2624→128 là shift lớn, nên cần đọc baseline separation trước khi kết luận semantic-vs-geometry preference.

| F / XYZ | Pred extent X/Y/Z(mm) | Distance to F baseline(mm) | To XYZ baseline(mm) |
|---|---|---:|---:|
| 0/0 | 32.223/30.800/33.667 | 0.000 | 0.000 |
| 1/1 | 39.563/38.749/41.716 | 0.000 | 0.000 |
| 0/1 | 31.770/30.354/33.169 | 0.808 | 14.292 |
| 1/0 | 40.075/39.296/42.238 | 0.913 | 14.398 |

0=small,1=large; matched-baseline extent separation 13.485mm. Probe vẫn OOD appearance/geometry conflict và target-aware resampling, không phải normal inference/training augmentation.

Quan trọng: matched large baseline dưới128-target-point control đổi từ native 69.60/71.81/66.64 sang 39.56/38.75/41.72 mm. Đây là density shift lớn đối với local/sparse backbone; không tái sử dụng kết luận mạnh từ E143 như thể baselines vẫn giống native. Cross outputs gần F baseline là quan sát trong probe, chưa đủ chứng minh PTv3 không dùng geometry trên native full cloud.

## Verification và artifacts

Official source diff sạch; checkpoint hash; classifier mode false;4 U-Net up stages; per-point features[2,74.356,64] lưu `per_point_features.npy` đúng thứ tự `e146_local/data.npz.points`; output original voxel order được hook-verify trước inverse. CPU float64/SciPy xác nhận raw/aligned CD trong0,001mm. Synthetic duplicate-voxel/batch inverse test pass. Không tự viết lại PTv3 architecture.

Một seed,2 samples và budget1k chưa đủ đánh giá generalization hoặc quy nguyên nhân geometry failure. Hình dạng thay đổi theo input không tự chứng minh dùng metric XYZ thay vì appearance shortcut; point-density và coverage của decoder vẫn cần kiểm tra.

Code `OATFlow/ptv3_cloud`: `smoke`, `benchmark`, `train`, `finish`; env `.venv-concerto/bin/python`. Exact configs `train/config.json`, source git commit trong `smoke.json`, `benchmark.json`, `summary.json`. [Viewer](http://localhost:17070/ptv3/viewer.html), `learning_curve.png`. Artifacts `/workspace/robotics_assets/experiments/e147_ptv3`.
