# E145 — Test checkpoint tốt nhất với input dày hơn

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Dense inference là diagnostic cũ, giữ bảng/ảnh; không cần một implementation riêng. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Dừng train E144 theo yêu cầu ở last logged15.750/planned18.000 updates, dùng best checkpoint **step15000**. Không train/finetune thêm. Tăng input từ4k tới full74.356 points không cải thiện shape rõ rệt; large aligned CD vẫn quanh12,09mm.

## Data và inference

Hai E142/E144 cube20/80mm cùng body center/pose/task0, đúng RGB320×240/depth/camera/clutter. Giữ E14416.384-point prefix, thêm uniform permutation của unused depth points seed145 để có nested4.096/8.192/16.384/32.768/full74.356. Không crop, sampling theo mask, duplicates hay đổi GT128. Full depth chứa150/2.624 observations target small/large. Frozen SAM3 features được sample đúng pixel của added points; prefix16k bit-exact cache E144.

Exact architecture PointCloudGrounder878.979 trainable parameters và frozen SAM3 encoder+neck454.038.784, weights giữ nguyên. FP32 point head, TF32/autocast off, task0; output128 points. Dùng memory-efficient exact SDPA và tắt fused MHA eval fastpath để tránh lưu dense N×N matrix. Đây là đổi kernel, **không đổi global self-attention hay giảm O(N²) compute**. Verified8k prediction matches original backend. No action/flow representation.

## Kết quả

| N input | Target input nhỏ/lớn | Raw CD nhỏ/lớn(mm) | Aligned CD nhỏ/lớn(mm) | Head batch2(s) | Peak allocated GPU(GiB) |
|---:|---|---|---|---:|---:|
| 4,096 | 4/158 | 0.7833/2.1441 | 3.3681/12.1084 | 0.1307 | 0.341 |
| 8,192 | 17/294 | 0.7777/2.0373 | 3.3611/12.0914 | 0.0347 | 0.388 |
| 16,384 | 32/597 | 0.7914/2.0496 | 3.3564/12.0944 | 0.1456 | 0.482 |
| 32,768 | 69/1169 | 0.7974/2.0446 | 3.3555/12.0944 | 0.5512 | 0.670 |
| 74,356 | 150/2624 | 0.8001/2.0733 | 3.3551/12.0913 | 2.9741 | 1.148 |

Timing là1 measured forward/batch2 sau feature caching, GPU synchronized; không benchmark many repeats, không gồm SAM3/image/data loading. Peak là torch allocated memory; cache allocator/driver overhead không được tính. Added memory beyond resident input/model có riêng trong summary.

### Large-object extents X/Y/Z(mm)

| Input | Pred extent |
|---|---|
| 4,096 | 72.790/74.898/62.131 |
| 8,192 | 71.990/74.195/61.613 |
| 16,384 | 71.881/73.975/61.465 |
| 32,768 | 71.980/74.103/61.541 |
| 74,356 | 71.823/73.941/61.384 |

GT visible-cloud extent: 78.022/76.654/79.342 mm. Centroid-aligned CD chỉ trừ point mean, không rotation/scale alignment. Decoder output density/coverage vẫn có thể lệch; tăng input cardinality không tăng output128 hay sửa Chamfer objective.

## Diễn giải và architecture proposal

Dù target lớn có đủ2.624 observed points và target nhỏ đủ150, output vẫn gần như cũ và thiếu Z. Với checkpoint này, lỗi shape không được khắc phục chỉ bằng cấp thêm points; không kết luận kiến trúc không thể học geometry, hoặc XYZ không được dùng cho vị trí. N inference cao hơn N train cũng là cardinality extrapolation.

Model cũ đã có128 output queries cross-attend tới point memory, nhưng trước đó vẫn có2 layers global point self-attention. Thay phần encoder đó bằng kNN/voxel local geometry là một kiến trúc khác cần train lại; không được coi là cùng checkpoint sau khi bỏ layers. Local encoder +query cross-attention giảm phần attention từN² vềkN+128N, nhưng speedup toàn model không tự động bằng64× vì còn neighbor construction/MLP/FFN. Chưa train biến thể mới trong E145.

Verification: unchanged checkpoint SHA256, feature prefix error0,8k kernel max output difference 0.000000946m, CPU float64/SciPy aligned-CD max residual 0.00002129mm. `summary.json`, `verification.json`, `predictions.npz` lưu đủ inputsizes/outputs/protocol.

Code `OATFlow/dense_overfit/infer_dense.py`, `infer_report.py`; artifacts `/workspace/robotics_assets/experiments/e145_dense_infer`. [Viewer](http://localhost:17070/dense-infer/viewer.html), `results.png`. E144 stopped-run report vẫn ở `/workspace/arm-manipulation/docs/research/e144_README.md`.
