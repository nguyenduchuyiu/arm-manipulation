# E136 — Frozen Concerto → task-conditioned target point cloud

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Concerto kém SAM3 và rất nhạy outlier/depth trong cấu hình đã thử. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




## Data và pretrained encoder

Dùng lại **1.792 scene / 7.168 truy vấn E135**, cùng RGB/depth/camera, XYZ, GT target, split và bảy corruption đã lưu. Train 1.024 scene, val 128; mỗi test 128 scene × 4 task. Scene vẫn có robot, ghế, thảm, bàn, sàn và bốn cube random XYZ/size/rotation. Không tạo lại dữ liệu hay đổi test sau khi train.

Đã tải [code chính thức](https://github.com/Pointcept/Concerto) tại commit `5b41117160cdd92c3538be1911277b17e7ef5871` và [weights chính thức](https://huggingface.co/Pointcept/Concerto) `concerto_base_origin.pth`, HF revision `efc130718a564eba0ce4899318f3c20b19d9c752`. Đây là Base gốc NeurIPS 2025, **108,461,376 tham số**, không phải checkpoint mặc định mới hơn. File weights: `/workspace/robotics_assets/pretrained/concerto/concerto_base_origin.pth`.

Input encoder: **4.096 điểm full-scene XYZ + RGB tại pixel tương ứng + normals PCA 16-neighbor từ chính XYZ**, hướng về camera. Normals không lấy từ simulator/GT và được tính lại khi XYZ nhiễu. Voxel 5 mm (mặc định repo 20 mm; chọn trước test để giữ cube nhỏ). Dùng CenterShift toàn scene theo official preprocessing, giữ XYZ gốc ở decoder; không centering theo target.

## Kiến trúc và training

**Concerto frozen → features phân cấp → map ngược pooling/voxel → 1.088 chiều/điểm → [XYZ, feature] → point MLP + Transformer → task embedding → decoder 128 điểm XYZ**. Ghép ba tầng sâu theo demo chính thức. Encoder eval/FP32, cache FP16; không nhận task, mask hay instruction.

Train downstream **987,139 tham số**: MLP→128, scene Transformer 2 lớp, decoder 2 lớp, 4 heads, FFN384, task embedding 4×128. Giữ loss Chamfer unsquared, AdamW, batch 8 scene/32 task, LR3e−4→3e−5, seed135 và 6.000 bước như SAM3; không noise augmentation. Chọn step 5750 theo validation; train 3.5 phút, peak allocated GPU 10.56 GiB. Chỉ input LayerNorm/MLP đổi chiều 256→1088 nên số tham số head khác.

## Kết quả

CD, centroid error và center-aligned CD đều tính bằng mm; accuracy = object có Chamfer tới prediction nhỏ nhất, chỉ dùng GT lúc chấm.

| Test | CD Concerto↓ | CD SAM3↓ | Centroid Concerto↓ | Aligned CD Concerto↓ | Accuracy Concerto↑ | Accuracy SAM3↑ |
|---|---:|---:|---:|---:|---:|---:|
| IID | 9.37 | 6.80 | 18.71 | 5.98 | 99.8% | 100.0% |
| XYZ ngoài train | 42.48 | 23.67 | 60.83 | 6.74 | 89.1% | 98.8% |
| Rotation mới | 12.56 | 10.12 | 22.36 | 5.88 | 99.2% | 99.4% |
| Size mới | 18.28 | 11.29 | 29.31 | 6.66 | 93.8% | 97.9% |
| Màu–vị trí mới | 13.55 | 9.20 | 25.38 | 6.06 | 98.2% | 100.0% |

**Nhiễu paired trên cùng 128 scene IID, GT đích sạch giữ nguyên:**

| Corruption | CD Concerto | CD SAM3 | Accuracy Concerto | Accuracy SAM3 |
|---|---:|---:|---:|---:|
| RGB σ10/255 | 10.58 | 7.71 | 100.0% | 99.8% |
| RGB σ25/255 | 13.83 | 9.18 | 98.6% | 100.0% |
| Depth σ2 mm | 14.64 | 6.79 | 99.4% | 100.0% |
| Depth σ5 mm | 26.87 | 6.81 | 93.4% | 100.0% |
| Mất 20% điểm | 10.43 | 7.08 | 100.0% | 100.0% |
| 5% outlier | 49.69 | 6.91 | 71.9% | 99.8% |
| Nhiễu kết hợp | 56.71 | 8.47 | 68.9% | 99.8% |

Combined = RGB σ15/255 + depth σ5 mm + mất 20% điểm + 2% outlier. Các corruption XYZ/RGB giống hệt SAM3; Concerto recompute normals và point features từ input đã nhiễu.

**Diagnostic:** `constant_task0`: accuracy 25.0%, CD 112.95 mm; `zero_features`: accuracy 35.0%, CD 163.70 mm; `zero_xyz`: accuracy 82.4%, CD 36.53 mm; `shuffle_features`: accuracy 66.6%, CD 55.16 mm; zero XYZ chỉ zero nhánh XYZ của head, vẫn giữ Concerto features đã encode geometry; không phải baseline bỏ toàn bộ 3D. Shuffle/zero feature là diagnostic inference, không phải ablation train lại.

IID: CD 9.37 mm, accuracy 99.8%; đổi task trên cùng scene đúng cả 4 ở 99.2% scene. Nhiễu kết hợp đổi CD +47.34 mm, accuracy -30.9 điểm %. XYZ ngoài train: CD 42.48 mm.

**Kết luận:** trong cấu hình/budget này, Concerto kém SAM3 về sai số hình học và robust với nhiễu depth/outlier. Phá correspondence feature–point làm accuracy Concerto giảm xuống66,6%, trong khi SAM3 vẫn100%; đây là diagnostic cho mức phụ thuộc vào correspondence, không chứng minh tổng quát hóa đầy đủ.

Quan sát preprocessing: với5% outlier, mốc Z của CenterShift(min-Z toàn scene) thay đổi median -1.44m. Normals/voxel groups cũng được tính lại từ input nhiễu. Chưa có causal ablation để tách đóng góp từng khâu; không thay preprocessing hay retrain sau khi xem test.

Inference online XYZ+point RGB+task đã chạy không đọc GT; max coordinate delta so với cache/eval BF16 3.906mm. Metrics đã đối chiếu CPU trên20 truy vấn của5 split.

Giới hạn: một seed, nền/camera cố định, nhãn chỉ mặt nhìn thấy; chưa thử scene/camera thực. So sánh cùng data/loss/schedule nhưng modality/features khác (Concerto có point RGB+normals; SAM3 có dense image features), encoder 108M vs 454M và chiều input head khác; không phải so sánh chỉ thay một yếu tố.

Đã kiểm tra frozen-parameter fingerprints và SHA256 checkpoint không đổi; source config/weights/version được pin. Artifacts: `/workspace/robotics_assets/experiments/e136_concerto`. Checkpoint downstream `train/best.pt`, metrics per-query `evaluation/*.npz`, config frozen `features/config.json`. [Viewer GT/Concerto/SAM3](http://localhost:17070/concerto/viewer.html). Code: `OATFlow/concerto_cloud/`.
