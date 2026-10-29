# E137 — SAM3 + full SO(3) và translation

> **Code giữ lại:** [code/README.md](/workspace/robotics_assets/experiments/e137_sam3_se3/code/README.md). core: full SO(3) + translation and feature/XYZ controls. [Danh mục/bài học](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Giữ nguyên kiến trúc 878.979 tham số, frozen SAM3 features, data/splits E135 và một Chamfer loss. Mỗi scene nhận R~Haar SO(3), t~Uniform([−0,5;0,5]³)m; áp dụng chung cho toàn scene và bốn GT target. RGB/F giữ nguyên. SE(3) trước scale isotropic XYZ×2, không mean/std từng trục. Head không nhận extrinsics; nếu chiếu ảnh, extrinsics được đổi thành S@T nên UV không đổi.

Train từ đầu **18.000 bước**, checkpoint chọn bằng fixed-SE(3) val tại step **18000**; không dùng test/swap chọn checkpoint. Lưu thêm bước 6000. Nhiều bước hơn baseline cũ 6000, kiến trúc không đổi.

| Input đúng | Chamfer tới GT(mm)↓ | Chọn đúng object↑ |
|---|---:|---:|
| Cũ, frame gốc | 6.80 | 100.0% |
| Mới, frame gốc | 6.58 | 100.0% |
| Cũ, full SE3 test | 602.51 | 26.5% |
| Mới, full SE3 test | 6.83 | 100.0% |

Equivariance CD `CD(f(F,SP), S f(F,P))`: cũ **604.00mm**, mới **4.84mm**, trên 128 scene × 4 task × 4 transform mới.

## XYZ A + feature B

512 truy vấn, cùng pairing B=(A−1)mod128 như trước. GT B chỉ là reference chấm điểm.

| Model | Accuracy theo A | CD tới A(mm) | Accuracy theo B | CD tới B(mm) |
|---|---:|---:|---:|---:|
| Cũ | 33.2% | 123.50 | 100.0% | 8.52 |
| Mới, frame gốc | 23.8% | 187.73 | 24.0% | 188.50 |
| Mới, frame SE3 | 25.4% | 205.32 | 24.4% | 205.69 |

Trong frame SE3, GT A và GT B cùng được đổi bằng S_A. Đây là thí nghiệm đo shortcut, không thêm loss ép counterfactual.

| Control trên model mới | Chamfer(mm) | Accuracy |
|---|---:|---:|
| zero_xyz | 320.13 | 25.6% |
| shuffle_features | 186.31 | 23.2% |
| zero_features | 882.82 | 23.4% |

SE(3) khuyến khích model dùng XYZ để xác định frame, nhưng riêng điều đó chưa bảo đảm model bám điểm của đúng object. Nền cố định có thể cho phép học global frame rồi biến đổi dự đoán từ feature ảnh. Đọc kết quả swap cùng chất lượng input đúng và distance tới input surface trong báo cáo chi tiết.

**Kết quả trên phép thử này:** input khớp vẫn đạt 100.0% với CD 6.58 mm, nhưng khi ghép sai scene, accuracy theo cả A và B về khoảng25%, CD khoảng188–189 mm. So với run cũ giữ100% accuracy theoB, shortcut đi theo nguồn ảnh giảm rõ. Đây là fail khi chấm bằng GT; decoder vẫn sinh128 điểm, chưa có cơ chế abstain/reject và chưa có bảo đảm equivariance tuyệt đối.

Đã đối chiếu metrics với CPU, xác nhận state-dict/architecture không đổi và projection consistency; sai số UV do đổi frame tối đa 1.71e-13px.

[Chi tiết đầy đủ: test XYZ/rotation/size/combinations, sensor noise, surface distance, mốc6000](/workspace/robotics_assets/experiments/e137_sam3_se3/DETAILS.md). [Viewer swap cũ/mới](http://localhost:17070/se3/viewer.html). Artifacts `/workspace/robotics_assets/experiments/e137_sam3_se3`; checkpoint `train/best.pt`; code `OATFlow/se3_cloud/`.
