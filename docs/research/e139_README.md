# E139 — Giữ tâm target, chỉ đổi shape XYZ

> **Code giữ lại:** [code/README.md](/workspace/robotics_assets/experiments/e139_target_shape/code/README.md). test: target-only shape counterfactual. [Danh mục/bài học](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




**Kết luận:** checkpoint SAM3 full-SE(3) theo translation tốt ở E138, nhưng gần như không theo biến đổi hình dạng tại tâm cố định. Test này chưa cung cấp bằng chứng model tái tạo metric shape từ XYZ riêng của target; hành vi phù hợp với semantic selection + vị trí XYZ + shape prior từ ảnh/training. Đây là diễn giải về hành vi trên counterfactual OOD, không chứng minh model hoàn toàn không dùng geometry.

## Data và protocol

Không train thêm, nguyên checkpoint E137 bước 18.000; baseline E135 checkpoint tốt nhất bước 5.750, train budget 6k. 128 scene IID × 4 target × 37 operations = **18.944 can thiệp/model**: 18.432 shape changes và 512 no-op; mỗi can thiệp query cả 4 task, 75.776 outputs/model. Scene vẫn có robot, ghế, thảm, bàn, sàn. Không lọc case theo prediction hay physics.

Giữ nguyên SAM3 features và feature–point identity, RGB, point order, task và mọi XYZ ngoài target. Mask simulator và side length chỉ tạo can thiệp, không vào forward. Input model vẫn là full-scene XYZ + per-point features + task ID; không GT center/size/mask.

- Single-axis scale: một trục base XYZ nhân 0,8/1,2 hoặc 0,6/1,4; hai trục khác giữ nguyên.
- Mixed-axis scale: lần lượt mọi cặp trục, nhân một trục 1,2 hoặc 1,4, trục kia nghịch đảo, trục còn lại 1. Đây là affine bảo toàn thể tích trước sampling.
- Bend: 3 cặp cyclic (output X←input Y, Y←Z, Z←X), hai dấu, hệ số a=0,05/0,10. Với q quanh centroid, q'_out=q_out+a·side·(q_in/(side/2))². Trừ mean displacement để tâm không đổi; a là hệ số field, không phải độ dịch tối đa bảo đảm.

Input target subset và GT target là hai tập sample khác nhau; mỗi tập được giữ **empirical centroid của chính nó**. Vì vậy affine shape tương ứng nhau sau centroid alignment; với bend, field dùng q quanh centroid riêng, không khẳng định hai sampling nằm trên cùng một continuous deformed surface. Reference prediction cũng được biến đổi quanh centroid riêng để đo transport-CD. Không render lại RGB/depth, không làm mới occlusion; visible surface points bị transport, có thể giao vật khác. Không coi đây là scene vật lý mới.

## Metrics

Centroid-aligned Chamfer (ACD) là symmetric mean nearest Euclidean distance, không bình phương, đơn vị mm. Chỉ trừ centroid từng cloud; không căn rotation hay scale. Extent mỗi trục E=max−min, đo trên visible sampled cloud, không phải amodal cube box.

So sánh ACD(pred_after,GT_after) với ACD(pred_before,GT_after): prediction đứng yên là control trực tiếp. Improvement = stationary ACD − after ACD; dương là tốt. Extent-response gain pooled = Σ(ΔE_pred·ΔE_GT)/Σ||ΔE_GT||²; lý tưởng 1, đứng yên 0. Chỉ tính gain khi ||ΔE_GT||≥0,5 mm; bend có thể đổi shape mà ít đổi bounding box, nên không dùng extent gain làm tiêu chí duy nhất.

## Kết quả SAM3-SE(3)

| Can thiệp | GT shape change ACD (mm) | ACD sau (mm) | ACD nếu giữ prediction cũ (mm) | Extent gain | Centroid drift target (mm) |
|---|---:|---:|---:|---:|---:|
| Single-axis scale | 2.313 | 6.8768 | 6.8763 | 0.00058 | 0.731 |
| Mixed-axis scale | 3.065 | 7.0100 | 7.0104 | 0.00155 | 0.945 |
| Quadratic bend | 0.828 | 6.8366 | 6.8358 | 0.01010 | 0.324 |

Các CI95% bootstrap theo scene cho paired ACD improvement (lặp can thiệp trong scene không được coi độc lập):

- Single-axis scale: -0.00044 mm, CI [-0.00330, +0.00254].
- Mixed-axis scale: +0.00033 mm, CI [-0.00228, +0.00319].
- Quadratic bend: -0.00078 mm, CI [-0.00377, +0.00217].

Tất cả CI chứa 0; chưa có cải thiện shape có ý nghĩa trong phép thử này. Prediction vẫn có thay đổi nhỏ (ACD before–after khoảng 0,56–0,73 mm), nhưng không đi theo thay đổi kích thước yêu cầu. Reference-transport CD cũng xấu hơn giữ nguyên prediction cũ ở cả ba nhóm; không được diễn giải mọi chuyển động point là geometry tracking.

### Kích thước từng trục — chỉ kéo +40% một trục

512 target/axis, trung bình mm; cột chỉ đo trục được can thiệp. Baseline widths giống nhau vì cùng 512 target.

| Trục | GT trước→sau (mm) | Prediction trước→sau (mm) | ΔGT (mm) | Δprediction (mm) |
|---|---:|---:|---:|---:|
| X | 41.77→58.48 | 41.28→41.26 | +16.71 | -0.025 |
| Y | 39.99→55.98 | 41.53→41.56 | +16.00 | +0.023 |
| Z | 40.69→56.96 | 41.07→41.09 | +16.28 | +0.023 |

MAE extent so với GT mới, từng trục X/Y/Z (mm):

| Can thiệp | MAE X | MAE Y | MAE Z |
|---|---:|---:|---:|
| Single-axis scale | 9.319 | 9.254 | 8.829 |
| Mixed-axis scale | 10.467 | 10.301 | 10.026 |
| Quadratic bend | 7.364 | 7.411 | 6.762 |

Full per-case extents, signed changes, axis errors, ACD và centroid drift nằm trong NPZ; summary có mild/strong, từng trục/dấu. Extent gain single-axis/mixed của model cũ lần lượt 0.00060/0.00056, cũng gần 0. Không quy chênh lệch giữa hai model chỉ cho augmentation vì train budgets khác nhau.

## Kiểm tra và giới hạn

Non-target XYZ bit-exact; centroid input target lệch tối đa 0.000089 mm, GT 0.000119 mm do FP32. 512 no-op có centroid/shape response bằng 0; checkpoint SHA256 giữ nguyên. Ba unit tests và kiểm tra độc lập 37 operations/model bằng NumPy float64 + SciPy xác nhận metric/transform trong `verification.json`.

Control precision: FP32 thuần, autocast/TF32 tắt, 16 scene đầu × 4 target × 18 strong changes = 1152 can thiệp. Extent gains single-axis/mixed/bend: 0.00013/0.00030/-0.00020; ACD improvements +0.00005/+0.00029/-0.00004 mm. Kết luận không chỉ do BF16 che mất đáp ứng shape.

Trong FP32, ACD giữa shape prediction trước/sau chỉ 0.0090/0.0118/0.0029 mm. Vì vậy phần lớn shape jitter 0,56–0,73 mm ở lượt BF16 có thể đến từ độ chính xác inference, không phải đáp ứng geometry có hướng.

Shape perturbations là OOD, đặc biệt cube→cuboid/bent surface; RGB/features còn mô tả shape cũ, point input thưa (4.096 toàn scene), GT chỉ visible surface 128 points, bend nhỏ hơn sai số baseline. Vì vậy không kết luận model không có thông tin geometry: kết quả chỉ cho thấy **chưa có geometry-following response với XYZ counterfactual khi giữ feature cố định**. Kết quả E138 hỗ trợ target-specific position grounding; E139 giới hạn phát biểu đó, chưa mở rộng thành target-shape reconstruction từ XYZ.

[Viewer centroid-aligned trước–sau](http://localhost:17070/target-shape/viewer.html) · `shape_response.png` · Artifacts `/workspace/robotics_assets/experiments/e139_target_shape`. Code `OATFlow/se3_cloud/local_shape.py`; không sửa architecture hay weights.
