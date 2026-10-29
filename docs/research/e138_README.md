# E138 — Chỉ dịch XYZ của target, giữ feature–point identity và distractor

> **Code giữ lại:** [code/README.md](/workspace/robotics_assets/experiments/e138_target_motion/code/README.md). test: target-only translation counterfactual. [Danh mục/bài học](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




## Protocol

Không train thêm; dùng nguyên checkpoint SAM3-SE(3) E137 và baseline SAM3 gốc E135. Trên 128 scene IID, lần lượt chọn từng cube làm object can thiệp; dịch 20/50/100 mm theo ±X/±Y/±Z và một hướng 3D ngẫu nhiên cố định. **Chỉ XYZ rows thuộc object đó và GT target được dịch cùng δ**; feature vector, thứ tự point, RGB, task ID và mọi XYZ distractor giữ nguyên. Mỗi can thiệp hỏi cả 4 task để đo target response và độ drift của ba output còn lại.

Mask simulator chỉ xác định những rows cần can thiệp/chấm điểm; forward vẫn chỉ nhận XYZ, sampled SAM3 features và task ID. Đây là **OOD feature transport**: vị trí pixel/ngữ cảnh trong SAM3 features còn ở ảnh gốc; không render lại, không cập nhật occlusion hay lấp vùng trống cũ. “Giữ correspondence” nghĩa là giữ identity vector–point, không khẳng định projection vẫn phù hợp ảnh sau khi dịch riêng object.

**11,264 can thiệp mỗi model**, gồm 512 no-op và 10.752 lần dịch khác 0. Primary có **9,350 lần dịch** sau kiểm tra cube OBB/table và bề mặt quan sát, clearance 3 mm; 1,402 lần còn lại vẫn được chấm riêng, không bỏ theo prediction. Bộ lọc không chứng nhận physics/occlusion đầy đủ. Mỗi model có 45.056 task queries, lặp lại trên cùng 128 scene; CI bootstrap theo scene, không coi mọi lần dịch là scene độc lập.

Scene vẫn chứa robot, ghế, thảm, bàn và sàn như dữ liệu E137; toàn bộ XYZ của chúng giữ nguyên trong từng can thiệp.

## Metrics chính

Gọi Δc là dịch chuyển centroid prediction trước/sau, δ là dịch chuyển thực. **Gain = dot(Δc,δ)/||δ||²**, lý tưởng 1, đứng yên 0. Tracking error = ||Δc−δ||. Transport CD = CD(pred_after,pred_before+δ). Accuracy nearest-GT chỉ là metric phụ vì vẫn có thể cao dù prediction không đi theo target.

| Model/frame | Gain↑ (ideal 1) | Tracking error(mm)↓ | Drift trung bình 3 output khác(mm)↓ | Transport CD(mm)↓ | CD tới GT sau(mm)↓ |
|---|---:|---:|---:|---:|---:|
| SAM3 gốc | 0.026 | 51.89 | 0.15 | 36.54 | 38.33 |
| SAM3 train full SE3 | 0.898 | 5.54 | 0.16 | 3.23 | 7.07 |
| Cùng model trong frame SE3 mới | 0.895 | 5.71 | 0.20 | 3.57 | 7.28 |

Gain SE3 **0.898**, CI95% bootstrap scene **[0.896,0.899]**. Model đi theo khoảng 90% displacement, còn under-tracking khoảng 10%; các distractor gần như đứng yên. Không diễn giải gain thành tỷ lệ đóng góp chính xác của từng nhánh mạng.

### Theo độ dịch — primary-valid

| Dịch target | Gain SE3 | Tracking error SE3(mm) | Error SAM3 cũ(mm) | Drift distractor SE3(mm) |
|---|---:|---:|---:|---:|
| 20mm (3493 cases) | 0.899 | 2.07 | 19.47 | 0.13 |
| 50mm (3154 cases) | 0.898 | 5.16 | 48.71 | 0.16 |
| 100mm (2703 cases) | 0.896 | 10.47 | 97.50 | 0.21 |

Operational tracking-pass đặt trước khi chạy: gain∈[0,8;1,2], error≤max(5mm,20% displacement), max centroid drift của ba output khác≤5mm. Đây là tolerance thực nghiệm, không phải “prediction hoàn hảo”.

Pass rate: **99.25%** cho SE3, **0.00%** cho baseline, **97.78%** trong frame SE3 khác. No-op 512 cases có response/drift/CD-transport bằng 0 cho cả ba lượt, kiểm tra pipeline không tạo motion giả.

## Kiểm tra độ nhạy và giới hạn

Tính trên **mọi 10.752 lần dịch**, kể cả trường hợp chạm bộ lọc hình học, SE3 gain 0.899, tracking error 5.83mm; không chỉ có các case đã lọc cho kết quả tốt. Tất cả hướng và màu, point-count strata cùng rejected-case stats ở `summary.json`.

Primary SE3 p90 tracking error **10.87mm**; p90 max drift của ba distractor **0.40mm**, worst-case distractor drift **8.23mm**. Nếu giữ prediction đứng yên, CD tới GT mới trung bình 38.38mm thay vì 7.07mm đo được.

Khi target chỉ có 1–7 input points (186 can thiệp), gain giảm còn 0.859; nhóm ≥32 points có gain 0.901. Đây là phân tầng mô tả, không phải can thiệp độc lập vào mật độ point.

**Diễn giải:** target di chuyển gần cùng vector trong khi ba query khác ổn định. Kết quả này không phù hợp với giải thích chỉ áp dụng một global rigid frame chung lên mọi prediction; nó bổ sung bằng chứng model SE3 dùng semantic feature gắn với metric position riêng của target. Chưa chứng minh model học mọi chi tiết geometry: test chỉ translation, không local rotation/scale/deformation; RGB/features cố ý stale, data synthetic và chỉ một checkpoint. Có khoảng 10% shrinkage và một số case không pass. Baseline và SE3 có budget train khác nhau 6k/18k, nên không quy toàn bộ chênh lệch riêng cho augmentation.

**Fidelity:** runtime kiểm tra bit-exact mọi XYZ không thuộc object can thiệp; feature không bị permute/recompute; sai số cộng translation dưới 0,0001 mm. Mask và δ không vào forward. Checkpoint không thay đổi. Xem `config.json`, `verification` trong `summary.json`, case records và arrays per-case.

Artifacts: `/workspace/robotics_assets/experiments/e138_target_motion`. [Viewer target/distractor trước–sau](http://localhost:17070/target-motion/viewer.html). `motion_response.png` tổng hợp toàn bộ primary cases; example 100 mm chọn gần median tracking error, không chọn best-case. Code `OATFlow/se3_cloud/local_motion.py`.

Kiểm tra độc lập: 3 unit tests pass; tính lại centroid/gain/CD bằng NumPy float64 và SciPy trên 21 case phân tầng mỗi lượt (63 tổng), sai khác metric khoảng cách tối đa dưới 0,00003 mm. SHA256 cả hai checkpoint trùng trước/sau. Chi tiết `verification.json`; chạy lại bằng `python -m OATFlow.se3_cloud.verify_local_motion`.
