# E135 — Task-conditioned target point cloud

> **Code giữ lại:** [code/README.md](/workspace/robotics_assets/experiments/e135_grounding/code/README.md). core: SAM3 + custom point encoder, original data generation. [Danh mục/bài học](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




## Data

**1.792 scene / 7.168 truy vấn**: 1.024 scene train, 128 val, 128 cho mỗi test IID/position/rotation/size/combinations. Bốn task cùng scene không bị chia sang split khác nhau. Giữ nguyên camera `cam_front` vừa thử, RGB 320×240; scene luôn có **robot, ghế, thảm, bàn, sàn và bốn cube**.

XYZ, size và orientation của từng cube được lấy mẫu riêng, rồi reject giao nhau/xuyên bàn/quá ít pixel nhìn thấy. Scene tĩnh, cube có thể treo cao để random Z độc lập. Train world XYZ: X 0,36–0,70; Y −0,37…−0,13; Z 0,035–0,125 m; cạnh 24–50 mm; roll/pitch ±10°, yaw ±20°. Test dùng band XYZ ngoài train; rotation |roll/pitch|20–35°, |yaw|30–45°; cạnh 14–22 hoặc 54–68 mm. Test màu–vị trí giữ lại một quadrant cho từng màu; train vẫn thấy mỗi màu ở ba quadrant khác.

Depth → full-scene XYZ trong robot base frame → lấy mẫu đều **N=4.096**, không crop theo GT. Simulator mask chỉ tạo nhãn **K=128 điểm bề mặt nhìn thấy**; mỗi scene sinh bốn task 0:red / 1:blue / 2:green / 3:yellow. Không đưa GT center, size, mask, pose hay object tokens vào model.

## Kiến trúc

`RGB → frozen SAM3 vision_features → sample tại Proj(XYZ) → [XYZ, feature] → MLP + Transformer → task embedding + point decoder → 128 XYZ`.

Dùng `get_vision_features(...).fpn_hidden_states[-2]`: **256×72×72**, trước prompt/mask heads; không text prompt. SAM3 ViT + neck 454M tham số frozen. Point MLP 259→128→128, encoder 2 lớp, decoder 2 lớp, 4 attention heads, task embedding 4×128; **878.979 tham số trainable**. Chỉ loss Chamfer Euclidean đối xứng unsquared. AdamW, batch 8 scene/32 task, 6.000 bước; chọn step 5750 bằng validation. Train 3.3 phút trên RTX3090.

## Kết quả

Mỗi dòng test có 512 truy vấn; CD và lỗi tâm tính bằng mm.

| Test | CD↓ | Centroid↓ | Center-aligned CD↓ | Chọn đúng object↑ |
|---|---:|---:|---:|---:|
| IID chưa thấy scene | 6.80 | 11.60 | 5.02 | 100.0% |
| Vị trí XYZ ngoài train | 23.67 | 34.36 | 5.33 | 98.8% |
| Rotation ngoài train | 10.12 | 17.32 | 5.47 | 99.4% |
| Kích thước ngoài train | 11.29 | 17.32 | 5.71 | 97.9% |
| Tổ hợp màu–vị trí mới | 9.20 | 15.99 | 5.04 | 100.0% |

Centroid là trung bình cloud nhìn thấy; accuracy chấm bằng GT object có Chamfer tới prediction nhỏ nhất. Đổi bốn task trên cùng scene IID chọn đúng cả bốn ở 100% scene; vị trí ngoài train ở 95,3% scene.

**Nhiễu paired trên 128 scene IID, model chỉ train dữ liệu sạch:**

| Nhiễu | CD(mm) | Accuracy |
|---|---:|---:|
| RGB Gaussian σ=10/255 | 7.71 | 99.8% |
| RGB Gaussian σ=25/255 | 9.18 | 100.0% |
| Depth Gaussian σ=2 mm | 6.79 | 100.0% |
| Depth Gaussian σ=5 mm | 6.81 | 100.0% |
| Mất 20% điểm | 7.08 | 100.0% |
| 5% outlier | 6.91 | 99.8% |
| RGB σ15 + depth σ5 mm + mất 20% + 2% outlier | 8.47 | 99.8% |

Feature được tính lại từ RGB nhiễu; depth noise dịch XYZ theo camera ray; dropout/outlier chọn điểm không dùng GT. GT đích sạch giữ nguyên. Nhiễu kết hợp tăng CD 1,68 mm, giảm accuracy 0,2 điểm%; ngoại suy XYZ là điểm yếu lớn hơn nhiễu sensor đã test.

**Chưa đủ bằng chứng về alignment 2D–3D:** zero SAM3 feature giảm accuracy xuống 28,5%, constant task 0 xuống 25%; nhưng hoán vị feature giữa các điểm vẫn 100%. Đặt nhánh XYZ về 0, giữ feature đã sample, vẫn 99,6% với CD 10,30 mm; đây không phải RGB-only baseline. Model có dùng XYZ để cải thiện hình học, nhưng có thể dựa mạnh vào thông tin ảnh/vị trí trong SAM3 features dưới camera cố định.

Giới hạn: một seed; nền/robot/camera cố định, chưa random vị trí ghế/thảm hay test ảnh thực; không reconstruction mặt khuất. Các input/target đã audit, loss/metrics đối chiếu CPU và inference online không đọc GT đã chạy.

Chi tiết cấu hình, diagnostic và artifacts: [e135_grounding/DETAILS.md](/workspace/robotics_assets/experiments/e135_grounding/DETAILS.md). [Viewer đổi task trên cùng scene](http://localhost:17070/grounding/viewer.html). Checkpoint: `/workspace/robotics_assets/experiments/e135_grounding/train/best.pt`; code: `OATFlow/task_cloud/`.
