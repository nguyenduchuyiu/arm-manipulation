# E151 — global XYZ FPS, xem point cloud trước khi train

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Preview FPS; hàm FPS đã giữ gọn trong E157/sampling.py. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: FPS preview hoàn thành, **chưa train**. Full74k E150 đã dừng theo yêu cầu người dùng để xem sampling trước; last training metric step600, tqdm cuối625, best checkpoint500 chưa hội tụ. Frozen SAM3 map cache E150 giữ để tái sử dụng, không chạy encoder lại.

Data preview: đúng E140/E135 `train/00000.npz`, cam_front320×240, robot/ghế/thảm/bàn/sàn và4 cubes đỏ/xanh dương/xanh lá/vàng. Lấy toàn bộ74.356 valid depth points trong robot base frame; màu RGB từ full_pixel_uv. Không rerender scene, không thay XYZ, không dùng GT để chọn điểm.

Sampling: **exact greedy farthest point sampling** trên Euclidean XYZ float32, bắt đầu row0, mỗi bước chọn điểm có khoảng cách nhỏ nhất tới tập đã chọn là lớn nhất. Không normalize từng trục, không crop/workspace/object bias, không dùng RGB/features/masks. Chọn16.384 points; prefix8.192 cũng được xuất để xem. Uniform16.384 không replacement, seed150000, làm mốc so sánh. GT labels chỉ đếm số điểm/cube sau sampling và phóng to hình để kiểm tra, không đưa vào FPS.

Preview chưa huấn luyện model, không có checkpoint mới, không inference SAM3/PTv3, không training loss/optimizer. Dự định tiếp theo sau khi người dùng xem sampling là giữ PTv3+EMD và full RSP trên1.024scene/4task; chưa chốt chạy FPS toàn dataset. N16.384 là mốc đã đề xuất trước, không tự khởi chạy train.

Artifacts `/workspace/robotics_assets/experiments/e151_fps`: indices/points/colors, metadata, PNG và Plotly viewer. Source `OATFlow/ptv3_rsp/fps.py`, `fps_preview.py`; input file giữ nguyên ở E135. Thời gian/coverage radius và số điểm từng cube sẽ được bổ sung sau khi chạy. Một scene và một seed FPS không bảo đảm counts tương tự trên mọi scene.

## Preview kết quả

Exact FPS16k hoàn tất trong1.61s; CPU KDTree xác nhận cover radius của GPU trong0,01mm; selected indices unique. Đúng train/00000, không model mới.

| Sampling | Points | Red | Blue | Green | Yellow | Max nearest distance fullscene(mm) |
|---|---:|---:|---:|---:|---:|---:|
| full | 74356 | 489 | 348 | 604 | 765 | 0.00 |
| uniform16384 | 16384 | 100 | 76 | 130 | 160 | 232.16 |
| fps16384 | 16384 | 57 | 27 | 72 | 88 | 6.11 |
| fps8192 | 8192 | 30 | 20 | 33 | 49 | 8.92 |

FPS phân bố theo khoảng cách metric toàn scene nên có thể dành nhiều điểm cho vùng xa/clutter và ít điểm hơn cho cube gần, khác uniform sampling theo pixels. Counts trên1 scene, chưa phải audit toàn1.024scene. PNG hàng dưới chỉ zoom vùng4cube để quan sát, không crop đầu vào FPS. [Plotly pointcloud](http://localhost:17070/fps/viewer.html), `pointcloud.png`, `sample.npz`, `metadata.json`. Chưa train.

Web publication: server cũ có cwd trỏ vào temporary directory đã bị xóa. Đã đổi wrapper supervisor `sam3_viewer` sang `/workspace/robotics_assets/experiments/web`, giữ127.0.0.1:17070 cho VS Code forwarding, khôi phục các URL alias cũ và FPS; legacywebpath symlink về stable root. FPS viewer/PNG/metadata và E149/E140 viewer trảHTTP200. Không đổi Caddy/public ports.
