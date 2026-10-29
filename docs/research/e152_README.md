# E152 — bỏ nền xa, FPS trong bán kính robot base

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Preview radius+FPS; giữ kết quả mật độ, hàm FPS chung ở E157. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: chuẩn bị point-cloud preview, **chưa train**. User yêu cầu bỏ phần scene ở xa sau khi xem globalFPS E151. Lượt full E150 vẫn dừng, không tự khởi chạy training mới.

Preprocess cố định: giữ `||XYZ_base||₂ ≤ 1,2 m` quanh origin robot base trong canonical frame, rồi exact XYZ-only greedyFPS16.384; prefix8.192 để xem. Lọc/FPS trước RSP, không lọc sau random rotation/scale/translation. Không dùng GT mask, object center/size, taskID hoặc RGB để quyết định giữ/chọn điểm. GT chỉ audit điểm cube sau preprocessing. XYZ không đổi; chỉ chọn subset original row indices, giữ full_pixel_uv/RGB/SAM3 feature correspondence. RGB và feature map SAM3 vẫn là ảnh gốc, không rerender simulator.

Preview đúng E140/E135 train/00000, camera cam_front320×240. Robot và clutter gần nằm trong vùng giữ lại; phần nền xa ngoài1,2m bị bỏ. Giữ nhãn target128 ban đầu. Scene mẫu:74.356 points→69.155 sau radius crop, toàn bộ489/348/604/765 điểm đỏ/xanh dương/xanh lá/vàng vẫn còn. Đây là crop theo khoảng cách đã được user yêu cầu, thay nguyên tắc full-scene không giới hạn trước đó.

Chưa có model/loss/optimizer mới. FPS và sampler uniform trước đây E151 chỉ là baseline preview. Dự định train sau khi xem: PTv3 từ đầu + frozen SAM3 + exactEMD + RSP trên1.024scene/4tasks; chưa khởi chạy. Shared SAM3 maps E150 được giữ để dùng lại.

Artifacts `/workspace/robotics_assets/experiments/e152_near_fps`; code `OATFlow/ptv3_rsp/near_fps_preview.py`, reuse `fps.py`. Ảnh và viewer so sánh E151 globalFPS16k với cropped raw/FPS16k; hàng dưới chỉ zoom4cube. Audit toàn1.792 scene sẽ kiểm tra target retention sau radial filter, không thay GT hoặc file dataset gốc.

## Kết quả preview

Radius crop audit toàn1.792 scene: **0 target points bị loại**. Exact FPS16k hoàn tất trong1.95s trên train/00000. Giữ69,155/74,356 source points.

| Sampling | N | Red | Blue | Green | Yellow |
|---|---:|---:|---:|---:|---:|
| global_fps16k | 16384 | 57 | 27 | 72 | 88 |
| near_all | 69155 | 489 | 348 | 604 | 765 |
| near_fps16k | 16384 | 74 | 47 | 91 | 99 |
| near_fps8k | 8192 | 44 | 23 | 53 | 68 |

[Viewer Plotly](http://localhost:17070/fps-near/viewer.html), `pointcloud.png`, `sample.npz`, `metadata.json`, `dataset_audit.json`. Sampling sử dụng XYZ thuần; camera RGB gốc và original row IDs/uv bảo toàn. Chưa train.
