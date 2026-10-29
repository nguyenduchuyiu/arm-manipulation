# E150 — full-scene PTv3 + exact EMD, E140 RSP augmentation

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: PTv3 full-scene RSP tốn bộ nhớ/thời gian, nhánh được dừng/thay thế. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: đã dừng lượt full74k theo yêu cầu xem FPS trước khi train; last metric600, tqdm cuối625, best500. Chưa hoàn thành18k hoặc evaluation.

## Data và model

Dùng đúng dataset E140/E135: train1.024 scene, val128, IID128, thêm các split held-out positions/rotations/sizes/combinations128 mỗi loại. Bốn cube đỏ/xanh dương/xanh lá/vàng, task0/1/2/3; robot, ghế, thảm, bàn, sàn giữ nguyên. Cam_front320×240 như trước. RGB/depth/GT128 mỗi task giữ nguyên; dùng **toàn bộ74.356 valid depth points/scene** (`full_scene_points`, `full_pixel_uv`) thay vì4.096 points của E140 cũ. Không crop/mask-based input sampling. GT dùng supervision/test construction, không vào model.

Kiến trúc đúng E149: official PTv3 Mode1 U-Net cls_mode=False, grid1 mm, patch64, non-Flash, finest-point64D → projection128D → task-conditioned128-query decoder. Backbone47.178.272; tổng47.684.771 trainable. **Toàn bộ PTv3 và head train từ đầu seed135; không pretrained PTv3 hoặc warm start E149.** Frozen SAM3 vision encoder+neck454.038.784 pretrained parameters, lấy `get_vision_features(...).fpn_hidden_states[-2]`,256×72×72, không prompt/masks. SAM3 weights hash6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a.

Chỉ cache FP16 map256×72×72/ảnh (~4,75GB cho1.792 scene), không copy RGBD và không cache N×256 cho cả dataset. Bilinear sample FP32 tại original canonical pixel_uv, align_corners=False, rồi lưu feature tensor FP16 trong batch. Canonical feature-point correspondence giữ nguyên khi augment3D, không project augmented XYZ bằng camera cũ.

## RSP và training dự kiến

`P′ = R diag(s) P+t`, `G′ = R diag(s) G+t`: Haar-uniform SO(3); sx/sy/sz độc lậpUniform[0,6;1,4]; tx/ty/tz độc lậpUniform[−0,5;0,5]m. Scale trước rotation trước translation, trước XYZ×2. Cùng transform cho full scene và cả4 GT targets, SAM3 features/task IDs không đổi. Dùng lại implementation `OATFlow/scale_cloud/transforms.py`, augmentation seed140. Camera/extrinsics không vào model; đây là coordinate augmentation, không rerender scene.

Train18.000 optimizer updates, AdamW3e-4→3e-5 cosine/warmup100, weight_decay.01, clip1; batch dự kiến2 scene/8 task queries (chờ benchmark GPU), không auxiliary loss. Chỉ exact equal-mass EMD128↔128, mean Euclidean không bình phương, m; SciPy exact assignment và FP32 gradient qua matched pair như E149. Encode mỗi scene một lần rồi decode cả4 task. BF16 attention/MLP, FP32 sparse convolution/XYZ head/master weights/optimizer/loss, TF32 off. Voxel map được dựng lại từ augmented XYZ mỗi batch, inverse giữ feature cho từng original point. Không dùng GT center/size/mask/object tokens trong forward.

Đổi so với E140 cũ: architecture878.979→47.684.771 params, lossChamfer→EMD, input4.096→74.356 points; batch8scene/32query cũ không mặc định giữ khi model/fullcloud lớn hơn. Số bước giữ18k; exact batch/runtime chốt bằng benchmark và ghi config, không coi là matched-compute/capacity comparison.

Best chọn bằng EMD trên128 validation scenes với chính fixed RSP transforms của E140, mỗi500 updates; canonical validation chỉ diagnostic. Không test-driven selection. Giữ best và một resume optimizer checkpoint, không lưu history weights. Log ở một window riêng trong attached tmux `ssh_tmux`, giữ window sau khi kết thúc và dùng tee lưu file.

## Evaluation dự kiến

128 IID scenes,4 tasks: canonical, global anisotropic scaling, combinedRSP, SE3(rotation+translation). Dùng lại4 transforms/scene lưu trong E140 archives để đối chiếu đúng bộ biến đổi. Target-only shape test dùng cùng scene/task/variant cases của E140/E139, nhưng mask được suy ra từ full pixel_uv/GT labels thay vì4k-point masks cũ. Features/other objects giữ nguyên; staleRGB/feature counterfactual là OOD, không physical rendering.

Đo EMD, directional Chamfer riêng pred→GT/GT→pred, Q95−Q5 XYZ và max−min, surface coverage với voxel4 mm và ngưỡng2/5mm trên visible GT, có GT128 reference; Z bins khi phân tích shape. Equivariance với transformed canonical prediction và extent-response gain (ideal1) cho global/target-only tests. Chỉ camera-visible surface, không amodal mesh. Centroid-aligned CD nếu giữ để đối chiếu E140 là metric phụ, không diễn giải thành shape error thuần khi sampling khác nhau.

Artifacts `/workspace/robotics_assets/experiments/e150_rsp`; dataset gốc `/workspace/robotics_assets/experiments/e135_grounding`, source code `OATFlow/ptv3_rsp`, reused model/loss `OATFlow/ptv3_cloud`. SAM3 map provenance và benchmark sẽ nằm trong `features/config.json`, `benchmark.json`; cấu hình/tiến độ thực tế trong `train/config.json`, `train/progress.json`. Chưa có kết quả mới để báo cáo.

## Benchmark và bộ nhớ đã đo

Feature maps1.792 scene hoàn thành trong314,30s, BF16→FP16 conversion error0. Batch2scene x4task OOM24GB; dùng1scene/microbatch, accumulation2 để đạt2scene/8task/update. BatchNorm official dùng1scene mỗi microbatch, không hợp statistics qua accumulation. Benchmark randomRSP1,027s/update, scale1,4 worst1,247s, allocated peak10,68–11,89GiB. Đây là ước tính~5,14h train18k chưa tính validation/eval, nhưng run đã dừng để đổi sampling. Memory audit: before-forward0,59GiB, afterPTv310,98GiB, afterdecoder11,47GiB, afterbackward0,77GiB; peak11,64GiB. Giữ `memory_audit.json`, `benchmark.json`, `benchmark_B2_OOM.log`.
