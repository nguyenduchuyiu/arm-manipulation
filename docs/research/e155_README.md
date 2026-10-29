# E155 — PTv3 SE3: Chamfer thay EMD

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: PTv3+SE3 Chamfer dừng950; không coi run ngắn là kết luận về augmentation. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: đã dừng theo yêu cầu, last logged step950; chưa hoàn thành18k/evaluation.

Đối chứng E154: chỉ thay **training loss EMD bằng symmetric half-sum mean nearest Euclidean Chamfer**, unsquared, đơn vị mét. Không thêm loss. Giữ tiêu chí chọn checkpoint **minimum fixed-SE3 validation EMD mỗi500updates** như E154, đồng thời log validation Chamfer và hai chiều riêng để so cùng metric. Không so trực tiếp training EMD với training Chamfer.

## Data và kiến trúc

Cùng E135 train1.024scene/4task, val128, IID128. Reuse nguyên radius1,2m→exactFPS16.384 indices E153 và frozen SAM3 vision_features256×72×72 maps E150; UV association giữ nguyên. GT128/task, RGB320×240 từ camera cam_front cố định như cũ. Robot, ghế, thảm và clutter trong vùng giữ nguyên. Không scale vật trong simulator, không augment scale XYZ, không GTmask/center/size/object tokens vào model. Không tạo cache ảnh/feature/FPS mới.

Official PTv3 Mode1 U-Net +128-query task decoder, 47.684.771 trainable (backbone47.178.272). Toàn PTv3/head từ đầu seed135, không pretrained PTv3/không warm-start. Chỉ SAM3 pretrained454.038.784 đóng băng; cùng weights SHA như E154. Output128XYZ/task trong robot-base frame sau augmentation; không phải action policy, không action horizon/flow steps.

## Training control

18.000updates; batch2scene×4task=8query, không accumulation; khoảng35,16train-scene passes. Cùng data RNG135, augmentation CPU RNG140 và sampler E154 (consume scale random draws như cũ nhưng S=I) nên cùng scene draws/R/t sequence. Full Haar SO(3), translation từng trục Uniform[−0,5;0,5]m; apply chung scene/GT trước XYZ×2; features/task giữ nguyên. FPS canonical trước augmentation, voxel metadata recompute sau augmentation.

AdamW3e-4→3e-5 cosine, warmup100, wd0,01, clip1. BF16 attention/MLPs; FP32 sparseconv, XYZ head, master weights/optimizer/loss; TF32off. Exact same fixed validation R/t/s của E154. Checkpoint selection bằng EMD giữ nguyên để thay đổi thực nghiệm chỉ nằm ở training objective. Chỉ giữ best và một resume mỗi1000steps.

Preflight kiểm tra rigid transform/correspondence, finite full forward/backward Chamfer; kiểm tra config/source/cache/validation match E154 trước training. Train từ đầu tách process sau benchmark để reset seed. Log riêng window `e155-log` trong attached tmux `ssh_tmux`, tee stdout/stderr; giữ window người dùng đang chọn. E154 tiếp tục chạy. GPU dùng chung: so tốc độ hội tụ theo optimizer updates/scene exposures; không diễn giải wall-clock như benchmark loss độc lập.

## Evaluation dự kiến

Best chọn từ validation, không dùng test chọn checkpoint. Canonical/SE3 heldoutIID128scene×4task; giữ nguyên E140 global_scale/combinedRSP và18.944target-onlyshape cases làm OOD stress tests như E154. ExactEMD, directionalCD riêng hai chiều, raw coordinates không căn centroid; robust Q95−Q5 XYZ; full-visibleGT voxel4mm coverage2/5mm so GT128reference, selectionaccuracy, equivariance và extent-response. Masks chỉ dùng tạo/chấm counterfactual. Các tập input/GT128/visibleGT của legacy local deformation giữ tâm empirical riêng; không phải simulator rerender. Noise augmentation không có, noise seed không áp dụng.

Report tự tạo sau train/eval, đường học so E154 trên cùng validation EMD và Chamfer. Nếu E154 chưa xong khi report được tạo, comparison ghi snapshot thực tế. Chưa có kết quả hoặc kết luận hội tụ E155.

Artifacts `/workspace/robotics_assets/experiments/e155_se3_cd`, code `OATFlow/ptv3_se3_cd`; cấu hình chính `train/config.json`, audit `training_control.json`, log `train.log`, tiến độ `train/progress.json` và `train/validation.jsonl`. Run bằng `.venv-concerto/bin/python -m OATFlow.ptv3_se3_cd.benchmark`, sau đó `... -m OATFlow.ptv3_se3_cd.train`; mỗi output mới chỉ tạo một lần.

## Launch đã kiểm tra

Benchmark6 full updates finite Chamfer/gradient, shared-transform/correspondence checks pass. Median0,752s/update khi chạy song song E154, peak7,25GiB; không dùng con số này để so throughput loss. Audit training_control.json:45config fields identical E154; fixedR/s/t arrays identical và baseline sources hashes unchanged. Training thực tế đã qua50updates (trainChamfer573,49mm, last50); chưa có validation hoặc kết luận hội tụ. E154 tiếp tục chạy, tổngGPU VRAM hai process khoảng15,5GiB. Window `ssh_tmux:1:e155-log`, Ctrl+B rồi1, remain-on-exit on; window0 người dùng vẫn được chọn.

## Dừng theo yêu cầu

Dừng cảE154/E155 để train encoder ban đầu4096points +EMD/RSP. Checkpoints/logs được giữ; snapshot chính xác trongstop.json. Không có kết quả test cuối.
