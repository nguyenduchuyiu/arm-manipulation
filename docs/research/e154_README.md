# E154 — bỏ scale, train full SE(3) với PTv3 + EMD

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: PTv3+SE3 EMD dừng5300; không coi run chưa đủ budget là chứng minh SE(3) sai. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: đã dừng theo yêu cầu, last logged step5300; chưa hoàn thành18k/evaluation.

Thay đổi chính so với E153: **S=I**, training/validation chỉ `P′=RP+t`, `G′=RG+t`. Giữ Haar-uniformSO(3), independent tXYZUniform[−0,5;0,5]m. Không yaw-only. RGB/depth/simulator và vật gốc không đổi. Giữ feature-point correspondence, không reproject augmentedXYZ bằng camera cũ.

## Data/model/training

Reuse nguyênE153 radius1,2m→exactFPS16.384 index cache và frozen SAM3 mapsE150; không FPS/encode lại, không copyRGBD. Train1.024scene,4task màu0:red/1:blue/2:green/3:yellow; val128 và IID128. GT128/task giữ nguyên. Robot và clutter gần trong scene; không GT center/size/mask/object tokens vào model. Lọc/FPS canonical trước augmentation, XYZ×2 sau augmentation.

PTv3 officialMode1 U-Net cls_mode=False, grid1mm, patch64, non-Flash, point64D→128D→128-query taskdecoder,47.684.771 trainable(47.178.272 backbone). Frozen pretrainedSAM3 vision encoder+neck454.038.784, vision_features256D/point. Toàn PTv3/head **train từ đầu seed135**, không warm-start E153, không pretrained PTv3. Same architecture/source/optimizer/batch/precision/schedule; không đổi normalization hoặc thêmXYZ skip.

18.000updates, batch2scene/8task, no accumulation. AdamW3e-4→3e-5 cosine/warmup100, weight_decay.01, clip1. Chỉ exactEMD128↔128 mean unsquared Euclidean1–1 assignment, m; BF16 attention/MLP và FP32 sparseconv/XYZ head/master weights/optimizer/loss; TF32off.

Augmentation RNG seed140 dùng cùng R/t sequence E153: vẫn consume scale RNG draws để giữ sequence nhưng **s luôn[1,1,1] và không apply random scaling**. Data RNG135 nhưE153, cùng scene draws và model initialization. Validation dùng cùngR/t của128 fixedE153 val scenes, thay mọi s bằng1, chọn best theo **SE3-validationEMD mỗi500updates**; cleanval diagnostic. So sánh E153↔E154 trên cùng heldoutcanonical/SE3 test, không so trực tiếp RSP-val với SE3-val như cùng phân phối.

Giữ best và một resume model+optimizer/RNG mỗi1000steps, log50updates. Window riêng trongattached tmux `ssh_tmux`, remain-on-exit on, tee stdout/stderr.

## Verification và evaluation

Preflight: R/t sequence khớp E153 vàs=1, RᵀR=I/detR=1; rigid distances/GT chungtransform/inverse; frozenfeatures unchanged; cùngFPS/GT/model/source. Full forward/backward finite-gradient benchmark. Không tạo feature cache mới.

Sau train: heldoutIID128scene/4task, canonical vàSE3 là chính. Giữglobal_scale/combinedRSP vàtarget-onlyshape cases củaE140 làm **OOD scale stress tests**, không phải scaling augmentation trong training. Same raw-coordinate directionalCD hai chiều, exactEMD, robustQ95−Q5XYZ, visibleGT voxel4mm coverage2/5mm vớiGT128reference; selectionaccuracy, equivariance vàextentresponsegain. GTreference toàn bộ visible target surface, khôngFPSsubset. Legacylocalshape dùngcentroid từngsampling riêng nhưE153; không diễn giải thànhphysical rerender hoặc shape-only error.

Mục tiêu: xem bỏ scaling có cải thiện hội tụ/grounding vàSE3 soE153(canonicalEMD116,34mm,SE3EMD127,99mm; selection27,73/30,13%) hay không. Dataset/sampler/model/loss giữ nguyên; validation selection phân phốiđổi sangSE3 đúng yêu cầu.

Artifacts `/workspace/robotics_assets/experiments/e154_se3`; code `OATFlow/ptv3_fps_se3`. Chưa có kết quả mới. Exactconfig/benchmark/progress cập nhật theo trạng thái thật; báo cáo cuối sẽ ghi cả kết quả thất bại nếu không hội tụ.

## Launch thực tế

Preflight32 paired draws xác nhậnR/t trùngE153 vàmọi s=1; rigid distances, sharedscene/GT transform, inverse, feature equality, fullforward/backward/finitegradient đều pass. Benchmark batch2scene/8query median0,430s/update, peak7,25GiB; estimate18k~2,15h trước validation/eval. Không encode/FPS lại: symbolic links chia sẻ cùng files E153/E150, comparison_control.json xác nhận21 config fields giữ nguyên vàsamevalidationR/t, onlyscaleidentity.

Đã khởi chạy18k từđầu trong `ssh_tmux:6:e154-log` (Ctrl+B rồi6), remain-on-exit on, full stdout/stderr tee `train.log`. Tiến độ trongtrain/progress.json, SE3val vàcanonicalval trongtrain/validation.jsonl. Chưa có kết quả cuối hoặc kết luận hội tụ tại thời điểm launch. E153 RSPval không so trực tiếp vớiE154SE3val; report cuối so cùngcanonicalval vàheldoutSE3.

## Snapshot so với E137 — 2026-10-26 16:39 UTC

E154 đang chạy, snapshot step2800/18000: trainEMD(last50)471,43mm so587,05mm ởstep50. FixedSE3valEMD:578,10→390,25→396,47→382,25→405,93mm tại500/1000/1500/2000/2500; best2000. Có giảm ban đầu, từ1000 đến2500 chưa có xu hướng hội tụ rõ.

So cùng symmetric unsquared Chamfer tớiGT128, không căn centroid. E154CD tính bằng half-sum hai directional means đã log; không so trực tiếp trainingEMD với trainingChamfer.

|Step|E137 SE3val CD(mm)|E154 SE3val CD(mm)|
|---|---:|---:|
|500|84,21|564,35|
|1000|66,27|376,32|
|1500|37,38|382,36|
|2000|34,07|368,24|
|2500|24,13|391,79|

E137 cuối18000 SE3valCD6,75mm. E154 hiện học chậm hơn rõ theo sốupdates; chưa chạy xong. Không kết luận nguyên nhân làPTv3 hoặcEMD: E1370,879M/4096points/batch8/Chamfer, E15447,685M/radius1,2m+FPS16k/batch2/EMD; batchE137 có4lần sốscene/update. Cùngvalidation128scene/GT128, nhưng fixedSE3transform khácseed; canonicalval cùngframe gốc cũng kém hơn(E13721,64 vsE154303,48mm tại2500).

Snapshot dữ liệu vàfigure: artifacts `train/e137_comparison.json`, `train/e137_comparison.png`. Đây là snapshot, không phải kết quả cuối; không sửa hoặc ngắt training.

## Dừng theo yêu cầu

Dừng cảE154/E155 để train encoder ban đầu4096points +EMD/RSP. Checkpoints/logs được giữ; snapshot chính xác trongstop.json. Không có kết quả test cuối.
