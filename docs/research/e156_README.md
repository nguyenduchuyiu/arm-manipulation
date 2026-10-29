# E156 — encoder gốc 4k + EMD + RSP

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: RSP coordinate-scale đạt selection cao nhưng target-only shape response gần0; dùng data physical resize nhất quán. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành18k và evaluation; best17500.

Theo yêu cầu, dừng E154/E155 và quay lại đúng `PointCloudGrounder` của E140: XYZ×2 + LayerNorm(SAM3feature256), MLP259→128→128, TransformerEncoder2 layers/4heads/FF384, task embedding4×128 +128 learned queries, TransformerDecoder2 layers, XYZ MLP128→128→3. **878.979 trainable**, tất cả encoder/task/fusion/decoder được train từ đầu seed135; SAM3 pretrained454.038.784 đóng băng. Không PTv3/pretrained3D, không đổi kiến trúc.

## Data

E135/E140 train1.024scene×4task, val128, IID128; camera cam_front cố định, RGB320×240, bốn cube màu và robot/ghế/thảm/clutter như cũ. Input đúng **4.096 scene_points + pixel_uv gốc** của từng scene, không FPS16k hoặc radius crop. GT128visiblepoints/task giữ nguyên. Không GT center/size/mask/object tokens vào model. GT masks chỉ tạo supervision hoặc tạo/chấm counterfactual.

Cache per-point E135 đã dọn trước đó. Reuse nguyên frozen SAM3 spatial maps E150 (vision_features256×72×72, cùng pretrained-weight SHA), reproject XYZ gốc bằng K/T như code E140 (sai số so pixel_uv dưới1e-3px), rồi bilinear sample align_corners=False/border, giữ feature-point correspondence. Chỉ giữ sampled features FP16 trong GPU memory cho lượt train; không ghi cache feature mới ra đĩa, không chạy lại SAM3. BF16 feature-map values trong FP16 cache được giữ chính xác; không giả định per-point sampling bit-identical cache E140 đã xóa.

## Training

18.000updates từ đầu, batch8scene/32queries như E140, khoảng140,625 lượt qua train-scene dataset. AdamW3e-4→3e-5 cosine/warmup100, wd0,01, clip1. BF16 autocast forward như encoder gốc, FP32 weights/optimizer và EMD; output head giữ nguyên encoder gốc. Sole loss exact equal-mass EMD128↔128: mean unsquared Euclidean optimal1-to-1 assignment, không auxiliary loss/Sinkhorn. Assignment SciPy linear_sum_assignment trên detached costs; coordinate gradients từ selected pairs.

Full **RSP** như E140: Haar SO(3), sx/sy/sz độc lập Uniform[0,6;1,4], tXYZUniform[−0,5;0,5]m. Áp dụng `P′=(P*s)@R.T+t`, GT y hệt, trước XYZ×2; F/task giữ nguyên. Scale là augmentation hệ tọa độ, không scale vật thật ở simulator. Data RNG135, augmentation CUDA RNG140 và batch8 như E140; same fixed validation R/s/t lấy nguyên E140. No sensor-noise augmentation.

Chọn best bằng fixed-RSP validation **EMD** mỗi500updates, canonical validation diagnostic. Log cả EMD và directional/symmetric Chamfer; E140 chọn best bằng Chamfer nên final selected-checkpoint comparison không phải chỉ đổi một yếu tố chọn checkpoint. Giữ best và một resume checkpoint mỗi1000steps; log50steps. Train18k rồi tự evaluation/report; không dùng test chọn checkpoint.

## Verification/evaluation dự kiến

Preflight: source/state keys/shapes/878.979params giữ nguyên E140, original4k rows, shared scene/GT RSP, unchanged features, inverse transform; full finite forward/backward với EMD. Benchmark process tách training để reset initialization seed. Các lượt cũ được giữ artifacts nhưng dừng, không chạy song song GPU.

HeldoutIID128×4task, cùng E140 canonical/global_scale/combinedRSP/SE3 transforms và18.944target-onlyshape cases. Metrics: exactEMD, directionalCD raw coordinates hai chiều, robust Q95−Q5 XYZ, full-visibleGT voxel4mm coverage2/5mm so GT128reference, targetselection, equivariance và extent-response. Không căn centroid cho metrics chính; legacy local intervention giữ empirical centroid riêng từng sampling, RGB/features stale nên là OOD counterfactual, không physical rerender. Các split generalization khác/noise test chưa chạy trong protocol này. Output128XYZ/task; không action horizon/flow steps/noise seed.

Artifacts `/workspace/robotics_assets/experiments/e156_rsp_emd`, code `OATFlow/rsp_emd`, exact config `train/config.json`, logs `train.log`, progress `train/progress.json`. Window riêng `e156-log` trong attached `ssh_tmux`, remain-on-exit on, tee stdout/stderr và giữ current window. Tại thời điểm launch chưa có kết quả cuối; xem phần Kết quả hoàn thành bên dưới.

## Launch đã xác nhận

E154/E155 đã dừng, cửa sổ logs giữ lại; không còn process train hai lượt đó trên GPU. E156 chạy trong `ssh_tmux:2:e156-log` (Ctrl+B rồi2), remain-on-exit on; current window0 giữ nguyên. Benchmark6 fullupdates pass, source/state-shapes encoder giữ nguyênE140. Median0,137s/update, batch8scene/32queries; estimate18k~41,1phút trước validation/eval. Training thực tế step350, trainEMD(last50)185.70mm, elapsed47.7s, peak2.99GiB gồm sampledfeatureGPUcache1024+128scene. Đây là snapshot launch; xem kết quả hoàn thành bên dưới.

First validation step500: RSP EMD147.049mm, RSP Chamfer130.638mm, canonical EMD139.050mm. First best checkpoint đã lưu; đây là early snapshot, chưa kết luận hội tụ.

## Kết quả hoàn thành

Train 29.77 phút, eval 6.53 phút; peak allocated 2.99GiB. Best theo fixed-RSP validation EMD, không dùng test chọn checkpoint.

| IID test | EMD(mm) | Pred→visibleGT(mm) | VisibleGT→pred(mm) | Coverage5mm | GT128 reference5mm | Target accuracy |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 10.937 | 6.933 | 6.376 | 52.91% | 92.21% | 100.00% |
| global_scale | 11.579 | 7.100 | 6.752 | 52.42% | 91.37% | 100.00% |
| combined_RSP | 12.108 | 7.447 | 6.914 | 49.98% | 91.47% | 99.95% |
| SE3 | 11.345 | 7.142 | 6.512 | 51.11% | 92.43% | 100.00% |

Robust extent Q95−Q5 XYZ trong summary/archives, đồng thời min-max extent và equivarianceCD. Coverage chỉ visible surface, voxel4mm/tolerance2/5mm, GT128reference cùng point budget. Không dùng alignedCD làm shape error thuần.

| Target-only variant | Cases | Coverage5mm | Extent response gain | Robust extent response gain |
|---|---:|---:|---:|---:|
| sham | 512 | 52.91% | — | — |
| axis_scale | 6144 | 52.58% | -0.0006 | -0.0008 |
| mixed_scale | 6144 | 49.87% | -0.0002 | -0.0007 |
| bend | 6144 | 52.77% | -0.0106 | -0.0107 |

Exact E140 stale-feature OOD variants. Input subset, GT128 and visible-GT surface each preserve their own empirical centroid, as the legacy protocol; not one shared continuous physical deformation.

Một seed, synthetic scenes/camera cố định. Training objective và checkpoint-selection khác E140; per-point features re-sample từ frozen maps, không chứng nhận bit-identical cache đã xóa. So learning theo cùng fixed-validation Chamfer; so selected checkpoint cần đọc cả selection criterion. SE3 pass không tự chứng minh local shape grounding. [Viewer](http://localhost:17070/rsp-emd/viewer.html).

## Đối chiếu và diễn giải sau kiểm tra

**Encoder gốc đã học chọn object và tọa độ tốt trên IID/RSP**, khác lượt PTv3 E153 chưa grounding tốt. E156 validationRSP EMD147,05mm ở500 →23,42mm ở8000 →19,06mm ở12000 →best12,33mm ở17500. Mean trainingEMD cửa sổ2k cuối:15,12→13,16→12,19mm; vẫn cải thiện chậm, không tuyên bố plateau tuyệt đối. Selected checkpoint17500 đạt canonicalEMD10,94mm/selection100%; combinedRSP12,11mm/99,95%; SE311,35mm/100%.

Đã chấm lại saved canonical predictions E140 bằng chính metric CPU/raw/full-visibleGT hiện tại, không train/inference lại:

| Metric canonical IID512queries | E140 Chamfer | E156 EMD |
|---|---:|---:|
| EMD(mm) |15,768|10,937|
| Pred→GT128(mm) |5,927|7,581|
| GT128→pred(mm) |7,133|6,397|
| Pred→visibleGT(mm) |5,288|6,933|
| VisibleGT→pred(mm) |7,099|6,376|
| Coverage5mm |37,70%|52,91%|
| Robust extent XYZ(mm) |21,37/25,47/25,71|29,64/27,82/28,92|

GT128reference coverage5mm92,21%, robustextent35,06/33,37/34,51mm. EMD giảm nhưng pred→GT tăng trong khi GT→pred giảm: coverage tốt hơn là một tradeoff với độ sát bề mặt, không phải mọi metric đều tốt hơn. So với E140, checkpoint-selection metric đổi và SAM3 per-point features đã được resample; không coi đây là một ablation bit-identical chỉ có loss.

**Scale/shape grounding vẫn chưa đạt:** global-scale extent gain0,0212 (ideal1), robustgain0,0207. Target-only axis/mixed scale gains−0,00063/−0,00017; prediction gần như giữ shape cũ dù XYZ target đổi. Đây là bằng chứng về đáp ứng thấp trên các phép thử OOD đã định nghĩa, không kết luận một cơ chế nội bộ duy nhất. Coverage5mm canonical mới52,91% vẫn thấp hơn GT128reference92,21%.

Kiểm tra hoàn thành: checkpointSHA/source hashes khớp, beststep khớp validation, fixedvalidation R/s/t bằng E140, recompute summary means/selection từ toàn bộ global archives;16globalqueries đối chiếu CPUfloat64 exactEMD/directionalCD, maxerror3,94e-6mm; sham predictions bit-identical, distractorXYZ giữ nguyên, centroid target sai số<1e-6m. Viewer/curve HTTP200. Xem completion_audit.json và comparison_E140_metrics.json.

## Rotation-only audit bổ sung

Đã test checkpoint17500, sameheldoutIID128, scale1/t0, X/Y/Z90° và4randomSO3sets, BF16+FP32. FP32 centererror7,08mm so no-rotationcentercontrol462,55mm; shapeCD tới rotatedreference2,71mm so staticcontrol2,91mm (chỉ giảm6,7%). Shape gần unrotatedreference hơn ở93,7%queries. Có position rotation tracking, shapeequivariance còn yếu trên pointsetdiagnostic này; không coi là angularposeaccuracy. Chi tiết [rotation audit](e156_rotation.md), artifacts test_rotation/.
