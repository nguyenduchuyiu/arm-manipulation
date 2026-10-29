# E161 — EMD + matched pairwise distances

> **Code giữ lại:** [code/README.md](/workspace/robotics_assets/experiments/e161_pair/code/README.md). core: custom encoder + SAM3 + EMD+pair and resize diagnostics. [Danh mục/bài học](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành18k/evaluation; best18000.

## Data và kiến trúc

Giữ nguyên kiến trúc E156: original PointCloudGrounder, 878,979 tham số trainable;
4,096 full-scene XYZ + frozen SAM3 vision_features + task ID → 128 XYZ.
Hai lớp point self-attention, hai lớp query decoder, shared XYZ MLP không đổi.
Không dùng probe E159 hay phép hiệu chỉnh E160. Không warm start.

Reuse immutable E157: 1,024 train scenes, 128 validation; 4 task màu/scene,
robot/chair/mat/table và camera cố định. Cube kích thước local XYZ đã được scale
thật trong simulator (0.6–1.4), RGB/depth/GT/SAM3 features render/encode nhất quán.
Data và feature cache được symlink, không tạo lại hay nhân bản.
GT instance masks chỉ dùng supervision/evaluation, không làm input.

## Training

18,000 updates từ đầu, seed135; batch8 scenes/32 queries, BF16 forward,
FP32 parameters/optimizer/loss, TF32 off. AdamW3e-4→3e-5 cosine, warmup100,
weight decay0.01, gradient clip1. Full-scene Haar SO(3)+translation±0.5m
áp dụng đồng thời scene/GT trước XYZ×2; features gắn đúng point giữ nguyên.
Không coordinate-scale augmentation. Action representation: N/A, target XYZ cloud.

L = EMD + 1.0 × pair, cả hai unsquared Euclidean distance, đơn vị mét.
Một exact SciPy linear_sum_assignment trên cost128×128 tối ưu EMD;
reuse cùng π cho pair, không giải assignment theo combined objective.
EMD = mean ||pred_i−GT_π(i)||; pair = mean |d(pred_i,pred_j)−d(GT_π(i),GT_π(j))|,
all ordered i≠j: 16,256 pairs/query. Matching detached, hai loss có gradient XYZ.
Log riêng EMD/pair/total. Best chọn minimum fixed SE3 validation EMD mỗi500 updates,
giữ cùng criterion E157; combined validation cũng được ghi. Không chọn bằng test.

## Evaluation đã chạy

Best E161 vs E15718k EMD-only; cùng data, initialization, SE3, optimizer/schedule,
batch và 18k budget; chỉ objective thay đổi. E157 best16500 là baseline cố định.
128 scenes ×4 tasks/split: IID, positions, rotations, sizes, combinations.
IID thêm4 fixed SE3 views/scene. K128 GT/pred, RGB320×240; flow steps N/A.
Same E137 heldout transforms; same E157 physical-case seeds157600000/157700000.
Physical scale:128 scenes, target-only/all-object single-local-axis×0.6/1.4;
fresh render RGB/depth/GT và SAM3 mỗi case, cùng rejection/visibility protocol.
Không thêm sensor noise, frame budget một observation/case; không rollout.
Metrics: EMD, directional CD, coverage voxels4mm at2/5mm, robust extent Q95−Q5,
paired physical-extent response gain, target-selection accuracy và SE3 equivariance.
Best-vs-GT visualization dùng raw coordinates, không centroid alignment.

Run: `.venv-concerto/bin/python -m OATFlow.pair_cloud.benchmark`, rồi
`.venv-concerto/bin/python -m OATFlow.pair_cloud.train` (tự evaluate/report).
Exact configs và hashes: `config.json`, `train/config.json`, `data_config.json`,
`features/config.json`; verification/benchmark: `benchmark.json`.

Giới hạn: một seed, camera cố định, synthetic visible-surface labels. Pairwise loss
không tự bảo đảm shape grounding hay hội tụ; phải đo test độc lập.

## Kiểm tra trước train

Exact loss: brute-force assignment5, manual ordered pair formula, gradcheckfloat64, rigid/permutation invariance, zero-distance finite gradients, known scale units và K128=16,256 ordered pairs đều pass. CPU NumPy/SciPy metric độc lập khớp PyTorch loss. Sáu GPU updates có loss/gradient finite. Config audit khớp E157 về steps/seed/batch/LR/schedule/SE3/normalization/fixed-validation transforms; architecture source hash giữ nguyên.

Benchmark median0.160s/update (~48min18k, chưa gồm validation/evaluation); peak allocated GPU của lượt train đầy đủ ban đầu2.99GiB. Không giữ checkpoint/feature cache riêng cho preflight. Log trong tmux `ssh_tmux:e161-log`, file `../e161_train.log`.

## Kết quả hoàn thành

Train31.61phút, eval7.28phút; peak allocated2.99GiB. Best18000, validation EMD12.104mm.

| Test | Model | EMD mm | Pair mm | Aligned CD mm | Pred→surface mm | Surface→pred mm | Coverage5mm | Selection |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iid | E157 | 11.631 | 8.438 | 6.628 | 7.218 | 6.768 | 52.47% | 100.00% |
| iid | E161 | 11.574 | 6.555 | 6.210 | 7.652 | 6.153 | 51.01% | 100.00% |
| SE3 | E157 | 12.798 | 8.424 | 6.738 | 7.976 | 7.248 | 47.69% | 100.00% |
| SE3 | E161 | 12.073 | 6.749 | 6.323 | 7.696 | 6.219 | 49.96% | 100.00% |
| positions | E157 | 13.407 | 8.564 | 6.659 | 7.933 | 7.564 | 47.01% | 100.00% |
| positions | E161 | 12.999 | 6.799 | 6.302 | 8.171 | 6.646 | 46.89% | 100.00% |
| rotations | E157 | 13.191 | 8.168 | 6.538 | 8.091 | 7.018 | 50.19% | 100.00% |
| rotations | E161 | 12.386 | 6.585 | 6.194 | 8.010 | 6.187 | 51.02% | 100.00% |
| sizes | E157 | 16.621 | 13.491 | 8.383 | 10.654 | 9.363 | 51.81% | 100.00% |
| sizes | E161 | 15.316 | 7.406 | 6.397 | 10.337 | 8.078 | 42.94% | 99.80% |
| combinations | E157 | 11.759 | 8.650 | 6.709 | 7.233 | 7.023 | 51.36% | 100.00% |
| combinations | E161 | 11.618 | 6.686 | 6.298 | 7.606 | 6.246 | 50.48% | 100.00% |

| Physical test | Model | EMD mm | Pair mm | Coverage5mm | Extent gain | Robust extent gain |
|---|---|---:|---:|---:|---:|---:|
| target_only | E157 | 11.485 | 8.377 | 52.78% | 0.008 | 0.007 |
| target_only | E161 | 11.446 | 6.558 | 50.91% | 0.321 | 0.306 |
| all_objects | E157 | 11.466 | 8.316 | 53.08% | 0.008 | 0.007 |
| all_objects | E161 | 11.422 | 6.531 | 51.14% | 0.327 | 0.313 |

iid: E157−E161 EMD = +0.057mm, scene-bootstrap95%CI [-0.116,+0.227].

SE3: E157−E161 EMD = +0.724mm, scene-bootstrap95%CI [+0.509,+0.950].

Physical attempts1280, scored1271, exclusions9; all cases/rejections in physical_scale/cases.jsonl. Gains use full visibleGT paired deltas; ideal1. Raw metrics except explicitly named centroid_aligned_CD. Robust extents/Q95−Q5 and coverage2mm saved in archives/summary. No test selection.

Pair loss constrains matched interpoint distances; it does not establish independent XYZ grounding or guarantee metric-shape generalization. One seed and changing visibility remain limitations.

[Best vs GT](http://localhost:17070/pair-emd/viewer.html) · [Learning curves](http://localhost:17070/pair-emd/learning_results.png).

## Nhận xét và audit sau hoàn thành

EMD+pair cải thiện size response: target-only extentgain0.321/robustgain0.306 so EMD-only0.008/0.007; lý tưởng1. Đây là slope qua gốc trên pooled paired extent deltas, không phải accuracy. Prediction vẫn khá tròn và không khớp đầy đủ bề mặt GT. IID alignedCD6.628→6.210mm, pair8.438→6.555mm; IID EMD11.631→11.574mm có bootstrapCI gồm0. SE3 EMD12.798→12.073mm cải thiện rõ trong tập này. CoverageIID52.47→51.01%; OODsizes51.81→42.94%, dù OODsizesEMD16.62→15.32mm. Không thể kết luận geometry đã được giải quyết chỉ từ pair loss/extent gain. IID/SE3 selection100%; OODsizes99.80%.

Train50-stepmeans: EMD584.63→11.58mm, pair24.33→6.61mm. Bestval ở18000, cuối budget; chưa có bằng chứng plateau từ run này. Totaltrain31.61min vàeval7.28min.

Completion audit pass: training-source/checkpoint hashes khớp; best đúng minimum validationEMD; logtotal=EMD+pair. E157 baseline predictions bit-identical mọi split, physicalcases/GT/baseline bit-identical lượt cũ. Độc lập tính lại EMD/pair từ prediction/GT và extent/robustextent gains đều khớp. Xem completion_audit.json. Viewer chỉ best18000 vsGT, rawcoords; serverlocalhost17070 trả200.

## Gallery resize

[10 bộ ×3 mức scale](http://localhost:17070/pair-emd/resize_samples/):8 target-only,2 all-object, scene0/1/2, đủ4 màu/3 trục. RGB và crop cùng GT/best18k, hình chiếuXY/XZ/YZ và optional3D đồng bộcamera. Fixed scene/task selection, không chọn theo metric. RGBD rerender đúng seed, GT128 khớp archive; prediction lấy nguyên từ kết quả evaluation. Xem [cấu hình gallery](e161_resize_README.md). Old best-vs-GT viewer giữ nguyên.
