# E163 — Surface folding tới tổng40k

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: 40k cải thiện vị trí/EMD nhưng shape và arrangement vẫn sai; không giữ làm pipeline shape thành công. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành tổng40k/evaluation; best38000.

## Thay đổi và training

Full-state resume E162 từ update18000 tới tổng40000, thêm22000updates.
Source `e162_surface/train/resume.pt`: giữ model, AdamW moments/step counters,
data RNG, augmentation RNG, CPU/CUDA RNG; không khởi tạo lại network/optimizer.
Giữ best18k (update18000, validationEMD12.240979mm) tới khi validation cải thiện.
LR18001–40000 constant3e-5, không warmup/restart; phase0–18000 giữ nguyên
cosine3e-4→3e-5 của E162. AdamW weightdecay0.01, gradientclip1.
Batch8scenes/32taskqueries, BF16 learned ops, FP32 folding composition/params/
optimizer/loss, TF32off. Tổng312.5 scene passes; thêm171.875 passes.

Data/features immutable E157 dùng symlink:1024train,128validation,4tasks/scene;
RGB320×240, fullscene4096XYZ, frozenSAM3vision_features256D, robot/chair/mat/table.
Same physical simulator sizes, HaarSO3+translation±0.5m, không coordinate scale;
features gắn đúng point. GT128/target là visible-surface point cloud.

Kiến trúc E162 giữ nguyên964236trainableparams: originalencoder +128decoderqueries,
8-query patchaggregator,8patches×16fixedUV, c+u*a+v*b+delta. SAM3frozen454038784params.
Loss exactEMD +1.0matched allpair distances,16256orderedpairs; no addedregularizer.
Không input GTmasks/center/pose/shapeparams; action/flow/rolloutN/A.

## Evaluation đã chạy

Best theo minimum fixedSE3validationEMD mỗi500updates, inheritedbest18k;
canonical validation diagnostic, không chọn theo test. So E16340k với E16218k,
cả hai cùng SurfaceGrounder và nhận cùng observations. Không đổi kiến trúc/loss.
128scenes×4tasks mỗi IID/positions/rotations/sizes/combinations, IID thêm4SE3views
theo fixedE137 archives. K128pred/GT, một observation/case.
Physicalresize target-only/allobjects localaxis×0.6/1.4: same E157 seeds
157600000/157700000, rerender RGBD/GT/SAM3features, same clearance/visibilityrules.
Metrics: EMD/pair, directionalCD, centroid-alignedCD, robustextentsQ95−Q5,
coverage2/5mm, selection và pairedextentresponsegain. Viewer bestvsGT giữ rawXYZ,
có nút8patchcolors. Không overwrite kết quả/viewer E16218k.

Run: `.venv-concerto/bin/python -m OATFlow.surface_40k.train` (auto evaluate/report).
Config/hash: train/config.json,config.json,source_hashes.json,resume_state_audit.json.
Log live tmux `ssh_tmux:e163-log`, file `../e163_resume.log`.
Giới hạn: mộtseed/camera, visibleGT, không bảo đảm train lâu hơn sửa được shape;
so với18k là thay đổi budget và continuation LR, không phải matched-budget ablation.

## Kiểm tra resume

Passed: modelweights/buffers và AdamWmoment/step counters khớp checkpoint18k bit-for-bit. CPU/CUDA RNG, numpydataRNG và SE3generator khôi phục; nextdata/SE3draw khớp, không bỏ bước. Fixedvalidationarchive và toàn bộ parenttraining sourcehashes giữ nguyên. Configaudit giữ đúng architecture/loss/SE3/batch/seed, chỉ tăngbudget và đặtcontinuationLR3e-5.

Resume lúc đầu đã qua update18150 với gradient/loss finite. Fulltrain peakallocated2.99GiB; lượt40k và evaluation nay đã hoàn thành.

## Kết quả hoàn thành

Tổngtrain58.45phút (resume27.17phút), eval7.27phút; peak allocated2.99GiB. Best38000, validation EMD10.794mm.

| Test | Model | EMD mm | Pair mm | Aligned CD mm | Pred→surface mm | Surface→pred mm | Coverage5mm | Selection |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iid | E162 | 11.598 | 6.900 | 6.476 | 7.642 | 6.114 | 50.76% | 100.00% |
| iid | E163 | 10.260 | 6.599 | 6.173 | 6.993 | 5.514 | 54.57% | 100.00% |
| SE3 | E162 | 12.182 | 7.063 | 6.561 | 7.896 | 6.278 | 48.78% | 100.00% |
| SE3 | E163 | 10.764 | 6.762 | 6.351 | 7.287 | 5.751 | 52.36% | 100.00% |
| positions | E162 | 12.587 | 7.045 | 6.471 | 8.078 | 6.427 | 48.43% | 100.00% |
| positions | E163 | 10.974 | 6.724 | 6.254 | 7.301 | 5.732 | 52.03% | 100.00% |
| rotations | E162 | 12.989 | 6.983 | 6.568 | 8.225 | 6.549 | 47.94% | 100.00% |
| rotations | E163 | 11.154 | 6.617 | 6.152 | 7.386 | 5.805 | 52.15% | 100.00% |
| sizes | E162 | 14.746 | 7.884 | 6.793 | 9.852 | 7.804 | 44.53% | 100.00% |
| sizes | E163 | 12.770 | 6.858 | 6.014 | 8.712 | 6.734 | 49.62% | 99.61% |
| combinations | E162 | 11.607 | 7.020 | 6.519 | 7.696 | 6.208 | 50.08% | 100.00% |
| combinations | E163 | 10.402 | 6.737 | 6.252 | 7.056 | 5.629 | 54.23% | 100.00% |

| Physical test | Model | EMD mm | Pair mm | Coverage5mm | Extent gain | Robust extent gain |
|---|---|---:|---:|---:|---:|---:|
| target_only | E162 | 11.511 | 6.888 | 51.10% | 0.236 | 0.277 |
| target_only | E163 | 10.216 | 6.618 | 54.63% | 0.372 | 0.307 |
| all_objects | E162 | 11.447 | 6.869 | 51.31% | 0.242 | 0.275 |
| all_objects | E163 | 10.180 | 6.587 | 54.98% | 0.373 | 0.312 |

iid: E162−E163 EMD = +1.338mm, scene-bootstrap95%CI [+1.208,+1.465].

SE3: E162−E163 EMD = +1.418mm, scene-bootstrap95%CI [+1.293,+1.547].

Physical attempts1280, scored1271, exclusions9; all cases/rejections in physical_scale/cases.jsonl. Gains use full visibleGT paired deltas; ideal1. Raw metrics except explicitly named centroid_aligned_CD. Robust extents/Q95−Q5 and coverage2mm saved in archives/summary. No test selection.

Surface patches add geometric coupling without extra losses; no GT face assignment. This does not establish independent XYZ grounding or guarantee coverage. One seed, longer training and changing visibility remain limitations. E163 adds22k updates at constantLR3e-5 to the same E162 surface architecture.

[Best vs GT](http://localhost:17070/surface40k/viewer.html) · [Learning curves](http://localhost:17070/surface40k/learning_results.png).

## Nhận xét và audit sau hoàn thành

Train thêm22k cải thiện rõ trong tập này: IID EMD11.598→10.260mm (−1.338mm), SE3 12.182→10.764mm (−1.418mm); scene-bootstrapCI95 cải thiện đều trên0. IIDcoverage50.76→54.57%, centroiderror6.541→4.917mm, alignedCD6.476→6.173mm. Shape cải thiện ít hơn lỗi vị trí; ảnh vẫn chưa khớp đầy đủ bề mặt cube. Target-only extentgain0.236→0.372 nhưng robustgain0.277→0.307, còn cách xa lý tưởng1. Selection IID/SE3 100%, OODsizes99.61% (2/512queries sai). Không kết luận surfacefolding tốt hơn directXYZ ở cùng40k vì E161 chỉ18k.

Cuối run giảm chậm rõ: meanvalidationEMD36k–38k11.261→38k–40k11.222mm; pair6.637→6.624mm. Best38000=10.794mm,40000=11.390mm. Gần plateau hơn lượt18k, còn nhiễu giữa checkpoints; không tự launch thêm training.

Audit pass: checkpoint/source/parentresume hashes đúng,40000updates hoàn thành,best38000 đúng minimumfixedvalidationEMD,loss tổng=EMD+pair. E162baseline canonical/SE3 khớp bit-for-bit; physicalcase metadata/GT128 vàbeforepredictions khớp. Có1/1271physicalaftercase khác nhẹ baseline rerun,1/2021scoredqueryEMD lệch0.02541mm (meanabs0.00001257mm), giữrawresults. Tính lại losses và extentgains từarchives độc lập khớp; xem completion_audit.json.

Sửa nhãn experiment kế thừa trong top-levelconfig E162→E163; model/checkpoint/metrics không thay đổi. Viewerbest38000 đã tạo, có8patchcolors; browsercheck lưu trong best_viewer_browser_check.json.

## Độ phẳng từng patch (best38000, IID512queries)

Mỗi16-pointpatch fit bestplane bằng centeredcovariance eigenvectors: medianorthogonalRMS0.1582mm,p950.2136mm; medianprincipalRMSaxes0.1582/8.2901/9.1318mm. Không patch nào có second-eigenfraction<1%, nên làmiếng2D chứ không linecollapse. Toàn128-pointcloud bestplaneRMSmedian9.708mm: cácpatch không đồngphẳng toàncloud. Kết quả đođộphẳng patch, không chứngminh patch khớp đúng mặtGTcube. Rawstats patch_planarity.json.
