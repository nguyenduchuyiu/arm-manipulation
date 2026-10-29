# E158 — resume E157 từ18k tới tổng40k

> **Code giữ lại:** [code/README.md](/workspace/robotics_assets/experiments/e158_40k/code/README.md). core: custom encoder + SAM3 + EMD, SE3, 40k checkpoint. [Danh mục/bài học](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành tổng40k và evaluation; best36500.

Tiếp tục **đúng full-state checkpoint E157/train/resume.pt tại18000**, không load best16500 để thay thế resume state, không train lại từ đầu. Khôi phục model, AdamW moments/step counters, data RNG, augmentation CUDA generator, CPU/CUDA global RNG. Output riêng e158_40k, giữ nguyên toàn bộ E15718k và kết quả test cũ.

## Data/model/training giữ nguyên

Reuse immutable E157data/features bằnglinks, không render hoặc trích SAM3 lại cho training. 1024train/128valscenes,4task,4096uniformfull-scene points/128GTpoints mỗi task; camera cam_front320×240, robot/ghế/thảm/clutter như cũ. Physical resize từngcube localXYZ[0,6;1,4] trongsimulator trước render; cached SAM3vision_features từRGB mới, pretrained weights/frozenparameters giữ nguyên. GT masks chỉ supervision/test, không center/size/objecttokens vào model.

Unchanged PointCloudGrounder878.979trainable, encoder/task/fusion/decoder đều tiếp tục train; SAM3pretrained454.038.784 đóng băng. Một exactEMD128×128 unsquaredEuclidean/equalmass optimalassignment như E157. BF16forward encoder/XYZhead nguyên bản, FP32parameters/optimizer/loss, TF32off. Batch8scene/32queries, wd0,01, clip1; original model seed135, continuation RNG đúng state18000.

**40k là tổngsteps: chạy thêm22.000updates.** 0→18k giữ nguyên lịch đã chạy E157(3e-4→3e-5 cosine/warmup100). 18k→40k dùng **LR cố định3e-5**, tiếp tục mức cuối18k, không warmup/restart hoặc nângLR vềcosine40k. Khoảng312,5totaltrain-scene passes; phần chạy thêm171,875passes. Không tuyên bố đây là40kfromscratch với một cosine schedule mới.

Only coordinate SE3 augmentation: fullHaarSO(3)+translationXYZUniform[−0,5;0,5]m, không scaleXYZ. SameE157sampler và RNG137 tiếp tục, F/task/pointassociation giữ nguyên. FixedSE3validationR/t/identitys giữ nguyên E157/E137. Best theo validationEMD mỗi500updates, canonicalval diagnostic. Mang best16500 từE157 làm mốc; best chỉ đổi khi validationEMD cải thiện. Model/optimizer/RNG resume lưu mỗi1000updates; logs mỗi50. Copy tiny train/valhistories để vẽ đủ0→40k, không duplicatefeaturecache.

## Evaluation dự kiến

Best-validation checkpoint sau40k được so với E157best18k(step16500) trên **cùng fresh observations**. HeldoutIID/positions/rotations/sizes/combinations128scenes mỗi split, sameSE3transforms4×128×4task. Physicalscale tests dùng đúng E157protocol/render seeds: fresh baseline scale1, target-only/all-object localaxis×0,6/1,4, cùng sourceposes; rerenderRGB/depth/GT vàSAM3 mỗi physicalcase. Không coordinate scale test làm primary. Case/rejection lists phải khớp E157; không dùng test chọn checkpoint.

Metrics rawcoordinate EMD, directionalCD tớiGT128/fullvisibleGT, robustQ95−Q5XYZ, coveragevoxel4mm/tolerance2/5mm vớiGT128reference, nearestGTselection và pairedphysicalsize extent/robustgains tới fullvisibleGT. Không centroidalignment cho primarymetrics. FreshRGBsuccess chưa tách riêng XYZbranch; một seed vàsyntheticcamera cố định. Noise/action horizon/flow steps không áp dụng.

Audit trước resume xác nhận source/checkpoint hashes, modelweights+optimizerstates khớp, optimizerstep18000, RNGrestored vàsamevalidation. Giữ old18kbest trongfolder mới nếu chưaimprove. Same code tests được copy trongnamespace mới để giữ parentexperiment sourcehash intact. Window riêng attachedssh_tmux, remain-on-exit on, tee stdout/stderr; không đổi window người dùng.

Artifacts `/workspace/robotics_assets/experiments/e158_40k`, code `OATFlow/simscale_40k`; config train/config.json, provenance resume_state_audit.json, source_hashes.json, train.log, train/progress.json. Chạy `.venv-concerto/bin/python -u -m OATFlow.simscale_40k.train`; output mới chỉ tạo một lần. Tại launch chưa có kết quả40k; xem phần hoàn thành bên dưới.

## Launch đã xác nhận

Fullstate audit pass: model/AdamWstates bit-identical parentcheckpoint, tất cảoptimizerstep18000, CPU/CUDARNG vàaugmentationgenerator restoreexact, nextsceneIDs vànextSE3draw khớp; samevalidationarchive. ParentcheckpointSHA965fb0c7d0d7712a01164d235969a14c9dc958a5e5899ed08a13e134c21ff1df, không đổi checkpoint/data/features/source củaE157.

Lượtcontinuation đang chạy trong `ssh_tmux:1:e158-log` (Ctrl+B rồi1), remain-on-exit on, stdout/stderr tee e158_resume.log vàlinkedtrain.log; currentwindow0 giữ nguyên. Snapshot step18550/40000, trainEMD(last50)12.200mm, LR3e-05, peak2.99GiB; phần chạy thêm elapsed41.2s. Chưa có kết quả40k hoặc evaluation cuối.

## Kết quả hoàn thành

Tổngtrain55.99phút (phầnresume26.32phút), eval6.80phút; peakallocated2.99GiB. Best theoSE3validationEMD, checkpoint36500. Full exact configs/hashes trong train/config.json, data_config.json và features/config.json.

| Test | Model | EMD(mm) | Pred→visibleGT(mm) | VisibleGT→pred(mm) | Coverage5mm | GT128 reference5mm | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| iid | E157 | 11.631 | 7.218 | 6.768 | 52.47% | 91.96% | 100.00% |
| iid | E158 | 10.671 | 6.851 | 6.427 | 54.36% | 91.96% | 100.00% |
| SE3 | E157 | 12.798 | 7.976 | 7.248 | 47.69% | 92.13% | 100.00% |
| SE3 | E158 | 11.014 | 7.100 | 6.443 | 52.93% | 92.13% | 100.00% |
| positions | E157 | 13.407 | 7.933 | 7.564 | 47.01% | 90.61% | 100.00% |
| positions | E158 | 11.616 | 7.240 | 6.815 | 51.25% | 90.61% | 100.00% |
| rotations | E157 | 13.191 | 8.091 | 7.018 | 50.19% | 91.20% | 100.00% |
| rotations | E158 | 11.958 | 7.558 | 6.506 | 52.52% | 91.20% | 100.00% |
| sizes | E157 | 16.621 | 10.654 | 9.363 | 51.81% | 87.87% | 100.00% |
| sizes | E158 | 15.039 | 9.793 | 8.878 | 54.97% | 87.87% | 99.61% |
| combinations | E157 | 11.759 | 7.233 | 7.023 | 51.36% | 91.32% | 100.00% |
| combinations | E158 | 10.745 | 6.869 | 6.675 | 53.06% | 91.32% | 100.00% |

SE3 làwhole-scene coordinate transform sau fresh render, không scaleXYZ. Rawmetrics, không centroid alignment. RobustextentXYZ và coverage2mm có trong JSON/archives.

Physicalscale attempted1280, scored1271, excluded9; mọi exclusion được ghi trong cases.jsonl.

| Physical resize | Model | Queries | EMD(mm) | Coverage5mm | Extent gain | Robust extent gain |
|---|---|---:|---:|---:|---:|---:|
| target_only | E157 | 1021 | 11.485 | 52.78% | 0.008 | 0.007 |
| target_only | E158 | 1021 | 10.597 | 54.51% | 0.048 | 0.046 |
| all_objects | E157 | 1000 | 11.466 | 53.08% | 0.008 | 0.007 |
| all_objects | E158 | 1000 | 10.571 | 54.62% | 0.048 | 0.047 |

E157 target-only: visibleGT→prediction improvement so giữ prediction trước resize+0.025mm, scene-bootstrapCI95[-0.043,+0.106].

E158 target-only: visibleGT→prediction improvement so giữ prediction trước resize+0.075mm, scene-bootstrapCI95[+0.020,+0.132].

Extent/robustextent response gains chấm paired deltas tới fullvisibleGT, không dùng amodal size làm modelinput. Fresh views có thể đổi visibility/centroid/input sampling; clearance/visibility rejection làm tập scored có selectionbias. Một seed, syntheticcam cố định. FreshRGB success chưa tách riêng đóng góp XYZ geometry khỏi SAM3 hình ảnh. E157 và E158 dùng cùng data/features/architecture/augmentation/loss; E158 nối tiếp fulloptimizer/RNG từ18000, thêm22000updates LR3e-5.

[Viewer](http://localhost:17070/sim-40k/viewer.html). Checkpoints/data/features cũ giữ nguyên; không còn stale-coordinate-scale test trong primary physicalscale evaluation.

## Diễn giải và completion audit

**Thêm22kupdates có giúp, nhưng chưa giải quyết physical-size tracking.** Fullstate resume đạt40000; optimizerstep cũng40000. Best chọn bằngfixedSE3validation ở36500, EMD10,9597mm so E157best12,9204mm. Không dùng test chọn checkpoint. Cùng heldoutdata/features/cases vàsame18kbaseline đã được xác nhận bit-identical.

| Metric | E15718k(best16500) | E15840k(best36500) |
|---|---:|---:|
| IID EMD(mm) |11,6309|10,6709|
| SE3 EMD(mm) |12,7978|11,0137|
| IID coverage5mm |52,47%|54,36%|
| IID/SE3 nearestGTselection |100%/100%|100%/100%|
| Physical target-only extent gain |0,00796|0,04811|
| Physical target-only robust extent gain |0,00710|0,04625|
| Physical all-object extent gain |0,00755|0,04815|

Physicaltarget-only size response tăng khoảng6lần, nhưng mới~4,8% mức lý tưởng1. PairedvisibleGT→prediction improvement so giữ prediction cũ+0,0749mm, CI95[+0,0203;+0,1324]mm: có cải thiện nhỏ rõ hơn mốc18k, chưa đủ gọi shape grounding tốt. FreshRGB/XYZ/SAM3 nhất quán, không quy failure này cho stale features. Chưa kết luận riêng encoder/fusion/decoder là nguyên nhân.

Riêng tăng40% từnglocalaxis, đo delta robustextent trongbaseXYZ để tránh mean cancellation shrink/expand:

| Local axis | GT ΔXYZ(mm) | E158 prediction ΔXYZ(mm) |
|---|---|---|
| X |12,292/1,171/0,398|0,758/0,832/0,723|
| Y |1,364/13,117/−0,350|0,539/0,326/0,633|
| Z |0,606/−0,004/13,172|0,564/0,507/0,450|

Can thiệp localaxis khácbaseaxis doorientation vật; không quy trực tiếp cácvector thành angleaccuracy. JSON đầy đủ physical_scale/axis_expansion_summary.json. Coverage5mm vẫn dướiGT128reference91,96%. Sizesheldout selection99,61% (2/512queries sai), không khái quát100%selection cho mọi split.

Phầnresume26,32phút, tổngtraining0→40k55,99phút; eval6,80phút. Peak2,99GiB. LR3e-5 cố định trongphầnresume, data/architecture/EMD/SE3 không đổi.

Completionaudit pass: finalresume/AdamWcounters40000, parent18kcheckpointunchanged, source/checkpoint hashesmatch, beststepkhớpvalidation. Baselineglobalpredictions, physicalcasesfile vàtoàn bộphysicalGT/baselinepredictions bit-identical E157. Recompute toàn bộsummarymeans/extentgains;24CPUfloat64globalqueries maxmetricerror9,78e-7mm. Viewer/curve HTTP200. Xemcompletion_audit.json.

## Viewer best vs GT theo yêu cầu

Viewer hiện chỉ có **E158best36500 vàGT128**, bỏ baseline18k/learningcurves/aggregatecomparison khỏi trang xemcloud. Hai màu: vòngxanhGT vàchấmcambest. Mainview là6hìnhchiếuXY/XZ/YZ trước/sauphysicalscale, samebaseXYZ/mm vàsameaxisranges, không căncentroid/rotation/scale; 3Dorthographic mởtheoyêucầu, linkedcamera vànútẩn/hiệnGT/best. Cùngcase1/scene0/taskred/physicalX×1,4 nhưviewertrước, không chọn lạiexample để làm đẹp.

Không inference/train/evaluate lại; cachedexamplepredictions được đối chiếu bit-identical baseline/after archives E158 vàcheckpointSHAkhớpbest36500. Render code bổ sung `OATFlow/simscale_40k/visualize_best.py`; chạy `python -m OATFlow.simscale_40k.visualize_best` để chỉ tái tạo viewer/PNG. Best_viewer.json ghi nguồn vàdisplaylimits. Scientificresults/checkpoints giữ nguyên.

Browsercheckpass: HTTP200, chỉ4tracesGT/besttrước/sau, nútChỉGT hoạt động, 3Dcamera đồng bộ, nopageerrors. Đã xemPNG vàbrowserrender, best_vs_gt.png vàviewer.html publicquaexistingVSCodeforward17070. Verification: best_viewer_browser_check.json.

## Vì sao output giống template bo tròn — diagnosis không train thêm

Checkpointbest36500 được infer trên64train và64valscenes, canonical vàcùng64freshSE3draws(seed158902). CanonicaltrainEMD10,228mm/val10,420mm; SE3train10,635mm/val10,873mm. Đây là lỗi tồn tại ngay trêntrain, không chỉ heldoutgeneralization. RollingtrainEMD38000–40000mean10,977mm vàfixedvalmean11,239mm; loss đãgiảm chậm nhưng không fitshape chính xác.

SavedIID512queries: std robustextent predXYZ1,444/1,506/1,428mm, GT12810,228/11,030/10,231mm. Sau bỏownpredictionmean, querylayout chỉ RMS1,388mm khác per-taskmean template, so typicalRMSradius16,278mm (~8,53%). Đây là diagnostic của output/querylayout, không cănGT để trợgiúp model. Pred–GTextentcorrelation vẫn0,718/0,486/0,570: chưa phải hoàn toàn không dùng thôngtin size, nhưng biên độ đáp ứng bị co mạnh. Rawarrays/seed/protocol trongshape_diagnosis.json; không đổi model/train/eval chính.

Cơ chế khả dĩ, chưa xác định nhân quả: learned128queries với directabsoluteXYZhead dễ dùng một template quanh vị trí điều kiện hóa. Nếu conditioning vào orientation/size không đủ, giảm averageEMD trên nhiều rotations/shapes có thể ưu tiên một shape chung bo tròn. Code không ép sphericalprojection, vàEMD bản thân không bắt buộc shape cầu; đúngGT là nghiệm zeroEMD cho từngsample128điểm. RawEMD vẫn phạtshape sai; plateau chỉ cho thấy optimization chưa giải được conditioning này, không chứng minh geometrybranch không encode shape.

Overfit2mẫu khácbài toán: E142encoder878979/4k/Chamfer/FP32/noaugmentation cóthể phân biệt20/80mm nhưngstrictshapecriterion chưa đạt; E149overfitEMD~0,1mm dùngPTv3~47,7M/full74k/2fixedsamples/noaugmentation. Ngay cả cùng kiếntrúc, nhớ2outputshapes không đòi hỏi học mappingmetricshape across1024scenes/4tasks/continuousSE3. Vì vậy không dùng thành công overfit làm bằng chứng currentconditionalgeometrypath đã học được.

Đã xác nhận underfit/outputtemplate; chưa phân biệt shapeinformation thiếu trongencoder hay đãencode nhưngfusion/decoder không dùng tốt. Không train, tăngaugmentation hoặc sửaarchitecture trong diagnosis này.
