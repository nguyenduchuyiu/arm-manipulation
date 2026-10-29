# E177 — Kết quả cuối: DINOv3 thay SAM3 trong E175

**Hoàn thành 30.000 bước và toàn bộ test.** Bestcheckpoint chọn bằng canonicalvalidation ở **11.000bước**. Training mất **36.72phút**, không tính extraction/probe và evaluation. ChỉDINOv3frozen,871858tham sốpointencoder/decoder/head train từđầu. BF16scalar/transformertraining;FP32vectorgeometry/XYZreadout/EMD/masterweights/optimizer và evaluation.

## Data và kiến trúc

CùngE157:1024trainscene/128val/5×128heldout,4màu/taskID,RGB320×240/full4096XYZrobotbase→128targetvisiblepoints. Camera/robot/ghế/thảm/bàn/nền nhưE175. Cubeđược resize vật lý rồi renderRGB-D/GT nhấtquán trongdataset;khôngthêmSE3/XYZscaleaugmentation khi train. GTmask/center/pose/size/objecttokenskhông làm grounderinput.

GiữE175:2localscalar/vectorblocks,width128/8vectorchannels,UV16neighborgraph;2TransformerDecoderlayers/128queries/4heads;coordinatehead đọcXYZ+vectorV. Chỉvisualinputwidth256→1024. Frozenfacebook/dinov3-vitl16-pretrain-lvd1689m,normalizedpatchtokens**block6**,bỏCLS+4registers. RGBresize1152square→72×72grid đểsampleontoXYZ.

Block6chọn bằng balancedlinearridge5classprobe trên64train/32valscenes: foregroundmIoU72.54%,block24=70.09%,meanlast4=71.45%. Đây làproxyđọc màu/foreground,**không chứng minh layer giàu geometry nhất** hay end-to-end reconstruction tốt nhất. Khôngdùngtestđểchọnlayer;probeweightskhôngđưa vào model.

TrainB8scenes×4tasks,EMDonlyunsquaredHungarian,AdamW3e-4→3e-5cosine30k,warm100,wd.01,clip1,seed170. BestvalEMD4.60597mm ở11k. Last30kval4.739mm/IID4.863mm,khôngtốthơnbest11kIID4.656mm. Peakallocated11.22GiB,phầnlớn tăng do1024Dfeaturecache;train/valcache~9GiB.

## Canonical/OOD cuboid — best

| Test | DINO EMDmm ↓ | SAM3 EMDmm ↓ | DINO coverage5% ↑ | SAM3 coverage5% ↑ |
|---|---:|---:|---:|---:|
| iid | 4.656 | 4.419 | 89.70 | 91.75 |
| positions | 5.544 | 4.962 | 84.86 | 89.00 |
| rotations | 7.124 | 6.320 | 62.00 | 69.81 |
| sizes | 6.158 | 5.624 | 82.45 | 84.27 |
| combinations | 4.912 | 4.597 | 88.00 | 91.14 |
| SE3 | 4.656 | 4.418 | 89.93 | 91.87 |

IIDcentroiderror **2.259mm**,alignedCD2.888mm,forcedtargetselection100%. RobustextentMAEXYZ2.488/1.994/2.195mm,RgMAE0.949mm,RgR²0.9372.

Đổi hệtọađộ toànscene: pointwiseSE3driftRMS **0.001135mm**,max0.011244mm,RGB/DINO/UVassociationgiữnguyên. Đây không phải camera-view equivariance. DINOgeometryIID/OODnhìnchung kémnhẹSAM3 trongcấuhìnhnày.

## Camera vật lý — cùng raw observation fixtures với E175

| View | DINO EMDmm | SAM3 EMDmm | DINO coverage5% | SAM3 coverage5% |
|---|---:|---:|---:|---:|
| original | 4.620 | 4.422 | 89.86 | 91.49 |
| yaw_m15 | 5.090 | 4.796 | 86.05 | 89.78 |
| yaw_p15 | 4.939 | 4.851 | 86.30 | 88.40 |
| yaw_m30 | 5.506 | 10.550 | 80.05 | 74.38 |
| yaw_p30 | 5.488 | 19.802 | 79.90 | 68.01 |
| up15 | 6.363 | 5.084 | 72.12 | 84.19 |

Mỗi view128scenes×4tasks. MọiobjectđềucóvisibleGT vàinputpoints. RGB-D/calibration/samplinggiốngE175;freshDINOfeatures/UVgraph,khôngadaptmodel. DINOforcedselection100%ởmọiview;chỉsốnàykhôngphảingưỡnggeometryaccuracy.

Điểm mạnh rõnhất là yaw±30°: blueEMD−30/+30giảm từ26.291/63.549mm xuống**5.710/5.361mm**. DINO khôngcóqueryEMD>20mm ởhai viewnày (SAM3có73/105query trên512). Up15kémhơnSAM3, nênkhôngkếtluậnrobustmọicamera.

## Shape target chưa có trong train

| Shape | DINO EMDmm | SAM3 EMDmm | DINO coverage5% | SAM3 coverage5% |
|---|---:|---:|---:|---:|
| sphere | 6.143 | 5.948 | 82.46 | 80.97 |
| ellipsoid | 5.274 | 5.248 | 84.20 | 81.73 |
| cylinder | 4.293 | 4.220 | 90.33 | 90.22 |
| capsule | 9.743 | 8.427 | 79.04 | 79.82 |

Toànbộqueryđược giữ, bao gồm8sphere/2ellipsoid/18capsulequeryzero-target-inputpoint;densitysubsetsởood/summary.json. Hìnhmớinằmtrongcuboidboundscũ,khácvolume/visibility,densitykhôngmatched. DINO khôngcảithiệnEMDshape,tuycoveragevàextentmộtsốshape tốt hơn. Chưa testnonconvex/hollow/mixedfamily.

## Feature correspondence

| Condition | EMDmm | Coverage5% | Selection% |
|---|---:|---:|---:|
| baseline | 4.656 | 89.70 | 100.00 |
| other_index | 368.689 | 0.04 | 22.27 |
| other_uv | 183.828 | 0.91 | 31.84 |
| own_shuffled | 356.810 | 0.02 | 23.05 |

Indexswapảnhkhác,nearestUVswapảnhkhác vàshuffletrongsceneđềufailmạnh. Cầnđúngfeature–XYZassociation;testnàykhôngtáchriêngsemantic/geometry/pose hoặc chứngminhkhôngcònshortcut.

## So sánh các block DINOv3 — E177 và E178

**Trong hai lượt đã train, block 24 nhỉnh hơn trên cuboid IID và các split rotation/size, còn block 6 generalize tốt hơn trên cả bốn họ shape mới.** Camera không có một block thắng mọi góc: block 6 tốt hơn ở yaw ±30°, block 24 tốt hơn ở elevation +15°. Vì vậy giữ block 6 làm lựa chọn ưu tiên khi cần robust trên các shape mới và yaw lớn trong bộ test hiện tại.

### Hidden được lấy ở đâu?

Cùng frozen DINOv3 ViT-L/16 gồm 24 Transformer blocks. Hidden sau block 6 và block 24 đều có 1024 channels và cùng vị trí token trên grid 72×72; số block chỉ độ sâu xử lý, không thay đổi độ phân giải. Mỗi hidden đi qua cùng pretrained final LayerNorm (FP32), bỏ CLS + 4 register tokens, rồi bilinear sample vào đúng 4.096 XYZ. Block 24 đã normalize được kiểm tra khớp chính xác official last_hidden_state. Không dùng CLS/global pooling, không concatenate nhiều block và không fine-tune DINO.

### Linear probe và train end-to-end là hai đánh giá khác nhau

Probe E177 dùng 64 train / 32 validation scenes, balanced linear ridge 5 lớp, đánh giá foreground 4-color mIoU. GT masks chỉ cung cấp nhãn cho probe, không là input của DINO hay grounder; probe weights không được đưa vào model reconstruction.

| Feature | Probe foreground mIoU (%) ↑ | Reconstruction 30k đã train? |
|---|---:|---|
| block6 | 72.54 | Có — E177 |
| block12 | 71.10 | Chưa |
| block18 | 70.68 | Chưa |
| block24 | 70.09 | Có — E178 |
| mean_last4 | 71.45 | Chưa |

Block 6 được chọn ban đầu bằng probe này; block 24 được user chỉ định cho lượt đối chứng. Probe đo khả năng đọc tuyến tính màu/foreground, không đo trực tiếp metric shape. Block 12, block 18 và mean-last-4 mới có probe scores, chưa có kết quả reconstruction để xếp hạng chúng. mean-last-4 là trung bình hidden đã normalize của blocks 21–24.

### Cấu hình đối chiếu

Chỉ đổi feature block 6 → block 24. Giữ nguyên weights/revision DINO, LayerNorm, RGB resize 1.152×1.152, grid 72×72, feature width 1.024, E157 data và UV graph; cùng 871.858 tham số trainable, E175 scalar/vector encoder + grounded decoder + equivariant coordinate head. Tất cả module trainable khởi tạo lại từ đầu, không resume/load trained weights E177 sang E178.

Cả hai: 30.000 steps, B8 scenes × 4 tasks, model/data seed 170, AdamW 3e-4 → 3e-5 cosine, warmup 100, weight decay .01, clip 1; một unsquared Hungarian EMD loss. BF16 scalar MLP/Transformer, FP32 vector geometry/XYZ/loss/optimizer/eval. Không thêm SE(3) hay XYZ-scale augmentation; equivariance nằm trong kiến trúc.

| Run | Hidden | Train (phút) | Best step theo validation | Val EMD (mm) ↓ |
|---|---|---:|---:|---:|
| E177 | block 6 | 36.72 | 11,000 | 4.606 |
| E178 | block 24 | 35.85 | 12,000 | 4.445 |

Best được chọn độc lập bằng canonical validation mỗi 1.000 steps. Tất cả bảng bên dưới dùng best đó; không chọn checkpoint bằng IID/OOD. Mỗi physical OOD condition giữ đúng 128 scenes × 4 tasks, cùng raw RGB-D/GT observations; trích feature mới cho mỗi camera/shape và giữ mọi query, kể cả object không có input point.

### Reconstruction và độ phủ

EMD là unsquared Euclidean distance, mm (↓). Coverage là tỷ lệ visible-GT voxel 4 mm có predicted point trong 5 mm (↑).

| Test | Block 6 EMD | Block 24 EMD | Block 6 coverage (%) | Block 24 coverage (%) |
|---|---:|---:|---:|---:|
| IID | 4.656 | 4.574 | 89.70 | 90.36 |
| Unseen positions | 5.544 | 5.562 | 84.86 | 86.12 |
| Unseen physical rotations | 7.124 | 6.735 | 62.00 | 64.91 |
| Unseen sizes | 6.158 | 5.858 | 82.45 | 82.60 |
| Color–position combinations | 4.912 | 4.728 | 88.00 | 89.98 |
| shapes/sphere | 6.143 | 9.045 | 82.46 | 70.57 |
| shapes/ellipsoid | 5.274 | 7.252 | 84.20 | 66.39 |
| shapes/cylinder | 4.293 | 9.173 | 90.33 | 65.41 |
| shapes/capsule | 9.743 | 15.092 | 79.04 | 59.72 |
| camera/original | 4.620 | 4.516 | 89.86 | 90.52 |
| camera/yaw_m15 | 5.090 | 5.029 | 86.05 | 86.92 |
| camera/yaw_p15 | 4.939 | 5.158 | 86.30 | 84.66 |
| camera/yaw_m30 | 5.506 | 6.024 | 80.05 | 75.88 |
| camera/yaw_p30 | 5.488 | 6.145 | 79.90 | 74.57 |
| camera/up15 | 6.363 | 5.483 | 72.12 | 78.13 |

IID cải thiện nhỏ: EMD 4.656 → 4.574 mm (1.77%), centroid-aligned CD 2.888 → 2.834 mm và coverage 89.70 → 90.36%. Centroid error gần như giữ nguyên (2.259 → 2.264 mm), nên không kết luận block 24 cải thiện translation. Rg R² tăng 0.9372 → 0.9412.

Ngược lại, trên shape mới, block 24 vừa tăng EMD vừa giảm coverage ở cả sphere, ellipsoid, cylinder và capsule; riêng cylinder coverage giảm 90.33% → 65.41%. Đây là mất chất lượng reconstruction/coverage, không chỉ khác vài điểm cực trị. Target selection của hai model vẫn cao; forced-choice chọn đúng object không đồng nghĩa tái tạo đúng shape.

Đổi hệ tọa độ toàn scene vẫn giữ equivariance: pointwise RMS drift block 6 0.001135 mm và block 24 0.001020 mm, cùng 4 fixed global SO(3)+translation views trên toàn IID, FP32 inference. Test này giữ RGB/feature–XYZ/UV graph correspondence và khác với đổi góc camera vật lý.

### Diễn giải và giới hạn

Kết quả không ủng hộ quy tắc “block cuối luôn tốt nhất” cho bài toán này. Block 24 cải thiện nhẹ reconstruction của cuboid nhưng chưa đem lại robustness tổng thể tốt hơn. Không từ đây suy ra block 6 chỉ giữ geometry hoặc block 24 chỉ giữ semantic; muốn xác định cơ chế cần phép đo/ablation riêng.

Đây là hai lượt dùng cùng một seed, chưa có nhiều seed hoặc kiểm định ý nghĩa thống kê cho chênh lệch IID nhỏ. Shape fixtures cũng thay đổi volume/visibility/point density; giữ tất cả query và chưa coi đây là phép thử chỉ thay shape. So sánh hai block là có kiểm soát; so DINO với SAM3 còn khác feature width, resize, precision và LR horizon, nên không quy mọi khác biệt cho backbone.

Với pipeline hiện tại, ưu tiên **DINO block 6** nếu cần cân bằng shape mới và camera yaw lớn. **Block 24** phù hợp hơn nếu ưu tiên các split cuboid IID/rotation/size đã đo. **SAM3** vẫn có EMD tốt hơn trên camera gốc và các họ shape mới đã test, trong khi DINO block 6 tránh được lỗi lớn của SAM3 ở yaw ±30°; đây là trade-off theo các test hiện có, không bảo đảm mọi góc camera.

Nguồn đối chiếu: E177 commit `02b6078df115c69d2a46f02bffe9b8731150e8a0`; E178 commit `ed39d076aa330262ed0df6a91ce2ec5c60262886`. [Báo cáo E178](e178_dino24.md), [toàn bộ metrics E178](e178_results.md); raw comparison snapshots/hash ở `/workspace/robotics_assets/experiments/e178_dino24/evaluation/block6_comparison.json`. E177 audit 34.816 records; E178 kiểm tra hash/finiteness 34.816 records và tính lại độc lập EMD cho 5.632 best-IID/physical-OOD queries; không coi đây là tính lại độc lập toàn bộ coverage.

## Kiểm chứng và giới hạn

Independentpostaudit **passed34816queryrecords**:canonical21504,SE36144,physicalOOD5120,featurecontrols2048. RecomputeexactHungarianEMD,directionalCD,centroid vàrobustextent từrawclouds;GTidentityđốichiếudatasetgốc;source/checkpoint/DINO/cache/config hashes và toàn bộpointwiseSE3driftchecked. Coverageaggregatescheckedagainststoredrecords,khôngclaimindependentfullvoxelcoverage recomputation. Ba3DviewersHTTP200/WebGLrenderchecked,no pageerrors.

Giữtiêu chícheckpointcanonicalval,tấtcảbảngchínhdùngbest11k. Layerselectproxychỉmàu; đã train đối chứng block 24 trong E178 (mục so sánh ở trên), còn block 12/18/mean-last-4 chưa train reconstruction. Backbone,width,resolution,BF16precision,LRhorizon thayđổisoE175;khôngcausalablationthuầnbộvision. Oneseed,idealdepth/fixedclutter,visiblepartialGT,no realworld/hiddencompletionclaim.

Source train/test `02b6078df115c69d2a46f02bffe9b8731150e8a0`;audit `a2e77d2719a43352616956aa148f73a7161a469a`. Rawdata/code/checkpoint/calibration details in configs và [allresults](e177_results.md). Temporary test feature caches đã xóa; train/val block6 cache cũng đã được xóa sau audit để dành chỗ cho E178, giữ config/SHA/source/weights để tái tạo. E176GVP vẫn làrunstopped,khôngsofulltest chưa chạy.

[Best vs GT](http://localhost:17070/e177.html) · [Camera 3D](http://localhost:17070/e177_camera.html) · [Shape 3D](http://localhost:17070/e177_shapes.html) · [Audit](/workspace/robotics_assets/experiments/e177_dino/test_full/audit.json).
