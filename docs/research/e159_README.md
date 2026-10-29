# E159 — frozen hidden size probes

> **Code giữ lại:** [code/README.md](/workspace/robotics_assets/experiments/e159_size_probe/code/README.md). test: frozen latent size probe. [Danh mục/bài học](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành frozen extraction và linear probes.

Freeze toàn bộ checkpoint E158best36500 (`train/best.pt`, SHA a06ca868ef2744e7ed71aafd9d2d712a2d6379b91e52cda2dc6dfa04858b9d97), eval mode, không gradient/optimizer cập nhật model. SAM3features đã đóng băng/cache; không render/encode ảnh lại. Model878979params, fullscene4096points+SAM3feat256+taskID; pretrainedSAM3 weights như E158.

## Hai vị trí tap, tách rõ nghĩa

1. **Memory trước TransformerDecoder**: encoder output `[B,4096,128]`, chưa gắn task. Meanpool `[B,128]`, Linear128→28 xuất7labels cho cả4task rồi chọn theo taskID; 3612headparams. Không GT-guided pooling/crop/objecttoken/mask/center/size vào features.
2. **Hidden đã gắn task trước XYZ MLP**: output frozenTransformerDecoder `[B,4,128queries,128]`. Meanpool queries→`H_target[B,4,128]`, Linear128→7;903headparams. Đây là strict linear probe chính trên task-conditioned hidden. Tap này nằm sau TransformerDecoder/fusion nhưng trước point-coordinate head; không gọi nhầm là trước toàn bộ TransformerDecoder.

Chạy thêm mean+populationstd pooling ở từngtap (256D; memory7196params,target1799params) như readoutcontrol. Stdpool là thống kê phi tuyến cố định, không layer được train. Meanprobe âm tính có thể do pooling triệt tiêu thôngtin; mọiprobe âm tính không chứng minh latent hoàn toàn mấtgeometry. Positiveprobe không xác định thôngtin từXYZ haySAM3.

## Nhãn và splits

Reuse freshphysical E157data:1024train,128val,128IID,128unseensizes,4task/scene. Primarylabels: ba localXYZ dimensions thật của visual/collision cube từmetadata (`dimensions_m`, mm), radiusofgyration(mm), ba covarianceeigenvalues theo thứ tự giảm dần(mm²) từ **full visibleGT surface** dựng bằngdepth/calibration/instanceGT. PopulationcovarianceC=(P−mean)ᵀ(P−mean)/N, Rg=sqrt(traceC). Localbodyaxes vàGTmasks chỉ tạo nhãn/metric; không vào frozenmodel/readoutinputs. Rg/eigenvalues bất biến SO(3)/translation, physicallocaldimensions không đổi khi đổi hệ tọa độ. Body-axis ambiguity/occlusion giới hạn diễn giải localdims; Rg/eigen không cần GTorientation.

Hai protocols độc lập:

- Canonical: fit/train/val/test hidden ởframe gốc.
- SE3: fit4fixedHaarSO(3)+tXYZUniform[−0,5;0,5]m views/scene, same feature-point association, s=1. Train/val/IID/sizes làscene-disjoint, không chia khácviews của cùngscene quafolds. Seed CPU159100+splitindex. Labels giữ nguyên; check rotationinvariance độc lập.

Cache5views(canonical+4SE3) hiddenpool float32, không flatten4096tokens thành head lớn. Dùng BF16frozenforward như E158; pooling/statsFP32. OriginalXYZdecoder predictions được đo localextent (GTorientation dùngchấm, không feedforward), Rg/eigen để đối chiếu. GT128statistics làreference cho sai số sampling, không upperbound. Chỉ nhãn7scalar được dùng supervisedprobe; không pointdecoder training.

## Fitting và đánh giá

Linearhead với intercept, standardizedX/Y theo **train statistics only**; solve ridgeMSE+L2 chính xác bằng eigendecomposition, không chờ optimizer hội tụ. Alpha[1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10] chọn bằng meanstandardizedMSE trênval của cùngprotocol; IID/sizes chỉ finaltest. Feature mean/std, targetmean/std được export cùng nn.Linear headweights; toàn bộreadout affine với meanpool, không MLP. Không clamp negativepredictions.

Metrics train/val/IID/sizes: MAE(mm) vàR² cho3localdims/Rg, MAE(mm²) vàR² cho3eigenvalues, relativeMAE localdims. Không tự quyR² thành angularaccuracy. Baselines: task-conditioned trainingmean, original frozenXYZhead, GT128sampling reference. Scene-wise shuffledtraininglabels (cùngtask/viewstructure) cho target_mean_std kiểm tra control; không dùng IID chọnalpha/tap. Báo đầy đủ4taps, không lựa tap theo test. Primarytarget_mean; meanstd làdiagnostic.

Interpretation: positivegeneralizing targetprobe cho thấy size đọcđược trướcXYZhead, chưa đủ nói TransformerDecoder trước đó hoặcXYZbranch đã grounding tốt. Memoryprobe tốt nhưngpostfusionprobe kém gợi ý khókhăn trongfusion/readout; negativeprobes không chứngminh thôngtin đã mất. Train tốt/valkém làdấu hiệu probe/readout generalization kém; phân biệt với linearreadout không đủ expressivity. OriginalXYZsize kém trongkhi latentprobe tốt mới làevidence không khai thác tốt size trong output head.

Run code `OATFlow/latent_size_probe/`, artifacts `/workspace/robotics_assets/experiments/e159_size_probe`; exact config.json, hiddenNPZs, fittedheadNPZs/linear_heads.pt, predictions, summary.json vàfigures. No epochbudget vì closedform; frozen inference batch8. Không actionhorizon/flowsteps/noiseaugmentation. Window riêng attachedssh_tmux, remain-on-exit on, stdout/stderr tee log. Tất cả E158weights/evaluation/viewer giữ nguyên.

## Kết quả

Primary strictlinearreadout: target_mean903params. Fullmodels không được train thêm.

| Protocol | Tap | Train Rg R² | Val Rg R² | IID Rg R² | IID Rg MAE(mm) | IID local size MAE(mm) | IID local size R² | IID covariance R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | memory_mean | 0.335 | 0.232 | 0.147 | 3.692 | 8.714 | 0.108 | 0.121 |
| canonical | memory_mean_std | 0.625 | 0.524 | 0.462 | 2.886 | 7.639 | 0.310 | 0.391 |
| canonical | target_mean | 0.925 | 0.920 | 0.924 | 1.070 | 3.892 | 0.797 | 0.815 |
| canonical | target_mean_std | 0.935 | 0.920 | 0.923 | 1.072 | 3.762 | 0.809 | 0.812 |
| SE3 | memory_mean | 0.285 | 0.193 | 0.129 | 3.754 | 8.849 | 0.086 | 0.105 |
| SE3 | memory_mean_std | 0.504 | 0.401 | 0.329 | 3.199 | 8.181 | 0.210 | 0.275 |
| SE3 | target_mean | 0.884 | 0.881 | 0.888 | 1.278 | 4.555 | 0.732 | 0.767 |
| SE3 | target_mean_std | 0.896 | 0.888 | 0.894 | 1.246 | 4.385 | 0.752 | 0.775 |

| Protocol | Baseline IID | Rg R² | Rg MAE(mm) | Local size MAE(mm) | Local size R² | Covariance R² |
|---|---|---:|---:|---:|---:|---:|
| canonical | task_mean | 0.001 | 4.021 | 9.351 | -0.001 | -0.000 |
| canonical | original_XYZ_decoder | -0.454 | 4.642 | 8.520 | 0.133 | -0.761 |
| canonical | GT128_reference | 0.989 | 0.387 | 1.196 | 0.894 | 0.959 |
| canonical | shuffled labels | 0.002 | 4.018 | 9.335 | 0.003 | 0.003 |
| SE3 | task_mean | 0.001 | 4.021 | 9.351 | -0.001 | -0.000 |
| SE3 | original_XYZ_decoder | -0.306 | 4.379 | 8.726 | 0.122 | -0.663 |
| SE3 | GT128_reference | 0.989 | 0.387 | 1.196 | 0.894 | 0.959 |
| SE3 | shuffled labels | 0.006 | 4.009 | 9.329 | 0.004 | 0.005 |

Unseen-sizes split chỉ finaltest, không chọnalpha:

| Protocol | Tap | Rg R² | Rg MAE(mm) | Local size R² |
|---|---|---:|---:|---:|
| canonical | memory_mean | 0.295 | 8.435 | 0.262 |
| canonical | memory_mean_std | 0.647 | 5.795 | 0.568 |
| canonical | target_mean | 0.923 | 2.303 | 0.849 |
| canonical | target_mean_std | 0.921 | 2.289 | 0.853 |
| SE3 | memory_mean | 0.263 | 8.667 | 0.231 |
| SE3 | memory_mean_std | 0.548 | 6.646 | 0.475 |
| SE3 | target_mean | 0.901 | 2.606 | 0.821 |
| SE3 | target_mean_std | 0.907 | 2.528 | 0.831 |

Linear decodability is readout-dependent. Negative mean/std probes do not prove all geometry information absent; positive results do not identify whether information originates in SAM3 or XYZ. Memory tap precedes task fusion; target tap follows TransformerDecoder and precedes xyz MLP.

OriginalXYZlocalextents dùngGTorientation chỉ đểchấm, không đưa orientation vào frozenmodel/probe. Meanstdreadouts cófixednonlinearpooling; chỉmeanreadout làstrictlinear so tensorinput. Không suyra fullshape reconstruction từsizeprobe. `summary.json` cómetrics đầy đủ/alpha/headparams, `linear_heads.pt` exporthead+bộstandardization, cácheadNPZ/predictions lưu riêng.

## Diễn giải sau audit

**Thôngtin size đọc được tốt ở target-conditioned hidden ngay trước XYZMLP.** Primarystrictmeanprobe chỉ903params, originalmodel/SAM3 hoàn toànfrozen. SE3 train/val/IID RgR²0,884/0,881/0,888 vàRgMAE1,301/1,303/1,278mm. CanonicalIID RgR²0,924, MAE1,070mm. Không thấy patterntrainfit/valfail trênprimaryprobe. Unseensizes SE3RgR²0,901/MAE2,606mm; rangevariance lớn hơn nên không gọi đây là chính xác hơnIID chỉ dựaR².

SE3IIDlocalXYZdimensionprobe MAE4,555mm, meanR²0,732; covarianceeigenR²0,767. OriginalXYZpointcloud localextentMAE8,726mm/R²0,122; nativeRgMAE4,379mm/R²−0,306. TaskmeanRgR²0,0009 vàshuffledlabelcontrol0,0060, nên cải thiện probe không chỉ làtaskprior hoặc đọclabelngẫu nhiên. Dimsprobe chưa hoàn hảo; size metrics không chứng minh latent đãencode đầy đủshape/orientation/surface.

PreTransformerDecoder memorymeanprobe chỉRgR²0,129 (meanstd0,329), nhưng không suyra geometrymấttrướcdecoder: fullscene pooling chưa task-specific, cònfrozenTransformerDecoder tạo hidden màsize decodeđược. Không GT-guidedpooling ở bất kỳtapnào. Saufusion cònthôngtin size, nên bottleneck nằmở quátrình biến latent thànhgeometryXYZ hiện tại; chưa xácđịnh do headarchitecture, optimization, orientationconditioning hoặc phối hợp cácphần. Positiveprobe không táchSAM3vsXYZ nguồnthôngtin.

### Control bổ sung sau kếtquả chính

Để tránh so một headđãfit với outputgeometry chưa calibration, fitthêmLinear35params từ4rotation-invariant statistics của chínhpointcloudXYZ(Rg+3eigen) sang7GTlabels; không dùngGTpose/center/masks làmfeatures, chọnalpha trênval như cũ. IIDSE3calibratedRgR²0,582/MAE2,425mm, canonical0,694/2,089mm. Vì vậy output vẫn cònsizeinformation trongcácbiếnthiênnhỏ, không biến mất hoàn toàn; nativegeometry bịco response mạnh. Latentprobe tốt hơn calibrationoutput, nhưng không khẳngđịnh tất cảsizeinfo bịmất sauhead. Head này chỉ đọcscalar, không sửapointcloud. Supplementaryposthoccontrol ghi riêng output_stats_control.json, không lựa lạiprimarytap hoặc sửdụngIID đểtune.

### Verification

Completionaudit pass: modelstate trước/sauextract bit-identical, tapforward bit-identical originalforward, originalXYZstats khớp savedE158predictions. Checkpoint/source/cacheSHAkhớp, source-scene fingerprints disjointtrain/val/IID/sizes, labelsRg/eigen rotationinvariant. Recompute toàn bộheadpredictions/metrics từweights/cache, train-onlystandardization xácnhận, L2 chọnvalonly; maxnormal-equation residual6,50e-14, predictionreproductionerror9,67e-13nativeunits. Exportnn.LinearFP32 matchedfloat64fits vớimaxerror0,00142nativeunits. Tổngextract+fit26,6s, không gradient/update vàoE158.

`linear_heads.pt` chứa8heads, keyprimary`SE3_target_mean` hoặc`canonical_target_mean`: standardizednn.Linearstate cùng x_mean/x_std/y_mean/y_std. Inferenceheadinput làmean128queryhidden sauTransformerDecoder, trướcxyz; normalize bằngtrainstats, linear, inversey-standardization. Cácstats làaffinenormalization, không MLP.
