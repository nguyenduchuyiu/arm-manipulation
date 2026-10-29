# E160 — inference-only Rg correction bằng frozen linear probe

> **Code giữ lại:** [code/README.md](/workspace/robotics_assets/experiments/e160_rg_scale/code/README.md). test: probe-guided size diagnostic (negative result retained as a test). [Danh mục/bài học](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




Trạng thái: hoàn thành inference-only rescale diagnostic.

Freeze E158best36500 và **đúng E159 SE3_target_mean head903params đãfit**, không train/refit bất kỳmodule/hyperparameter nào. Samehead dùng cho canonical vàSE3; không chọn canonical-specializedhead để tối ưutest. Model4096full-scene inputpoints/128outputpoints/task,4task, SAM3frozenfreshfeaturesE157; RGB/camera/robot/ghế/thảm như cũ.

## Inference transform

Đọc mean128queryhidden sauTransformerDecoder/trướcXYZMLP, dùng fixedstandardization/headweights từE159 lấy r_probe(mm), đổi sangm. Chỉ đưa predictedXYZ vàr_probe vào hàm:

`p′ = mean(p) + r_probe/(Rg(p)+epsilon) * (p-mean(p))`

epsilon=1e-9m cố định, không scale clipping/tuning. ProbeRg đã kiểmtra positive trênval/IID; không dùngGTsize đểclamp/hiệu chỉnh. Giữ nguyên pointcount/order, predictedcentroid vànormalizedrelativegeometry/orientation. Không GTcentering, alignment, reorientation hoặc resampling. GTsize/GTpose/GTmask chỉ dùng labels/chấmmetric, không thamgiarescaling.

## Evaluation

Exact E159 val128 vàIID128scenes,4tasks: canonical +4fixedSE3views/scenes. SameSE3probe dùng xuyênsuốt. Val512/2048queries vàIID512/2048queries (canonical/SE3). Originalmodel/probestates frozen. Taps được kiểmtra bit-identical E159cache; rawXYZstats phải khớp cacheoriginal. No newRGB/SAM3encoding needed; reuse E157data/features vàE159R/t.

Chấm raw/scaled paired: exact equal-mass EMD128↔128, directionalCD riêng tớiGT128/fullvisibleGT, equal-region surfacecoveragevoxel4mm tại2/5mm vớiGT128reference, RgMAE/R²,3physicalobject-localextentMAE/R²,covarianceeigenR² vànearestGTselection. Rg/eigenGT từfullvisibledepthsurface nhưE159; localdims từactualsimgeometry, GTorientation chỉ địnhnghĩaaxes khi chấm localextents. RgR² saucorrect phải gầnprobeR² theo côngthức, tựnó chưa chứngminhgeometry tốt hơn. BáoEMDdelta/fractionimproved vàCI95bootstraptheoscene (views/tasks cùngscene cluster).

Headalpha đã chọnval trongE159, nênval làdiagnosticdevelopment; IID làindependenttest. Không dùngIID chọnhead/epsilon/scalegain. Không oracleGTsize scaling. Pointdecoder vàbackbone không đượcfitlại, không thêmloss. Đây làdiagnostic, không giải pháp kiếntrúc cuối.

Geometrychecks: maxcentroiddrift<1e-12m, samecenterednormalizedcoordinates, resultingRg≈r_probe tớiepsilon tolerance; model/probeSHA/statekhông đổi. Raw/rescaledclouds, percasemetrics, exactconfigs/hashes vàexample/scientificplots lưu ở`/workspace/robotics_assets/experiments/e160_rg_scale`; code`OATFlow/probe_rescale/`. Chạy `.venv-concerto/bin/python -u -m OATFlow.probe_rescale.run`, attachedtmuxwindow riêng vàtee log. Noepochs/actionhorizon/flowsteps/noiseaugmentation.

## Kết quả

Same fixedSE3_target_mean probe cho cảcanonical/SE3. Không fit thêm.

| Split/protocol | Stage | EMD(mm) | Pred→visibleGT(mm) | VisibleGT→pred(mm) | Coverage5mm | Rg R² | Local size R² |
|---|---|---:|---:|---:|---:|---:|---:|
| val/canonical | raw | 10.494 | 6.777 | 6.277 | 55.17% | -0.367 | 0.151 |
| val/canonical | rescaled | 10.496 | 7.617 | 5.416 | 53.70% | 0.878 | 0.165 |
| iid/canonical | raw | 10.671 | 6.851 | 6.427 | 54.36% | -0.454 | 0.133 |
| iid/canonical | rescaled | 10.663 | 7.795 | 5.528 | 52.90% | 0.889 | 0.182 |
| val/SE3 | raw | 10.864 | 7.055 | 6.277 | 53.81% | -0.247 | 0.117 |
| val/SE3 | rescaled | 10.867 | 7.766 | 5.505 | 52.15% | 0.881 | 0.090 |
| iid/SE3 | raw | 11.033 | 7.095 | 6.434 | 52.79% | -0.306 | 0.122 |
| iid/SE3 | rescaled | 11.040 | 7.890 | 5.646 | 50.92% | 0.888 | 0.106 |

PairedEMD improvement=raw−rescaled; CI95bootstrap theo128sceneclusters, không coi4tasks/4views độc lập:

- val/canonical: -0.002mm, CI[-0.067,+0.065], 41.60%queries improve. Scale factor{'min': 0.564450827669902, 'median': 1.2141934244135275, 'max': 1.7273051257087437, 'p05': 0.7853173598496477, 'p95': 1.570210907778619}.
- iid/canonical: +0.008mm, CI[-0.069,+0.087], 42.38%queries improve. Scale factor{'min': 0.5095642497228675, 'median': 1.247402704361222, 'max': 1.795905581737721, 'p05': 0.7897290609082491, 'p95': 1.5835813942881825}.
- val/SE3: -0.003mm, CI[-0.066,+0.059], 42.92%queries improve. Scale factor{'min': 0.5304750917198829, 'median': 1.18920720314445, 'max': 1.8273630094112028, 'p05': 0.7801979239374573, 'p95': 1.550811015145837}.
- iid/SE3: -0.007mm, CI[-0.081,+0.069], 42.92%queries improve. Scale factor{'min': 0.5117960555747509, 'median': 1.209851880474039, 'max': 1.7788418678491935, 'p05': 0.7904969041044049, 'p95': 1.557030309242646}.

Inference-only diagnostic. Uniform scaling preserves relative shape/orientation and predicted center; it cannot fix them. Rg R² follows the probe by construction. Val was used to select the already-fitted probe alpha; IID is the independent evaluation.

Exact percasemetrics/raw-rescaledXYZ/GT128/R/t/factors trongval/iid_clouds.npz và*_metrics.npz; GT không tham gia chọnfactor. Geometry_checks.json kiểmtra centroid/normalizedshape invariants vàsamefrozenstates. Diagnostic không thay pointclouddecoder hoặc architecture.

## Diễn giải sau audit

**Đây làtrường hợp RgR² tăng nhưng EMD không cải thiện rõ.** Không cần train thêm hoặc GTsize tạiinference; fixedSE3linearprobe903params vàE158best36500 giữ nguyên. Tâm prediction/normalizedshape đượcgiữ đúng nhưcôngthức.

TrênIID+4SE3views(2048queries): EMD11,03285→11,04028mm; improvement(raw−corrected)−0,00744mm, scene-bootstrapCI95[−0,08078;+0,06877]mm chứa0. Coverage5mm52,790→50,918%; RgR²−0,3063→0,8880, RgMAE4,379→1,278mm. Val+SE3: EMD10,86383→10,86675mm, coverage53,809→52,152%, RgR²−0,2471→0,8809. CanonicalIIDEMD10,67087→10,66295mm: giảm0,00792mm, CI95[−0,06902;+0,08707]mm cũngchứa0. Tất cảprotocolmeanEMDchange nhỏhơn0,01mm. Không gọi chênhlệch nhỏnày làEMDtăng/giảm rõ rệt.

RgR² tăng gần đúngfixedprobeR² theo thiết kế, **không tựchứngminhgeometry đã sửađúng**. Local3extentR² trênIIDSE3 thậmchí0,1223→0,1064; covarianceeigenR²−0,6627→−1,4894. Nativepointcloud có normalizedcovarianceeigenfraction trungbình0,358/0,344/0,298, trongkhiGT0,554/0,317/0,129; uniformscale giữ nguyênfractionsnên không sửa anisotropy. Không suyra orientationerror cụthể từmetric này.

DirectionalvisibleGT distances IIDSE3: pred→GT7,095→7,890mm (xấuhơn), GT→pred6,434→5,646mm (tốthơn). Vì vậy tăng/giảm bánkính có thểđưa thêm điểmGT gầnprediction hơn nhưngđồngthời đẩy nhiều prediction xa surface; không giảiquyết massmatchingEMD/coverage5mm. Meannearest distance và thresholdregioncoverage làmetrics khácnhau.

Kếtluận giới hạn: scalaroverallspread cótronglatent nhưng **một hệ sốscale không đủ cải thiệnpointcloud hiện tại**. Relativegeometry/surface distribution còn sai vàcentroiderror~5,6mm vẫn nguyên; chưa phânđịnh chínhxác đónggóp của shape/orientation/position đếnEMD plateau chỉ bằngtest này. Không phủnhận thôngtin size đã đượcprobe đọcra, không coi diagnostic này làgiảipháp cuối.

Completionaudit pass: allscale factors khớp fixedprobe+formula; centroiddrift≤2,23e-15m, normalizedshapeerror≤1,30e-13, resultingRg lệchprobe≤1,83e-6mm doepsilon. Frozenmodel/probefile hashes khôngđổi; hiddenmeans bit-identicalE159, canonicalIIDrawXYZ bit-identicalE158. Independent12pairedCPUfloat64EMDchecks maxerror7,11e-15mm; summarymeans vàsizeR² recompute từarchives. Tổngruntime85,4s, no newtraining. [Metric chart](http://localhost:17070/probe-scale/results.png), [example raw/rescaled vsGT](http://localhost:17070/probe-scale/example.png).
