# E175 — Thay đổi góc camera vật lý

Status: complete. Frozen best12k; không train/adapt/chọn lại checkpoint. Đây là test góc nhìn mới với RGB-D mới, khác phép đổi hệ tọa độ toàn scene đã test trước đó.

## Protocol

128sceneIID ×4task, exactcuboidposes/dimensions/colors và robot/ghế/thảm/bàn/nền/lighting giữ nguyên. Cam_front orbit quanh pivot worldcốđịnh(0.54,−0.25,0.08)m: quay vịtrí và basis camera quanhworldZ ±15°/±30°, hoặc nâng elevation15° quanh original camera rightaxis. Radius/FOV/intrinsics giữ nguyên. K vẫn từ camera gốc320×240. Từng camera_to_base được lấy/kiểm tra từ simulator state; XYZ luôn ở robot baseframe cũ. Pivotkhông phải targetGTcenter và không là modelinput.

6condition ×128scene =768render,3072query (2560query góc mới). Baselinechính là originalcamera cũng lấy mẫu lại cùngpipeline, không trực tiếp dùng cachedbaseline4.419mm. Uniform4096points từ validdepth toàncảnh, không mask/crop; seed17564000+scene dùngcùng acrossviews. GT128visiblepoints seed17664000+scene, replacement nếu cần. Thay đổi viewpoint làm đổi visible surfaces/backgroundcomposition/pointdensity nên GT phải là bề mặt nhìn thấy từ từng viewmới.

SAM3frozen tính lại từRGBmới, BF16vision/FP16sampledfeatures, fpn_hidden_states[-2], batch4, khôngprompt/mask. Graph16nearestUV xây lại cho từng newpointset, canonicalties nhưE175. Equivariantencoder/groundeddecoder/coordinatehead frozen FP32/TF32off. KhôngGTpose/center/size/mask làm input. All128scenes/512queries mỗi view đo được: khôngtarget nào bị che hoàn toàn, khôngquery nào thiếu hẳn targetinputpoints.

Parenttraining:772018trainableparams,1024originalcameraE157trainingscenes,50kEMDsteps,B8scenes×4tasks,AdamW3e-4→3e-5,noSE3augmentation;bestcanonicalvalidation ở12k. ChỉSAM3frozen lúctrain.

## Kết quả chính

| View | EMD mm ↓ | Centroid mm ↓ | Aligned CD mm ↓ | Coverage@5mm % ↑ | Forced selection % |
|---|---:|---:|---:|---:|---:|
| Camera gốc (lấy mẫu lại) | 4.422 | 2.047 | 2.798 | 91.49 | 100.00 |
| Yaw −15° | 4.796 | 2.286 | 2.968 | 89.78 | 100.00 |
| Yaw +15° | 4.851 | 2.264 | 3.009 | 88.40 | 100.00 |
| Yaw −30° | 10.550 | 8.151 | 4.577 | 74.38 | 99.41 |
| Yaw +30° | 19.802 | 17.501 | 6.619 | 68.01 | 96.68 |
| Nâng góc nhìn +15° | 5.084 | 2.225 | 3.229 | 84.19 | 100.00 |

Mỗi row512queries. EMD unsquaredHungarian, rawbaseXYZ, khôngGTalignment/correction. AlignedCD chỉtrừ centroidriêng mỗi cloud. Coverage dùngGTvisible4mmvoxelrepresentatives và tolerance5mm. Targetselection là forcednearestcentroid trong4GTobjects; không phải ngưỡng chínhxác vịtrí. Tất cảquery đều thấy đủ4objects trong test này.

| View | Pred→GT mm ↓ | GT→pred mm ↓ | Robust extent MAE X/Y/Z mm ↓ | Rg R² ↑ |
|---|---:|---:|---:|---:|
| Camera gốc (lấy mẫu lại) | 2.659 | 2.569 | 2.41 / 1.77 / 2.00 | 0.950 |
| Yaw −15° | 2.898 | 2.724 | 2.20 / 2.15 / 2.43 | 0.882 |
| Yaw +15° | 2.875 | 2.854 | 2.88 / 2.03 / 2.21 | 0.941 |
| Yaw −30° | 6.421 | 4.143 | 6.40 / 6.09 / 7.99 | -2.274 |
| Yaw +30° | 13.877 | 5.347 | 15.93 / 12.34 / 18.72 | -15.299 |
| Nâng góc nhìn +15° | 3.159 | 3.122 | 2.27 / 2.25 / 3.52 | 0.937 |

Extent =Q95−Q05 của visible128pointGT và prediction trong robotaxes, MAE tínhperqueryrồi trungbình. RgR² nhạy vớioutlier: giátrịâm ở30° cho thấy sizeprediction chưa ổn ở các ca fail, không chỉ lỗi translation.

## Distribution tail và sampling

| View | Median EMD mm | P95 EMD mm | Max EMD mm | Query EMD>20mm /512 | Median targetinputpoints |
|---|---:|---:|---:|---:|---:|
| Camera gốc (lấy mẫu lại) | 4.181 | 6.987 | 11.767 | 0 | 23 |
| Yaw −15° | 4.489 | 7.448 | 43.482 | 1 | 23 |
| Yaw +15° | 4.564 | 7.662 | 16.777 | 0 | 22 |
| Yaw −30° | 5.659 | 38.542 | 130.326 | 73 | 25 |
| Yaw +30° | 5.520 | 95.033 | 239.940 | 105 | 24 |
| Nâng góc nhìn +15° | 4.878 | 7.515 | 11.276 | 0 | 20 |

Không cóqueryzero-target-input ở bất kỳviewnào. Trên matchedsubset mà cảnewviewvàoriginal có≥8targetpoints: yaw−30°478queries EMD10.206vsoriginal4.472mm; yaw+30°479queries19.027vs4.469mm. Vìvậy lỗi30° không chỉ là mất hẳn/thiếu một vài inputpoints. Không suyra nguyênnhân làUVgraph haySAM3 riênglẻ vì chưa ablate từng nhánh.

15°còn tươngđối tốt: EMDtăng khoảng8–10%,coveragegiảm~2–3percentagepoints. Nâng15° EMD5.084mm/coverage84.19%. Yaw30° có nhiềuoutlier: medianvẫn~5.5–5.7mm nhưngP95 lên38.54/95.03mm vàmean10.55/19.80mm. Không phải mọiquery30°đềufail; model thiếu robustnessởtail. Cácviews thay đổi physicalobservation, không phảiRigidtransform cùng mộtP/F/UVgraph; exactSE3equivariance không loại bỏ viewdomainshift.

## PixelUV graph dùng để làm gì

Mỗi pointđược backproject từ mộtpixel nên cóUV. Graphchọn16pointgần nhất theoUV để hai encoderblocks trao đổiSAM3features vàrelativeXYZ, chi phílocal theoN×16. UVchỉchọnconnectivity; geometrymessages vẫn dùngXYZ. Nó không phảiinstancemask/objecttoken. Hai pixelgần nhau có thể thuộc hai bề mặtxa nhau trong3D, nhất làbiênobject. Khi đổi hệtọađộ toànscene, UVgraph khôngđổi; khi đổi camera thật, phải xây lại theoảnh/pointsetmới nhưtest này. Khoảngcách3D cũngbấtbiến vớiSE3; không cókhẳngđịnh UVgraphlà cáchduynhất đểgiữequivariance.

## Verification / reproducibility

Source branchdiag/e175-camera, commit `584340436f02ed6520c5955c934888923ad9f01b`. Ownedcode `/workspace/robotics_assets/experiments/e175_equiv/test_camera/code`, hashes actualsource/config vàcopiedsources.json; khôngimportcross-experiment. CheckpointSHA256 `aebc05fc6e3afff34d28fd283b12a65e1ca97f82666c6fbcfe5d56affd206a52`.

Original4controlscenes RGB/depth/segmentation bitexactE157. HistoricalpointUVgraph và freshSAM3pointfeatures exactcache. Freshprediction maxFP32batchdrift 0.00014975480735301971mm. CameraSO3/extrinsic/orbit/pivotprojection checked everyrender; every inputpointreprojectswith maxerror 1.5432063662501605e-05pixels. Analyticcuboidinteriorchecks pass onepixeldepth/focalfootprint; SAM3weighthash unchanged.

Postrun independentlychecked all3072queries/HungarianEMD androbustextent, all768renderrecords/analyticchecks; results/independent_check.json. Browserverification results/browser_check.json. Allpredictionsfinite; invalid-targetlogic did notexcludeanyquery.

Rawartifacts `/workspace/robotics_assets/experiments/e175_equiv/test_camera/results`: RGB-D/GT/newpoints/newUV/calibration,perviewpredictionNPZ/metricJSON,config/summary,pairedsubsets,render_manifest/checks,RGBstrip andraw3Dviewer. SAM3pointfeatures streamed,no permanentcache/newweights. Pipelinebeforeplotting 213.7seconds, liveattachedtmux ssh_tmux:e175-camera,run.log. Run from testroot `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py` createsfreshresults, do notoverwritecompletedresults.

[Viewer](http://localhost:17070/e175_camera.html): IIDscene0,sixviews,taskdropdown,GTgreen/predictionorange,rawXYZ. Fixedscene0 isillustration, fullscoresall128scenes; othercasesespeciallybluecanhave largererrors. [Staticplot](/workspace/robotics_assets/experiments/e175_equiv/test_camera/results/camera.png).

Limitations: onlycuboids andfivecameraorbits, fixedrobot/assets/lighting/idealdepth; no360°/largeelevation/newintrinsics/movingrobot/realworldtest, no combinedOODshape+camera, singletrainingseed. No causalattribution toSAM3/graph/encoder fromthistestalone.


## Theo từng task: outlier30° tập trung ở xanhdương

| View | Red EMDmm | Blue EMDmm | Green EMDmm | Yellow EMDmm |
|---|---:|---:|---:|---:|
| original | 4.594 | 4.279 | 4.411 | 4.403 |
| yaw_m15 | 4.786 | 4.922 | 4.760 | 4.716 |
| yaw_p15 | 4.962 | 4.828 | 4.989 | 4.626 |
| yaw_m30 | 5.391 | 26.291 | 5.065 | 5.455 |
| yaw_p30 | 5.409 | 63.549 | 5.294 | 4.954 |
| up15 | 5.137 | 5.054 | 5.088 | 5.055 |

TaskIDs0/1/2/3=red/blue/green/yellow. Đây là quan sát per-task, chưa xác định do appearance/context củaSAM3, UVneighborgraph hay modelgrounding. Không có ablation để quy lỗi riêng một thành phần. Ba màu khác vẫn~5mm ở30° trong khi blue26.29/63.55mm.

Presentation-only repair: Plotly dropdown title changed to title.text, rebuilt from saved predictions with code/view.py, sourcecommit `0fd826e9cd2b6a44f51160a00bff41094db857ed`. No new inference or metric changes. Initial browser check caught title placeholder despite correct trace switching; corrected viewer is rechecked in browser_check.json.
