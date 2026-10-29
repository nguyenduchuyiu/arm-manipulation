> Cleanup update: user requested Markdown-only retention for historical experiments after E122. The code/artifact links below are historical. Current runnable source is unified_geometry_semantic; current E178 data/weights/features and moved dependencies are listed in robotics_assets/experiments/e178_dino24/current_assets.json.

# Point-cloud code giữ lại sau E134

Đã lọc theo yêu cầu: giữ pipeline dùng được và test quan trọng; nhánh không đạt hoặc thử lặp chỉ giữ báo cáo/kết quả. E134 trở về trước không thay đổi.

## Sáu bộ chính

| Bộ | Dùng để làm gì | Code |
|---|---|---|
| e135_grounding | SAM3 + custom point encoder; RGBD data generator, baseline, noise controls | [code/README.md](/workspace/robotics_assets/experiments/e135_grounding/code/README.md) |
| e137_sam3_se3 | Custom encoder + full SE(3), equivariance và feature–XYZ controls | [code/README.md](/workspace/robotics_assets/experiments/e137_sam3_se3/code/README.md) |
| e149_emd | PTv3 full-cloud + SAM3 + exact EMD; overfit hai mẫu | [code/README.md](/workspace/robotics_assets/experiments/e149_emd/code/README.md) |
| e157_sim_emd | Tạo data physical resize; RGB/depth/GT/SAM3 nhất quán, loaders và FPS | [code/README.md](/workspace/robotics_assets/experiments/e157_sim_emd/code/README.md) |
| e158_40k | Custom encoder + SAM3 + EMD, SE(3), best40k | [code/README.md](/workspace/robotics_assets/experiments/e158_40k/code/README.md) |
| e161_pair | Custom encoder + SAM3 + EMD+pair, physical resize gallery/correspondence | [code/README.md](/workspace/robotics_assets/experiments/e161_pair/code/README.md) |

“Dùng được” ở đây không có nghĩa shape grounding đã giải quyết hoàn toàn: SAM3/custom chọn đúng object nhưng shape còn hạn chế. E149 overfit GT128 tốt (~0.10mm EMD) trên hai mẫu, chưa chứng minh full-surface completeness hay grounding4task tổng quát.

## Test giữ lại

| Bộ | Câu hỏi diagnostic | Code |
|---|---|---|
| e138_target_motion | target-only translation counterfactual | [code/README.md](/workspace/robotics_assets/experiments/e138_target_motion/code/README.md) |
| e139_target_shape | target-only shape counterfactual | [code/README.md](/workspace/robotics_assets/experiments/e139_target_shape/code/README.md) |
| e159_size_probe | frozen latent size probe | [code/README.md](/workspace/robotics_assets/experiments/e159_size_probe/code/README.md) |
| e160_rg_scale | probe-guided size diagnostic (negative result retained as a test) | [code/README.md](/workspace/robotics_assets/experiments/e160_rg_scale/code/README.md) |

Ngoài ra giữ unit tests của model/data/SE(3), exact EMD+pair matching/gradient, directional Chamfer, robust extent, surface coverage; FPS ở [sampling.py](/workspace/robotics_assets/experiments/e157_sim_emd/code/e157/sampling.py). Test4tổ hợp feature nhỏ/lớn × XYZ nhỏ/lớn có correspondence đúng ở [correspondence.py](/workspace/robotics_assets/experiments/e161_pair/code/e161/correspondence.py), dùng fixture E143 và không phụ thuộc checkpoint E142 đã bỏ. `compare(model)` trả kết quả, không ghi đè artifact cũ.

## Các lỗi không lặp lại

1. **SE(3) vẫn giữ.** Full HaarSO(3)+translation±0.5m đổi hệ tọa độ scene/GT cùng nhau; features giữ đúng point. Đây không phải thay shape/size. Giữ E137 và tests equivariance/counterfactual.
2. **Đổi size thật phải render lại.** RGB/depth/GT/SAM3 cần nhất quán. Không dùng coordinate scaling với feature ảnh cũ để kết luận model không dùng geometry. Pipeline data chuẩn là E157.
3. **Selection100% chưa phải shape đúng.** Đo feature swap/zeroXYZ, target-only motion/shape, và4tổ hợp correspondence. Nền cố định có thể tạo shortcut global frame.
4. **EMD khớp GT128 chưa phải phủ đủ bề mặt.** E149 cube80mm EMD~0.10mm nhưng visible-surface coverage5mm chỉ55.06%. Giữ directional distances, robustQ95−Q5, voxel coverage và GT128 reference.
5. **Global attention tốn N².** Tăng input8k/16k có thể rất chậm và không tự tăng coverage target; đo số điểm thực rơi vào từng object. PTv3 nhiều điểm cũng có thể không hội tụ trong budget đã thử.
6. **Latent có size không có nghĩa XYZ head dùng tốt.** E159 linearprobe IIDSE3RgR²~0.888; E160 sửa scalarRg nhưng EMD/coverage không cải thiện. Giữ probe/rescale làm diagnostic, không coi là giải pháp shape cuối.
7. **Patch phẳng không tự ghép thành object.** Folding/patch-SA chưa cải thiện tổng thể18k;40k chủ yếu cải thiện vị trí, patches vẫn không khớp mặt cube. Không coi thêm attention là bảo đảm hết overlap.
8. **Đừng bỏ augmentation vì một model fail.** PTv3-SE3 E154/E155 dừng sớm; SE(3) có kết quả tốt ở E137. Một seed, fixedcamera và visibleGT là giới hạn chung.

## Nhánh bỏ: chỉ giữ kết quả và bài học

| Bộ | Lý do/bài học | Báo cáo |
|---|---|---|
| e136_concerto | Concerto kém SAM3 và rất nhạy outlier/depth trong cấu hình đã thử. | [e136](e136_README.md) |
| e140_rsp | Coordinate scaling với feature ảnh cũ không kiểm tra đúng physical shape; giữ SE(3), render lại khi đổi size. | [e140](e140_README.md) |
| e141_simscale | Pipeline physical-size cũ, được bản data nhất quán E157 thay thế. | [e141](e141_README.md) |
| e142_overfit2 | Overfit custom encoder20/80mm không đạt shape criterion; tăng số bước không đủ chứng minh shape tốt. | [e142](e142_README.md) |
| e143_correspondence | Bỏ runner phụ thuộc checkpoint E142; giữ balanced128 fixture và test correspondence mới ở E161. | [e143](e143_README.md) |
| e144_dense2 | Tăng dense input/global attention chậm và không giải quyết shape; không giữ run thử dở. | [e144](e144_README.md) |
| e145_dense_infer | Dense inference là diagnostic cũ, giữ bảng/ảnh; không cần một implementation riêng. | [e145](e145_README.md) |
| e147_ptv3 | PTv3 FP32 initial1k chưa overfit chính xác; giữ implementation PTv3 dùng được ở E149. | [e147](e147_README.md) |
| e148_bf16 | Chamfer dễ tụ điểm, thiếu coverage/Z; giữ comparison và metrics để đối chiếu EMD. | [e148](e148_README.md) |
| e150_rsp | PTv3 full-scene RSP tốn bộ nhớ/thời gian, nhánh được dừng/thay thế. | [e150](e150_README.md) |
| e151_fps | Preview FPS; hàm FPS đã giữ gọn trong E157/sampling.py. | [e151](e151_README.md) |
| e152_near_fps | Preview radius+FPS; giữ kết quả mật độ, hàm FPS chung ở E157. | [e152](e152_README.md) |
| e153_fps_rsp | PTv3+RSP18k không học grounding tốt: IIDEMD116.34mm, selection27.73%. | [e153](e153_README.md) |
| e154_se3 | PTv3+SE3 EMD dừng5300; không coi run chưa đủ budget là chứng minh SE(3) sai. | [e154](e154_README.md) |
| e155_se3_cd | PTv3+SE3 Chamfer dừng950; không coi run ngắn là kết luận về augmentation. | [e155](e155_README.md) |
| e156_rsp_emd | RSP coordinate-scale đạt selection cao nhưng target-only shape response gần0; dùng data physical resize nhất quán. | [e156](e156_README.md) |
| e162_surface | Các patch phẳng nhưng chưa ghép đúng bề mặt cube; không có cải thiện reconstruction tổng thể18k. | [e162](e162_README.md) |
| e163_surface40k | 40k cải thiện vị trí/EMD nhưng shape và arrangement vẫn sai; không giữ làm pipeline shape thành công. | [e163](e163_README.md) |
| e164_patchattn | Thêm8patchSA chưa cải thiện reconstruction tổng thể18k; attention không tự bảo đảm topology/không giao nhau. | [e164](e164_README.md) |

Reports/configs, metricJSON/JSONL, hình/HTML và resultNPZ được giữ. Code, checkpoint và input/feature cache riêng của các nhánh bỏ đã xóa; không giữ lại bản backup code của các nhánh đó. Pretrained/shared assets ngoài phạm vi experiment output giữ nguyên.

## Data và checkpoint được bảo vệ

- E135 RGBD gốc; E157 RGBD/GT/SAM3 feature cache; best/resume của các bộ giữ lại.
- E146 `data.npz`/`features.npy` là fixture full74k input cần cho E149, dù kiến trúc local E146 đã hủy. Không còn code/train model E146.
- E143 `balanced128.npz` là fixture/result cho test correspondence, không phải input crop của training.
- E159 linearprobe heads cần cho E160 được giữ.
- Cache SAM3 train/val của E135/E137 đã thiếu từ đợt dọn trước: dùng code features E135 để tái tạo khi retrain. E157 cache hiện tại vẫn đủ. Môi trường `.venv-concerto` hiện chứa dependencies SAM3/PTv3 dùng chung; tên venv là lịch sử.

## Kiểm tra cleanup

Giữ10 code bundles (6core+4test), bỏ19 code bundles. Import/provenance,14unit tests, EMD+pair và SE(3) pass. Model/checkpoint E135/E137/E158/E161 cho prediction giống từng bit; PTv3 E149 giữ nguyên AST các hàm/model và trong lượt kiểm tra cuối cũng khớp từng bit (có tolerance0.05mm cho nondeterminism GPU). RGB/depth/labels/point sampling/targets E157 khớp từng bit. 792 file kết quả/checkpoint giữ lại đã đối chiếu SHA256 trước/sau xóa. Giải phóng 7.53GiB. Manifest chi tiết ở `_maintenance/code_cleanup_20261027/curation_deletion.json`.

Chạy mã giữ lại qua `code/run.py --help`. Code riêng của thí nghiệm nằm trong chính `code/` của nó; helper chỉ giữ các hàm/lớp cần thiết, không import code từ thư mục thí nghiệm khác. Hash runtime và source gốc phục vụ provenance được ghi riêng trong `migration.json` và snapshot đã lọc. Source thay đổi mới phải ghi hash thật, không giả lập hash cũ.

Final check sau xóa: 154 runtime modules của10bộ giữ import/provenance pass; code/checkpoint của19bộ bỏ và backup chưa lọc không còn. SE(3)/EMD/FPS/correspondence và các dependency dùng chung đều có mặt.7viewer của cả nhánh giữ/bỏ còn trảHTTP200. Xem `_maintenance/code_cleanup_20261027/final_verification.json`.


FM particle-index constraint (E168): keep shared XYZ/time projections, no learned fixed particle-index embedding or index positional encoding. Grounding learned queries belong to conditionZ and must not be added to FMparticles by index. Preserve particle-permutation/condition-order/EMDpath/20step sampler tests under E168/test_perm. FP32 equivariance passed; BF16 finite-precision endpoint sensitivity measured separately. Positive centroid/Rg heads do not imply learned shape: E168FM did not beat its sphere prior on IID.
