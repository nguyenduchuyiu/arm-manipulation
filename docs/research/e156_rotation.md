# E156 — rotation-only audit

Trạng thái: hoàn thành rotation-only audit; không train thêm.

Checkpoint cố định `e156_rsp_emd/train/best.pt`, step17500; model encoder gốc878.979params train18k với EMD/fullRSP, batch8scene/32task, LR3e-4→3e-5, frozen SAM3. Không train hoặc fine-tune thêm; cùng source weights/camera/data/features như E156.

Test trên128heldout IID scenes×4task,4096inputpoints/scene,128outputpoints/task. **Rotation-only: P′=P@R.T, G′=G@R.T, s=1,t=0**, quay quanh robot-base origin, RGB/frozen SAM3 feature/task và feature-point correspondence giữ nguyên. Không rerender vật thật/camera, không thêm sensor noise/action horizon/flow steps. Ba rotations X/Y/Z90° và4 bộ Haar SO(3) lấy nguyên R của E140/E156 SE3 heldout archive; tất cả translation đặt0. Tổng3584rotated queries mỗi precision. Seed setup135; transforms cố định trong cases.npz.

Chạy BF16 như E156 và **FP32/TF32off control** cho cả canonical/rotated inference để tách lỗi precision. Same checkpoint, same points/features/transforms, no architecture edits. Identity control kiểm tra no-op. Toàn bộ GT/model predictions giữ tọa độ raw khi chấm EMD/CD/selection/center equivariance.

Metrics chính: EMD/CD tới rotatedGT, accuracy nearestGT, raw equivariance CD so f(F,RP) với Rf(F,P), centroid equivariance error. Riêng diagnostic hình dạng: trừ **empirical mean của chính mỗi prediction** rồi so shape mới với rotated canonical shape và unrotated canonical shape. No rotation/scale alignment, không dùng GT centroid để giúp model. Control giữ shape đứng yên nhưng đưa tâm đúng vị trí cần quay; đo paired improvement so control và bootstrapCI95 theo scene trên4randomSO3sets. Diagnostic này phân biệt quay vị trí với quay shape, không sửa prediction hoặc thay metric chính. Không quy Chamfer thành angle error/orientation accuracy vì cube có symmetry và visible sampled clouds không phải amodal shape.

Artifacts dưới `e156_rsp_emd/test_rotation`, không ghi đè evaluation/training E156. Lưu exact config/checkpoint/source hashes, per-case arrays/summary, PNG và báo cáo. Code `OATFlow/rsp_emd/rotation.py`, run `.venv-concerto/bin/python -u -m OATFlow.rsp_emd.rotation`; window riêng attached `ssh_tmux`, tee logs, remain-on-exit on. Canonical note tại docs/research/e156_rotation.md.

## Kết quả

4randomSO3sets ×128IIDscenes ×4tasks =2048queries mỗi precision. Ba fixed90° rotations thêm1536queries/precision.

| Precision | EMD tới GT(mm) | Raw equivariance CD(mm) | Center rotation error(mm) | Shape CD tới rotated reference(mm) | Static-shape control(mm) | Shape gần rotated reference hơn |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 10.737 | 4.228 | 7.107 | 2.827 | 2.978 | 8.06% |
| FP32 | 10.720 | 4.177 | 7.079 | 2.711 | 2.906 | 6.30% |

BF16: selection100.00%; shape error reduction vs static control5.06%; paired improvement0.151mm, scene-bootstrapCI95[0.141,0.160].

FP32: selection100.00%; shape error reduction vs static control6.69%; paired improvement0.194mm, scene-bootstrapCI95[0.184,0.206].

| Axis90° (FP32) | Raw EqCD(mm) | Center error(mm) | Shape rotated CD(mm) | Static control(mm) |
|---|---:|---:|---:|---:|
| X90 | 4.005 | 6.620 | 2.669 | 2.862 |
| Y90 | 3.826 | 6.153 | 2.742 | 2.766 |
| Z90 | 3.874 | 6.131 | 2.745 | 2.965 |

Coordinate-frame rotation of the whole scene, not a physical rerender or target-only rotation. Shape CD is a translation-removed equivariance diagnostic; cube symmetry prevents interpreting it as angular pose accuracy.

Artifacts summary.json, config.json, cases.npz, BF16.npz, FP32.npz, metrics.png và example.png. Identity control pass; BF16canonical predictions bằng chính archive E156 ban đầu. No checkpoint/source changes.

## Diễn giải và audit

**Có bằng chứng rõ model đưa vị trí prediction đi theo rotation của toàn scene; chưa có bằng chứng shape equivariance tốt.** FP32 center error7,079mm so control không quay tâm462,550mm. RawEqCD4,177mm, selection100%. ShapeCD tới rotatedcanonical2,711mm so staticshapecontrol2,906mm: chỉ giảm6,69%, dù pairedCI95 dương[0,184;0,206]mm. ShapeCD tới unrotatedcanonical1,890mm; chỉ6,30%query có centeredprediction gần rotatedreference hơn unrotatedreference. Đây không phải orientationaccuracy/angleerror.

BF16 cùng chiều: center7,107mm; shapeCD2,827 vscontrol2,978mm, giảm5,06%. Vì vậy không thấy lỗi chỉ do BF16. Shape diagnostic dùng cùng point budget và bỏ translation bằng ownmeans; sampling distribution hoặc prediction gần đối xứng vẫn có thể ảnh hưởng, không suy ra một cơ chế nội bộ duy nhất hoặc amodal cube pose. Kết luận giới hạn ở saved predictedpointsets và whole-scene coordinate rotations. Chưa test physical rotation hoặc target-only rotation ở đây.

Verification.json pass: checkpoint/test/data source hashes khớp, identitycontrol exact, RᵀR=I/detR=1, scale1/translation0, recompute8percase CPUfloat64metrics maxerror0 và toàn bộ randomSO3summarymeans từarchives. Không train thêm, không đổi checkpoint E156. Viewer PNG HTTP200: [minh họa](http://localhost:17070/rsp-emd/test_rotation/example.png), [metric chart](http://localhost:17070/rsp-emd/test_rotation/metrics.png). Log riêng `ssh_tmux:3:rot156-log`, đã hoàn thành.
