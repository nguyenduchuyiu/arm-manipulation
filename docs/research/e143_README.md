# E143 — Bốn tổ hợp SAM3/XYZ, giữ correspondence theo bề mặt vật

> **Chỉ lưu kết quả:** code, checkpoint và cache riêng đã bỏ theo yêu cầu; báo cáo/metrics/hình/result arrays còn giữ. Bài học: Bỏ runner phụ thuộc checkpoint E142; giữ balanced128 fixture và test correspondence mới ở E161. [Danh mục](/workspace/arm-manipulation/docs/research/pointcloud_retained.md).




**Kết quả:** sau khi sửa correspondence, size output vẫn gần như theo feature nguồn trong primary controlled test. Đây là bằng chứng trực tiếp trên hai mẫu/checkpoint này rằng thay đổi metric geometry của target chưa điều khiển tốt kích thước output. Không suy rộng thành không dùng XYZ để định vị, không chứng minh metric information hoàn toàn vắng trong encoder.

Không train lại: đúng hai RGB-D cube20/80mm E142, cùng tâm/pose/task0, checkpoint step17000, exact architecture878.979 trainable params. Head FP32, autocast/TF32 tắt. Frozen SAM3 cùng weights/fpn_hidden_states[-2]; trích lại dense maps và sample đúng vị trí. Native cache extraction max error[0.0, 0.0], tức giống cache E142.

## Sửa correspondence

Probe cùng pixel trước đây có **154 background→cube rows**, nên chưa bảo đảm semantic correspondence. Không dùng probe đó làm bằng chứng chính. Từ E142 có3938 input background rows cùng vị trí3D/instance, nhìn thấy ở cả hai ảnh.

Target được ghép bằng canonical box coordinate q=(p−center)/side, giữ cùng face/vị trí trên bề mặt. Với feature nguồn kích thước s_F, sample dense SAM3 tại Proj(center+s_F·q), rồi ghép với XYZ center+s_G·q. Kiểm tra face hướng camera, simulator label đúng target, depth visibility≤2mm (pipeline depth gốc có residual dưới1mm và pixel rounding). Background phải cùng physical3D surface nhìn thấy ở cả hai ảnh. Không hoán vị token tùy tiện.

Primary **balanced128**:128 target q chung và3968 background rows chung, cùng4096-row permutation cho mọi tổ hợp. Geometry20→80 có exact scale×4 quanh cùng center; background bit-exact; giữ F nhỏ/lớn là **giữ nguyên bit-exact feature tensor** khi đổi XYZ. Target-aware resampling dùng GT chỉ xây dựng can thiệp, không cấp mask/center/size cho forward; baselines được chấm dưới cùng sampling. Không claim đây là native uniform input distribution.

Secondary **native_density**: giữ4/158 target row counts, thay background bị occlude bằng common-visible rows và sửa surface rows không match. Vì số/vị trí hàng target khác nhau, baseline cùng feature source không nhất thiết có cùng feature multiset; đây là kiểm tra độ nhạy mật độ, không tách geometry sạch bằng balanced128. Chi tiết replacements trong `summary.json`.

## Bốn tổ hợp — primary balanced128

| SAM3 feature | XYZ | Pred extent X/Y/Z(mm) | Extent distance tới baseline F(mm) | Tới baseline XYZ(mm) |
|---|---|---:|---:|---:|
| Nhỏ | Nhỏ | 18.148/20.230/21.911 | 0.0000 | 0.0000 |
| Lớn | Lớn | 71.219/73.470/61.568 | 0.0000 | 0.0000 |
| Nhỏ | Lớn | 18.156/20.243/21.918 | 0.0170 | 84.9771 |
| Lớn | Nhỏ | 71.209/73.467/61.567 | 0.0114 | 84.9841 |

Hai cross cases có input XYZ target khác4×, nhưng extent chỉ lệch baseline F khoảng0,01–0,02mm, còn cách baseline XYZ khoảng85mm. Baseline là output cùng construction, không phải GT side length. E142 vốn chưa overfit shape chính xác; kết quả này nói về nguồn điều khiển size, không khẳng định prediction đúng geometry.

### Centroid-aligned Chamfer — primary

Chỉ trừ centroid từng cloud; không rotation/scale alignment.

| F / XYZ | CD tới predicted baseline F(mm) | CD tới predicted baseline XYZ(mm) | CD tới GT F(mm) | CD tới GT XYZ(mm) |
|---|---:|---:|---:|---:|
| 0/0 | 0.0000 | 0.0000 | 3.6081 | 3.6081 |
| 1/1 | 0.0000 | 0.0000 | 11.8959 | 11.8959 |
| 0/1 | 0.0021 | 6.3618 | 3.6084 | 19.8003 |
| 1/0 | 0.0022 | 6.3619 | 11.8956 | 7.6021 |

## Native-density sensitivity

| F / XYZ | Pred extent X/Y/Z(mm) | Distance tới baseline F / XYZ(mm) |
|---|---|---|
| 0/0 | 16.89/17.91/20.42 | 0.00/0.00 |
| 1/1 | 72.01/74.07/61.50 | 0.00/0.00 |
| 0/1 | 18.38/20.59/21.94 | 3.42/85.45 |
| 1/0 | 57.64/61.34/54.86 | 20.32/68.79 |

Ở native density, hai crossed outputs vẫn gần baseline F hơn baseline XYZ, nhưng ảnh hưởng input density/correspondence sampling lớn hơn; không được mô tả geometry effect là0 trong protocol này.

## Encoder và giới hạn

Giữ F nhỏ/lớn rồi đổi XYZ làm encoder memory relative L2 đổi 0.000579/0.000497. H không hoàn toàn bất biến với geometry; norm khác0 không chứng minh encode đầy đủ metric shape. Output extent vẫn gần như không theo XYZ trong primary probe, nên geometry path hiện chưa kiểm soát tốt size, có thể do representation và/hoặc decoder sử dụng thông tin đó yếu. Không định vị nguyên nhân chỉ bằng norm.

Vẫn là counterfactual OOD appearance–geometry conflict. Chính correspondence và matched baselines giúp loại bỏ giải thích random-token mismatch, không biến can thiệp thành real RGB-D observation. Hai mẫu một task/checkpoint chỉ cung cấp bằng chứng giới hạn cho model hiện tại. Masks và simulator center/size chỉ dùng xây dựng test/audit, không vào model. Không retrain/finetune hay sửa architecture.

Verification: canonical scale relation, background physical identity, target instance/depth/front-facing visibility, bit-exact fixed feature tensors, SHA256 checkpoint, CPU float64/SciPy CD/extents. Source và exact input/feature tensors/mapping positions lưu cùng artifacts.

Artifacts `/workspace/robotics_assets/experiments/e143_correspondence`; `summary.json`, `verification.json`, `balanced128.npz`, `native_density.npz`. Code `OATFlow/overfit_cloud/correspondence.py` và `correspondence_report.py`. [Viewer](http://localhost:17070/correspondence/viewer.html), `results.png`.
