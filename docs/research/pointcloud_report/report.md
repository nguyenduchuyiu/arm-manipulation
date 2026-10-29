# Tái tạo bề mặt vật thể theo task từ RGB và point cloud toàn cảnh

*Kết hợp visual backbone đóng băng với encoder–decoder SE(3)-equivariant*

**Báo cáo thực nghiệm · 28/10/2026**

## Tóm tắt

Chúng tôi xây dựng một hệ thống nhận ảnh RGB, point cloud toàn cảnh và task ID, rồi dự đoán 128 điểm 3D của vật được chỉ định trong hệ tọa độ robot. Hệ thống kết hợp đặc trưng thị giác pretrained với point encoder dạng scalar/vector, Grounded Decoder và coordinate head equivariant. Chỉ nhánh 3D và decoder được train, bằng một Earth Mover’s Distance loss; model không cần segmentation mask, tâm, kích thước hay pose của target khi inference. Trên dữ liệu mô phỏng bốn vật khác màu trong cảnh có robot, ghế và thảm, cấu hình SAM3 đạt EMD **4,42 mm**, centroid error **2,05 mm** và visible-surface coverage **91,75%** trên IID. Cấu hình DINOv3 block 6 đạt EMD **4,66 mm**, đồng thời giảm lỗi ở camera yaw +30° từ **19,80 xuống 5,49 mm** so với SAM3. Cả hai giữ tính equivariant khi đổi hệ tọa độ, với pointwise RMS drift khoảng **0,001 mm**, dù không dùng SE(3) augmentation. Đối chứng DINO block 24 cải thiện nhẹ cuboid reconstruction nhưng kém hơn block 6 trên các họ shape mới. Các kết quả xác lập khả năng grounding và tái tạo bề mặt nhìn thấy; chưa chứng minh complete-shape reconstruction hay robustness ngoài simulator.

## 1. Bài toán và mục tiêu

Trong thao tác robot, biết vật nào là target chưa đủ: hệ thống cần geometry của vật trong hệ tọa độ metric để phục vụ bước xử lý tiếp theo. Bài toán ở đây là ánh xạ một observation toàn cảnh và task ID sang point cloud của target:

$$
\hat{G}=f_\theta(I,P,q),\qquad P\in\mathbb{R}^{4096\times3},\quad\hat{G}\in\mathbb{R}^{128\times3}.
$$

I là ảnh RGB; P chứa XYZ của cả cảnh trong robot base frame; q chỉ định một trong bốn màu: đỏ, xanh dương, xanh lá hoặc vàng. GT G là 128 điểm lấy từ **phần bề mặt target nhìn thấy qua camera**. Đây là reconstruction có điều kiện theo một tập task đóng, chưa phải grounding bằng ngôn ngữ tự do.

Báo cáo tập trung vào ba kết quả: (i) gắn visual features với metric points để chọn và tái tạo target mà không crop bằng GT; (ii) ràng buộc SE(3) ở cả encoder lẫn coordinate head; (iii) đo riêng generalization về vị trí, pose, kích thước, shape và camera. Phương pháp cuối dùng decoder trực tiếp; không có flow matching, diffusion hoặc bước tích phân lặp.

## 2. Dữ liệu và supervision

### 2.1. Scene và quá trình tạo dữ liệu

Scene gồm bốn cuboid khác màu cùng robot, ghế, thảm, bàn và sàn. Camera gốc cố định, RGB/depth 320×240. Vị trí XYZ, orientation và kích thước từng vật được lấy mẫu độc lập, sau đó kiểm tra khoảng cách, xuyên mặt bàn, va chạm và độ nhìn thấy. Scene được render tĩnh; robot giữ cùng trạng thái gốc.

Để tạo biến thiên shape/size nhất quán giữa ảnh và geometry, kích thước vật được thay đổi **trong simulator**, độc lập trên ba trục với hệ số Uniform(0,6; 1,4). RGB, depth và GT masks được render lại sau thay đổi. Depth lý tưởng được back-project sang robot base frame, rồi lấy đều 4.096 điểm từ toàn cảnh mà không dùng mask để chọn input. Vì vậy robot và background cũng có mặt trong point cloud.

GT instance masks chỉ được dùng để trích nhãn target và tính metric. Mỗi scene tạo bốn query, dùng cùng RGB/XYZ nhưng khác task ID và target cloud. GT128 được lấy đều từ các điểm target nhìn thấy, có replacement nếu số điểm nhìn thấy không đủ. **GT masks, target center, size, pose và object tokens không là model input.**

### 2.2. Phân chia và các dạng generalization

| Split | Scene | Query | Mục đích |
|---|---:|---:|---|
| Train | 1.024 | 4.096 | Học grounding và reconstruction |
| Validation | 128 | 512 | Chọn checkpoint |
| IID | 128 | 512 | Scene mới cùng phân phối |
| Positions | 128 | 512 | Vị trí ngoài miền train |
| Rotations | 128 | 512 | Orientation vật ngoài miền train |
| Sizes | 128 | 512 | Phân phối kích thước khác train |
| Combinations | 128 | 512 | Tổ hợp màu–vị trí được giữ lại |

Train có roll/pitch trong ±10°, yaw trong ±20°. Split rotations có trị tuyệt đối roll/pitch 20–35° và yaw 30–45°. Trong split combinations, mỗi màu xuất hiện ở một quadrant không đi cùng màu đó trong train, trong khi các quadrant vẫn được thấy với những màu khác.

Base side của train là 24–50 mm; split sizes dùng 14–22 hoặc 54–68 mm trước anisotropic resize. Miền chiều dài sau resize có chồng lấn, nên sizes là một distribution shift có phần extrapolation, không phải mọi vật đều có extent ngoài toàn bộ miền train.

Các test shape/camera dùng 128 scene IID làm nền, thay đổi điều kiện rồi render lại. Mỗi condition có 512 query; chúng chia sẻ scene nền, không phải các bộ scene độc lập hoàn toàn.

## 3. Phương pháp

![Kiến trúc hệ thống](figures/architecture.png)

*Hình 1. Visual backbone được đóng băng. Point encoder, task embedding, Grounded Decoder và coordinate head được train. Scalar memory phục vụ grounding; vector memory và XYZ giữ thông tin metric cho đầu ra 3D.*

### 3.1. Gắn visual features với point cloud

Mỗi XYZ được project về ảnh bằng camera calibration, rồi bilinear-sample từ dense feature map. Calibration phục vụ preprocessing; encoder/decoder không nhận GT object pose hoặc camera extrinsics làm conditioning.

Hai visual backbone được đánh giá:

- **SAM3:** frozen vision/FPN features 256 chiều trên grid 72×72, trước prompt và mask heads; RGB được processor resize lên 1008×1008.
- **DINOv3 ViT-L/16:** normalized patch features 1.024 chiều trên grid 72×72, với RGB resize 1152×1152. Bỏ CLS và bốn register tokens. Cấu hình chính lấy hidden sau block 6; block 24 là đối chứng, qua cùng pretrained final LayerNorm.

Không dùng text prompt hay segmentation output của backbone. Feature vẫn gắn với đúng point trong mọi phép đổi hệ tọa độ.

### 3.2. Point encoder dạng scalar/vector

Encoder gồm hai local message-passing blocks. Mỗi point có scalar feature h gồm 128 channels và tám vector channels V trong không gian 3D. Graph nối 16 hàng xóm gần nhất trong **pixel UV**. Relative XYZ, khoảng cách, vector norms và scalar gates đưa thông tin metric vào các cập nhật.

Scalar features được xây dựng từ các đại lượng bất biến với rotation/translation. Vector features được trộn theo channel bằng cùng trọng số cho ba trục, kết hợp scalar gates và relative vectors. Khi toàn scene đổi hệ tọa độ, h giữ nguyên còn V quay theo scene. Mean dùng để center XYZ là mean của **toàn scene**, không phải GT centroid của target; scale cố định 0,1 m được dùng chung cho ba trục.

Graph UV cung cấp một cấu trúc lân cận ảnh, không phải semantic grouping được giám sát riêng. Trong phép đổi hệ tọa độ 3D, graph giữ nguyên cùng observation. Khi camera vật lý thay đổi, graph được tính lại. Nhánh point encoder tránh full self-attention trên toàn bộ 4.096 points.

### 3.3. Grounded Decoder

128 learned scalar queries được cộng với task embedding, rồi đi qua hai TransformerDecoder layers, width 128, bốn attention heads, FFN width 384. Queries self-attend với nhau và cross-attend tới scalar memory h, tạo target-conditioned latent Z có shape 128×128.

Z không trực tiếp được đưa qua một MLP sinh XYZ theo các trục cố định. Thay vào đó, Z điều khiển việc đọc metric geometry từ scene.

### 3.4. Equivariant coordinate head

Với mỗi output point k, latent Z tạo attention weights a trên input points và scalar coefficients b cho tám vector channels. Head kết hợp weighted XYZ với một vector offset:

$$
\hat{p}_k=\bar{p}+\sum_i a_{ki}(p_i-\bar{p})+s_0\sum_{c=1}^{8}b_{kc}\sum_i a_{ki}V_{ic},\qquad s_0=0.1\,\mathrm{m}.
$$

a là softmax theo input points; b dùng tanh. Vì a và b phụ thuộc vào scalar memory bất biến, còn XYZ/V biến đổi đúng theo scene, output thỏa:

$$
f_\theta(RP+t,F,q,\mathcal{N})=R\,f_\theta(P,F,q,\mathcal{N})+t,\qquad R\in SO(3).
$$

F là visual features gắn với points; N là UV graph. Tính chất này áp dụng cho **cùng observation được biểu diễn trong hệ tọa độ mới**. Nó không tự bảo đảm robustness khi xoay riêng vật, thay shape hoặc đổi camera, vì các thao tác đó tạo RGB, visible points và graph mới.

## 4. Thiết lập thí nghiệm

### 4.1. Loss, tối ưu và chọn checkpoint

Chỉ dùng một loss: exact one-to-one Hungarian matching tối thiểu hóa trung bình khoảng cách Euclidean không bình phương giữa 128 predicted và 128 GT points:

$$
\mathcal{L}_{\mathrm{EMD}}=\min_{\pi\in S_{128}}\frac{1}{128}\sum_{k=1}^{128}\left\|\hat{p}_k-g_{\pi(k)}\right\|_2.
$$

Loss được tính trong mét, kết quả báo cáo trong mm. Tất cả point encoder, task/query embeddings, decoder và coordinate head được khởi tạo từ đầu; chỉ pretrained visual backbone frozen. Không dùng auxiliary center/size loss, pairwise loss hoặc GT size correction ở inference.

Batch gồm tám scene × bốn task. Các lượt dùng seed model/data 170, AdamW, cosine LR 3×10⁻⁴ → 3×10⁻⁵, warmup 100 steps, weight decay 0,01, gradient clip 1. **Không thêm SE(3) hoặc XYZ-scaling augmentation** trong các lượt cuối này; dataset vẫn chứa vật có pose/size đa dạng được render thật.

| Cấu hình | Feature / chiều | Updates | Best step | Trainable | Train (phút) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAM3 | SAM3 FPN / 256 | 50 000 | 12 000 | 772 018 | 119.36 |
| DINO-B6 | DINO block 6 / 1024 | 30 000 | 11 000 | 871 858 | 36.72 |
| DINO-B24 | DINO block 24 / 1024 | 30 000 | 12 000 | 871 858 | 35.85 |

*Bảng 1. Trainable parameters không bao gồm frozen backbone. Thời gian trên RTX 3090, chỉ tính training sau khi cache visual features; không phải end-to-end inference latency.*

SAM3 train nhánh 3D bằng FP32. Hai cấu hình DINO dùng BF16 cho scalar MLP/Transformer, giữ FP32 cho vector geometry, XYZ readout, loss, master weights/optimizer và evaluation. Backbone extraction dùng BF16; sampled features cache FP16.

Mỗi 1.000 steps đánh giá canonical validation, chọn checkpoint có EMD thấp nhất. Mỗi cấu hình dùng **một checkpoint cố định đó** cho toàn bộ bảng test. SAM3 best ở 12k/50k; DINO-B6 best ở 11k/30k; DINO-B24 best ở 12k/30k. Train loss tiếp tục giảm sau best nhưng validation không cải thiện tương ứng.

So sánh DINO-B6/B24 chỉ đổi hidden layer, giữ nguyên kiến trúc, input dimensions, normalization, data, precision, seeds và training schedule. So SAM3/DINO còn khác feature width, image resize, precision và LR horizon; đây là đối chiếu các cấu hình hoàn chỉnh, chưa là ablation chỉ thay backbone.

### 4.2. Metric và protocol

Các bảng dùng dấu chấm thập phân; đơn vị được ghi ở từng bảng. **EMD** được tính trên raw predicted XYZ, không căn tâm, xoay hay scale bằng GT. **Centroid error** là khoảng cách giữa mean của hai clouds. **Directional Chamfer** đo riêng pred→GT và GT→pred bằng nearest-neighbor Euclidean distance. **Centroid-aligned CD** chỉ trừ centroid riêng của hai clouds, giữ nguyên scale và orientation, rồi lấy trung bình hai chiều.

**Robust extent** dùng Q95−Q05 trên mỗi trục; báo cáo MAE theo query. **Radius of gyration (Rg)** là RMS distance tới centroid, đo độ phân tán tổng thể; báo cáo MAE và R². Các đại lượng này mô tả visible surface, không phải complete object dimensions.

**Coverage @5 mm** chia toàn bộ visible GT surface thành voxel 4 mm, chọn một GT point thực làm representative mỗi voxel, rồi đo tỷ lệ representatives có predicted point trong 5 mm. Coverage không chỉ được tính trên GT128. **Target selection** là predicted centroid gần đúng target nhất trong bốn GT centroids; đây là forced four-way choice, không có ngưỡng sai số tuyệt đối.

## 5. Kết quả

![Kết quả trên cuboid, shape mới và camera mới](figures/results.png)

*Hình 2. Mean EMD trên toàn bộ query, dùng validation-best checkpoint. Ba cấu hình có trade-off khác nhau giữa cuboid reconstruction, shape generalization và camera robustness.*

### 5.1. Grounding và tái tạo cuboid

| Test | SAM3 | DINO-B6 | DINO-B24 |
| --- | ---: | ---: | ---: |
| IID | 4.42 | 4.66 | 4.57 |
| Vị trí mới | 4.96 | 5.54 | 5.56 |
| Orientation vật mới | 6.32 | 7.12 | 6.73 |
| Kích thước khác train | 5.62 | 6.16 | 5.86 |
| Tổ hợp màu–vị trí mới | 4.60 | 4.91 | 4.73 |

*Bảng 2. EMD (mm, thấp hơn tốt hơn); mỗi split 128 scenes × 4 tasks.*

SAM3 có EMD thấp nhất trong các split cuboid đã đánh giá. DINO-B24 cải thiện so với B6 ở IID, physical rotations, sizes và combinations, nhưng không cải thiện positions. Cả ba đạt 100% forced target selection trên IID; do đó cần xem thêm sai số geometry và coverage.

| Cấu hình | EMD ↓ | Centroid ↓ | Aligned CD ↓ | Coverage ↑ | Rg R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAM3 | 4.42 | 2.05 | 2.78 | 91.75 | 0.954 |
| DINO-B6 | 4.66 | 2.26 | 2.89 | 89.70 | 0.937 |
| DINO-B24 | 4.57 | 2.26 | 2.83 | 90.36 | 0.941 |

*Bảng 3. IID geometry. Các sai số khoảng cách có đơn vị mm; coverage là % tại 5 mm.*

| Cấu hình | CD pred→GT | CD GT→pred | Extent MAE X/Y/Z | Rg MAE |
| --- | ---: | ---: | ---: | ---: |
| SAM3 | 2.67 | 2.56 | 2.26 / 1.79 / 1.94 | 0.77 |
| DINO-B6 | 2.72 | 2.70 | 2.49 / 1.99 / 2.19 | 0.95 |
| DINO-B24 | 2.67 | 2.63 | 2.80 / 2.00 / 2.02 | 0.89 |

*Bảng 4. Tách coverage/shape khỏi vị trí trên IID. Directional CD dùng GT128; robust extent là Q95−Q05. Tất cả số đo khoảng cách là mm.*

SAM3 đạt centroid error 2,05 mm và coverage 91,75%. Tham chiếu GT128 được sampling từ cùng visible surface có coverage khoảng 91,96%; đây là mức tham khảo của sampling, không phải upper bound lý thuyết. Physical rotations là nhóm khó nhất: model vẫn chọn đúng object nhưng thiếu coverage. Chọn đúng target vì vậy chưa đồng nghĩa tái tạo tốt toàn bộ surface.

### 5.2. Đổi hệ tọa độ SE(3)

Toàn bộ 128 IID scenes được kiểm tra với bốn global transforms, full SO(3) và translation. Cùng transform áp dụng cho scene/GT; RGB, feature–point correspondence và UV graph giữ nguyên. Có 2.048 queries, tương ứng 262.144 output points cho mỗi model.

| Cấu hình | IID EMD | SE(3) EMD | RMS drift | Max drift |
| --- | ---: | ---: | ---: | ---: |
| SAM3 | 4.4185 | 4.4185 | 0.001178 | 0.005779 |
| DINO-B6 | 4.6557 | 4.6557 | 0.001135 | 0.011244 |
| DINO-B24 | 4.5735 | 4.5735 | 0.001020 | 0.006115 |

*Bảng 5. Tất cả đơn vị mm. Drift so trực tiếp output của transformed input với transformed original prediction, không matching hoặc alignment.*

EMD gần như không đổi, pointwise RMS drift khoảng 0,001 mm, phù hợp với equivariance tới sai số FP32. Kết quả này đạt được mà không train bằng SE(3) augmentation. Voxel coverage theo lưới trục tọa độ có thể đổi nhẹ khi quay; pointwise drift và Euclidean metrics là kiểm tra chính.

### 5.3. Họ shape chưa có trong train

Thay cả bốn cuboid bằng cầu, ellipsoid, trụ hoặc capsule trong simulator; giữ scene nền, màu, tâm, pose và camera. Mỗi condition gồm 128 scene renders và 512 queries. Render lại RGB-D/GT và trích lại features; giữ cùng 4.096 pixel rays. Shapes mới nằm trong local bounding boxes ban đầu, nên thể tích, visible area và mật độ point thay đổi.

| Shape mới | SAM3 | DINO-B6 | DINO-B24 |
| --- | ---: | ---: | ---: |
| Cầu | 5.95 / 80.97 | 6.14 / 82.46 | 9.05 / 70.57 |
| Ellipsoid | 5.25 / 81.73 | 5.27 / 84.20 | 7.25 / 66.39 |
| Trụ | 4.22 / 90.22 | 4.29 / 90.33 | 9.17 / 65.41 |
| Capsule | 8.43 / 79.82 | 9.74 / 79.04 | 15.09 / 59.72 |

*Bảng 6. Mỗi ô là EMD mm / coverage %. Kết quả chính giữ mọi query, gồm các trường hợp không còn target point trong input.*

SAM3 và DINO-B6 chuyển sang trụ khá tốt, với EMD khoảng 4,2–4,3 mm và coverage khoảng 90%. Với cầu, ellipsoid và capsule, lỗi tăng và coverage giảm. DINO-B24 kém hơn B6 ở cả bốn families; cylinder coverage giảm từ 90,33% xuống 65,41%, cho thấy khác biệt không chỉ ở vài điểm cực trị.

Sampling toàn cảnh 4K là một giới hạn quan trọng: median input points/target của cuboid là 22,5, còn cầu/ellipsoid/trụ/capsule là 7/11/14/5. Trong 512 queries mỗi condition, có 8/2/0/18 trường hợp target vẫn thấy trên RGB nhưng không còn point nào trong sampled input. Với SAM3, EMD của cầu/ellipsoid/capsule trên nhóm còn ít nhất một target point là 4,43/4,68/4,80 mm, so với 5,95/5,25/8,43 mm toàn bộ query. Subset này giúp chẩn đoán ảnh hưởng sampling; nó không thay thế bảng chính hoặc chứng minh mọi lỗi đều do thiếu point.

### 5.4. Thay đổi góc camera vật lý

Camera orbit quanh workspace: yaw ±15°, ±30° hoặc tăng elevation 15°, giữ intrinsics/FOV. XYZ vẫn ở robot base frame ban đầu. RGB, depth, sampling và UV graph đều được tạo lại; GT là visible surface từ góc mới. Camera gốc cũng được sampling lại bằng cùng pipeline. Mỗi view có 128 scenes × 4 tasks; mọi target đều nhìn thấy và có input points.

| Góc camera | SAM3 | DINO-B6 | DINO-B24 |
| --- | ---: | ---: | ---: |
| Gốc, lấy mẫu lại | 4.42 / 91.49 | 4.62 / 89.86 | 4.52 / 90.52 |
| Yaw −15° | 4.80 / 89.78 | 5.09 / 86.05 | 5.03 / 86.92 |
| Yaw +15° | 4.85 / 88.40 | 4.94 / 86.30 | 5.16 / 84.66 |
| Yaw −30° | 10.55 / 74.38 | 5.51 / 80.05 | 6.02 / 75.88 |
| Yaw +30° | 19.80 / 68.01 | 5.49 / 79.90 | 6.14 / 74.57 |
| Elevation +15° | 5.08 / 84.19 | 6.36 / 72.12 | 5.48 / 78.13 |

*Bảng 7. Mỗi ô là EMD mm / coverage %. Viewpoint change tạo observation mới, khác với phép đổi hệ tọa độ ở Bảng 5.*

DINO-B6 có ưu thế rõ ở yaw ±30°: EMD 5,51/5,49 mm, so với SAM3 10,55/19,80 mm. Với SAM3, lỗi lớn tập trung ở vật xanh dương, EMD 26,29/63,55 mm; DINO-B6 giảm xuống 5,71/5,36 mm. Tuy vậy, elevation +15° có thứ tự ngược lại: SAM3 tốt hơn DINO-B6. DINO-B24 cải thiện elevation so với B6 nhưng kém hơn ở yaw ±30°. Chưa có một cấu hình thắng mọi góc camera.

![RGB và point clouds predicted so với GT](figures/qualitative.png)

*Hình 3. Ví dụ cố định: scene index 0, task xanh dương, ở ba condition; hai cột point cloud là SAM3 và DINO-B6. Khung vàng đánh dấu target trên RGB; xanh là GT, cam là prediction. Các cloud giữ raw robot-base XYZ, không căn centroid. Mỗi hàng dùng chung giới hạn trục cho hai model. EMD trên hình là của từng query, không thay thế mean của toàn bộ test; vì vậy thứ hạng ở ví dụ có thể khác bảng tổng hợp. DINO-B24 được so sánh trong các bảng định lượng.*

## 6. Đối chứng về feature và correspondence

### 6.1. Chọn hidden DINO

Các hidden sau block 6/12/18/24 có cùng số patch tokens và 1.024 channels. Chỉ số block biểu thị độ sâu xử lý, không phải độ phân giải không gian. Probe ban đầu dùng balanced linear ridge trên 64 train và 32 validation scenes để đọc foreground bốn màu; GT masks chỉ là nhãn supervision của probe. Probe weights không được đưa vào model reconstruction.

| DINO hidden | Foreground mIoU (%) | Đã train reconstruction |
| --- | ---: | ---: |
| block6 | 72.54 | Có |
| block12 | 71.10 | Chưa |
| block18 | 70.68 | Chưa |
| block24 | 70.09 | Có |
| mean_last4 | 71.45 | Chưa |

*Bảng 8. Probe mIoU và tình trạng đối chứng reconstruction. mean_last4 là trung bình normalized blocks 21–24.*

Block 6 được chọn ban đầu từ probe; block 24 được chỉ định cho lượt đối chứng. Đổi B6→B24 giảm IID EMD 4,656→4,574 mm, khoảng 1,77%, nhưng làm xấu cả EMD lẫn coverage trên bốn shape families mới. Probe màu/foreground không thay thế được đánh giá reconstruction và OOD. Chưa train reconstruction với blocks 12/18 hoặc mean-last-4.

Kết quả không ủng hộ quy tắc “hidden càng sâu càng tốt” cho bài toán này. Tuy nhiên, chưa thể gán riêng geometry cho block 6 hay semantic cho block 24; các phép thử hiện tại chưa xác định cơ chế nhân quả đó.

### 6.2. Model có cần feature gắn đúng với XYZ?

Với DINO-B6 frozen, giữ XYZ, task và UV graph của scene nhận; thay visual features theo ba cách. Test trên đủ 128 IID scenes × 4 tasks, donor scene ghép cặp hai chiều, với cùng protocol cho mọi query.

| Feature cấp cho scene nhận | EMD (mm) | Coverage (%) | Selection (%) |
| --- | ---: | ---: | ---: |
| Đúng scene, đúng point | 4.66 | 89.70 | 100.00 |
| Scene khác, ghép theo array index | 368.69 | 0.04 | 22.27 |
| Scene khác, ghép theo UV gần nhất | 183.83 | 0.91 | 31.84 |
| Hoán vị giữa point trong cùng scene | 356.81 | 0.02 | 23.05 |

*Bảng 9. Feature corruption diagnostic của DINO-B6. Không update model trong test.*

Tráo theo array index phá correspondence vì point sampling giữa scene độc lập. Ghép theo UV gần nhất kiểm soát cách chuyển feature giữa ảnh, nhưng vẫn gây lỗi lớn vì nội dung scene khác nhau. Hoán vị feature trong cùng scene giữ nguyên tập feature nhưng phá liên kết với point, cũng làm selection giảm gần mức ngẫu nhiên 25%.

Các test cho thấy model phụ thuộc mạnh vào **feature–XYZ association**, không chỉ toàn bộ tập features. Đây là input mâu thuẫn ngoài phân phối; failure không đủ để khẳng định semantic/shape/pose đã được disentangle hoặc mọi shortcut đã bị loại bỏ.

## 7. Thảo luận và giới hạn

**Kết quả đạt được.** Một kiến trúc gọn, dưới 0,9 triệu tham số trainable, đã tái tạo được target visible surface từ full scene với distractors hiện diện, không cần GT segmentation ở inference. Kiến trúc giữ output nhất quán khi đổi hệ tọa độ và tách được phép kiểm tra này khỏi physical pose/camera generalization. Feature correspondence tests và các metric shape/coverage cung cấp bằng chứng bổ sung ngoài target-selection accuracy.

**Lựa chọn cấu hình.** Với camera gốc và geometry trong các split đã đo, SAM3 cho kết quả tốt hơn. Khi cần chịu yaw lớn và vẫn giữ khả năng chuyển sang shape mới tốt hơn B24, DINO-B6 là lựa chọn cân bằng của prototype hiện tại. B24 có lợi thế nhỏ trên cuboid IID/rotation/size nhưng chưa cải thiện robustness tổng thể. Mỗi nhận định gắn với các điều kiện đã test, không bảo đảm một backbone tốt nhất cho mọi deployment.

**Giới hạn quan sát và nhãn.** Model chỉ dự đoán 128 điểm của visible surface. Vật nhỏ có thể gần như biến mất sau 4K sampling; phần bị che khuất không được giám sát. Chưa đánh giá completion của shape kín, shape lõm/rỗng, topology mới hoặc mixed-family scenes.

**Giới hạn robustness.** Robot, ghế, thảm và các distractors có mặt nhưng layout/robot pose phần lớn cố định. Camera OOD có năm góc mới, không phủ mọi viewpoint. Depth là lý tưởng; chưa có kết quả về sensor noise, background/clutter mới hoặc chuyển sang dữ liệu thật. Không gọi sự hiện diện của clutter là bằng chứng robustness với mọi dạng nhiễu.

**Giới hạn đối chiếu.** Mỗi cấu hình chỉ có một training seed; các queries cùng scene tương quan. Chưa báo cáo multi-seed uncertainty hay significance cho chênh lệch IID nhỏ. Test sizes có support overlap; test shape thay đổi cả volume/visibility/density. So SAM3/DINO có nhiều khác biệt cấu hình, trong khi đối chứng DINO B6/B24 được kiểm soát chặt hơn.

## 8. Kết luận

Hệ thống đạt target-conditioned visible-surface reconstruction ở mức EMD khoảng 4,4–4,7 mm trên IID, với trên 89% coverage và coordinate SE(3) equivariance tới sai số FP32. SAM3 là cấu hình có geometry tốt nhất trong các split đã đo; DINO block 6 cải thiện mạnh camera yaw robustness. Thí nghiệm hidden DINO cho thấy lựa chọn layer phải dựa trên downstream reconstruction và OOD, không chỉ độ sâu hoặc probe màu. Các bước tiếp theo cần giải quyết target point sparsity, thêm multi-seed evaluation và kiểm tra depth/clutter/real-world shift trước khi kết luận về khả năng triển khai.

## Phụ lục. Nguồn kết quả và khả năng tái lập

Tài liệu tổng hợp từ các lượt đã hoàn tất, không chạy thêm training/inference hoặc chọn lại checkpoint theo test. IDs dưới đây chỉ phục vụ truy vết; toàn bộ phương pháp và protocol được mô tả trong phần chính.

| Cấu hình | Run / best step | Source commit |
| --- | ---: | ---: |
| SAM3 | E175 / 12,000 | 552a2a353aaae14f3b2e8faa90fb6a20d5781a41 |
| DINO-B6 | E177 / 11,000 | 02b6078df115c69d2a46f02bffe9b8731150e8a0 |
| DINO-B24 | E178 / 12,000 | ed39d076aa330262ed0df6a91ce2ec5c60262886 |

Dataset dùng chung là E157 (RGB-D được render lại sau physical resize). Các bảng đọc trực tiếp từ saved evaluation/OOD summaries; shape/extent bổ sung từ raw IID predictions/GT. Mỗi cấu hình giữ cùng best checkpoint trên mọi split. Các hình định tính dùng scene/task cố định, không chọn theo lỗi tốt nhất.

Nguồn chính trong workspace: các thư mục e175_equiv, e177_dino, e178_dino24 dưới /workspace/robotics_assets/experiments/. Source commits, checkpoint SHA256, full-precision số liệu, đường dẫn/hashes của nguồn và danh sách query minh họa được lưu cùng tài liệu trong evidence.json.
