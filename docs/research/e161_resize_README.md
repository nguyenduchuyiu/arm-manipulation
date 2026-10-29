# E161 resize samples

Trạng thái: hoàn thành10 bộ ×3 trạng thái; rerenderGT khớp archive; không training thêm.

Mục đích: xem RGB trước/sau physical resize cùng GT128 và best prediction E161.
Best18000, original4k encoder878,979 trainable khi training, frozen SAM3 features,
EMD+pair λ1, batch8, AdamW3e-4→3e-5,18k updates; xem ../train/config.json.
Gallery không train/infer lại; lấy nguyên predictions đã lưu ở physical_scale/E161.npz.
Checkpoint E161 train/best.pt và SHA trong ../evaluation/summary.json.

10 bộ ×3 trạng thái: scale0.6/1/1.4,8 target-only và2 all-object;
scene0/1/2, đủ4 màu và3 trục local. Chọn theo scene/task cố định trước khi xem
metric, không chọn các case tốt nhất. Mỗi trạng thái RGB320×240, GT/pred128XYZ,
full-scene input4096; một observation, không rollout/flow/action prediction.
Giữ tâm/orientation/camera; resize local axis thật trong simulator và rerender RGBD
bằng đúng seeds E157 evaluation157600000+scene /157700000+caseindex.
So khớp GT sau rerender với archive trước khi dùng cached prediction.
Không alignment/normalization cho cloud; raw baseXYZ đổi m→mm để hiển thị.
GT masks chỉ dùng crop/đánh dấu target trong ảnh, không đưa model.

Run `.venv-concerto/bin/python -m OATFlow.pair_cloud.resize_samples`.
Artifacts: index.html, samples.json, clouds.npz, overview.png và sampleNN_*.png.
Trục XYZ trong kích thước vật là object-local; extent của cloud là robot-base XYZ.
GT là phần bề mặt camera nhìn thấy; resize có thể đổi visibility và sampling.

Gallery: http://localhost:17070/pair-emd/resize_samples/
