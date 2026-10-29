# Unified geometry + semantic — E178

DINOv3 block24 frozen + scalar/vector point encoder + grounded decoder: 4096 scene points và task màu → 128 visible target points. **Weights và output E178 đã giữ trong `outputs/e178/`.** Semantic hiện là bốn màu, chưa open-vocabulary.

- `src/unified_geometry/`: data, geometry, models, losses, training, evaluation, simulation.
- `configs/`: resources và vision settings; `assets/`: scene/robot cố định.
- `tests/`: một bộ regression cho projection/graph, SE(3), permutation, BF16 gradients và retained checkpoint.
- `outputs/e178/`: checkpoints, features, metrics và viewers; không commit.

```bash
cd unified_geometry_semantic
/workspace/.venvs/unified_geometry_semantic/bin/python run.py test
/workspace/.venvs/unified_geometry_semantic/bin/python run.py --help
```

Máy khác: tạo venv Python3.11 ngoài project, cài `requirements.txt`, rồi dùng Python đó chạy `run.py`. Cần CUDA GPU và EGL. `full` tải pinned DINO, sinh RGB-D/benchmarks, trích features, train30k và evaluate; yêu cầu data/output mới.

`configs/resources.json` mặc định dùng `../datasets` và `../pretrained/huggingface`. Code/config/assets/tests đủ để copy/commit project độc lập; không cần E175/E177 hay project SAM3. Dữ liệu và pretrained là resource ngoài source.

Giữ nguyên model/loss: EMD-only, B8, BF16 scalar layers và FP32 geometry/loss. Tests integration skip nếu thiếu resource; chưa chạy lại full30k sau sắp xếp.
