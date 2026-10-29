# SAM3 reach/pick — E118 → E131 → E133

Voxel/expert data → point-cloud reach → pick finetuning → SAM3 front/left masks → depth fusion → FM rollout. **E118 và SAM3 còn weights; E131 cần train lại.**

- `src/sam3_pick/`: data, models, training, inference, evaluation, simulation.
- `configs/`: đường dẫn resources và mẫu mô tả run; `assets/`: robot/scene cố định.
- `tests/`: một bộ regression cho cloud/fusion, BF16 gradients và retained-model integration.
- `outputs/e118_reach`, `outputs/e131_pick`, `outputs/e133_sam3`: output riêng từng giai đoạn.

```bash
cd sam3_reach_pick
/workspace/.venvs/sam3_reach_pick/bin/python run.py test
/workspace/.venvs/sam3_reach_pick/bin/python run.py pick_train
```

Máy khác: tạo venv Python3.11 ở ngoài project, cài `requirements.txt`, rồi dùng Python đó chạy `run.py`. Cần CUDA GPU, EGL và FFmpeg. `run.py --help` liệt kê các stage; `full` sinh data và train/test từ đầu, yêu cầu output mới và quyền HF cho `facebook/sam3`.

`configs/resources.json` trỏ tới `../datasets` và `../pretrained/huggingface`; sửa hai đường dẫn khi cần. Không cần repo/project khác. Git chỉ lấy source/config/assets/tests; datasets, pretrained, outputs và venv không commit.

Model giữ nguyên: PointMLP + FM16layers, H25×10; pick23520updates/B128, BF16. Rollout K15/Euler10/noise0. Tests integration tự skip nếu chưa có data/weights; không phải phép đo pick accuracy.
