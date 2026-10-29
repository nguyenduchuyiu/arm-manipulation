# NexArm experiments

| Project | Mục đích | Trạng thái weights |
|---|---|---|
| [sam3_reach_pick](sam3_reach_pick/README.md) | Voxel reach → point-cloud pick → SAM3 rollout | E118 và SAM3 còn; E131 cần train lại |
| [unified_geometry_semantic](unified_geometry_semantic/README.md) | E178: DINOv3 + equivariant geometry/semantic grounding | Best checkpoint và data còn |

Mỗi project tự chứa `src/`, `configs/`, `assets/`, `tests/`, `run.py` và requirements; có thể copy/commit riêng, không import project còn lại.

- `datasets/`: dữ liệu dùng chung, bỏ khỏi Git.
- `pretrained/huggingface/`: pretrained HF, bỏ khỏi Git.
- `<project>/outputs/`: checkpoints, feature outputs, metrics và rollout của project đó, bỏ khỏi Git.
- `docs/research/`: ghi chép Markdown lịch sử.

Đường dẫn resources được cấu hình trong từng `configs/resources.json`, tương đối theo project. Venv hiện tại ở `/workspace/.venvs/`; source repo không chứa môi trường Python. `AGENT.md` giữ nguyên.
