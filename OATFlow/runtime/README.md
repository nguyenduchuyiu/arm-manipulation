# Runtime on a Vast local volume

Mount the same host volume at `/workspace` on the next instance. Vast local volumes
can be attached to instances on the same host. Confirm the mount with
`findmnt -T /workspace`; ordinary container storage does not survive recreation.

The code checkout is `/workspace/arm-manipulation`. Runtime artifacts live at
`/workspace/memory_occlusion/runtime`: a copied uv binary, managed Python 3.12.3,
the `.venv` for data collection / TCOW / flow training and inference, and a frozen
package list. Wheels and model caches live under `/workspace/memory_occlusion/cache`.
The interpreter and packages persist with the volume and do not depend on the
next image's system Python. The image still needs Git and the host NVIDIA/EGL driver.

Initial setup, with live logs in the `huy` tmux session:

```bash
cd /workspace/arm-manipulation
set -o pipefail
taskset -c 0-7 bash memory_occlusion/runtime/setup_volume.sh \
  2>&1 | tee /workspace/memory_occlusion/logs/memory_occlusion_runtime_setup.log
```

PyTorch uses CUDA 12.8 wheels for RTX 50-series GPUs. Generation dependencies
match the existing dataset environment. The original TCOW checkout, including
its bundled TimeSformer, is pinned and left unmodified. Model checkpoints are
separate inputs to the train and inference commands.

Pretrained inputs on this volume:

* `/workspace/memory_occlusion/checkpoints/memory_occlusion_tracker_init/checkpoint.pth`:
  original published TCOW weights, SHA256
  `682ba4957b86405f2c3f2a6f3fa7bfb5c8e71c8d3770ff0966ecf1063ed31060`.
* The adjacent `config.pt` contains only architecture and mask-loss settings;
  it supplies `--config-checkpoint`, without any finetuned weights.
* `/workspace/memory_occlusion/checkpoints/memory_occlusion_action_expert_init/expert.safetensors`:
  179 action-expert tensors from `lerobot/smolvla_base` revision
  `d9f33c94a60fb382c90dea2164c96845bd955e28`. The adjacent
  `import_report.json` records the verified source checksum and load report.
  The unused VLM weights are removed after extraction.

After mounting the volume on another instance:

```bash
source /workspace/arm-manipulation/memory_occlusion/runtime/activate.sh
python -m memory_occlusion.dataset.generate_dataset --help
python -m memory_occlusion.experiments.tcow_joint_flow.train_from_tcow_context --help
```

The existing repository `.venv` is the environment of the running collection job;
setup does not modify it. Use the runtime activation above for subsequent jobs.
