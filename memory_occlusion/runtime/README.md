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

After mounting the volume on another instance:

```bash
source /workspace/arm-manipulation/memory_occlusion/runtime/activate.sh
python -m memory_occlusion.dataset.generate_dataset --help
python -m memory_occlusion.experiments.tcow_joint_flow.train_from_tcow_context --help
```

The existing repository `.venv` is the environment of the running collection job;
setup does not modify it. Use the runtime activation above for subsequent jobs.
