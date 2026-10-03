# Source this after mounting the host volume at /workspace.
export UV_CACHE_DIR=/workspace/memory_occlusion/cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/memory_occlusion/runtime/python
export HF_HOME=/workspace/memory_occlusion/cache/huggingface
export TORCH_HOME=/workspace/memory_occlusion/cache/torch
export MPLCONFIGDIR=/workspace/memory_occlusion/cache/matplotlib
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="/workspace/memory_occlusion/runtime/bin:$PATH"
source /workspace/memory_occlusion/runtime/.venv/bin/activate || return
cd /workspace/arm-manipulation
