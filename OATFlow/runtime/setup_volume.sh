#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ "$(findmnt -n -o TARGET -T /workspace)" == /workspace ]]
[[ "$PWD" == /workspace/arm-manipulation ]]
runtime=/workspace/memory_occlusion/runtime
mkdir -p "$runtime/bin" "$runtime/python" /workspace/memory_occlusion/cache/uv
if [[ ! -x "$runtime/bin/uv" ]]; then
  cp "$(command -v uv)" "$runtime/bin/uv"
fi
export UV_CACHE_DIR=/workspace/memory_occlusion/cache/uv
export UV_PYTHON_INSTALL_DIR="$runtime/python"
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_INSTALLS=2 UV_CONCURRENT_BUILDS=1
uv="$runtime/bin/uv"
"$uv" python install 3.12.3
if [[ ! -d "$runtime/.venv" ]]; then
  "$uv" venv --python 3.12.3 --managed-python "$runtime/.venv"
fi
requirements=memory_occlusion/runtime/requirements.txt
if [[ -f "$runtime/requirements.lock" ]]; then
  requirements="$runtime/requirements.lock"
fi
"$uv" pip install --python "$runtime/.venv/bin/python" \
  --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  -r "$requirements"
"$uv" pip check --python "$runtime/.venv/bin/python"
"$uv" pip freeze --python "$runtime/.venv/bin/python" > "$runtime/requirements.lock"

vendor=memory_occlusion/third_party/tcow
revision=a72e3e13a45e4156137328e5290f9e848d360367
if [[ ! -d "$vendor" ]]; then
  git init "$vendor"
  git -C "$vendor" remote add origin https://github.com/basilevh/tcow.git
  git -C "$vendor" fetch --depth 1 origin "$revision"
  git -C "$vendor" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$vendor" rev-parse HEAD)" == "$revision" ]]
[[ -f "$vendor/third_party/TimeSformer/timesformer/models/vit.py" ]]
printf 'Runtime ready. Source /workspace/arm-manipulation/memory_occlusion/runtime/activate.sh\n'
