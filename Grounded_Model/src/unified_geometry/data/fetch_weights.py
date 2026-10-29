"""Fetch the pinned frozen DINOv3 model into the configured HF cache."""

import json
from huggingface_hub import snapshot_download
from unified_geometry.paths import PROJECT, HF_CACHE, digest


def main():
    selection = json.loads((PROJECT / "configs/vision.json").read_text())
    path = snapshot_download(
        selection["model"],
        revision=selection["revision"],
        cache_dir=HF_CACHE,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "LICENSE.md",
        ],
    )
    from pathlib import Path

    assert digest(Path(path) / "model.safetensors") == selection["weights_sha256"]
    print("Pinned DINOv3 weights verified", flush=True)


if __name__ == "__main__":
    main()
