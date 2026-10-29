"""Download only the pinned official SAM3 backbone into this bundle's runtime cache."""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from sam3_pick.paths import HF_CACHE
from sam3_pick.inference.prompts import REVISION


def main():
    path = Path(
        snapshot_download(
            repo_id="facebook/sam3",
            revision=REVISION,
            token=True,
            cache_dir=str(HF_CACHE),
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
    )
    h = hashlib.sha256()
    with (path / "model.safetensors").open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    (HF_CACHE / "sam3_identity.json").write_text(
        json.dumps(
            dict(model="facebook/sam3", revision=REVISION, weights_sha256=h.hexdigest()),
            indent=2,
        )
        + "\n"
    )
    print("Pinned SAM3 weights ready", flush=True)


if __name__ == "__main__":
    main()
