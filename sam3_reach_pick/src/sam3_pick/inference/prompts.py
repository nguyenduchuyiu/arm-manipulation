"""Original official SAM 3 BF16 image/text segmentation on fixed pick samples."""

import hashlib

REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
PROMPTS = ["red cube", "blue cube", "green cube", "yellow cube"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = ["PROMPTS", "REVISION", "digest"]
