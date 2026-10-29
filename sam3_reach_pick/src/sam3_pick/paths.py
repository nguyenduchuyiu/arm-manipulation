"""Project-owned outputs and explicit shared resource locations."""

from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parents[2]
ASSETS = PROJECT / "assets"
OUTPUTS = PROJECT / "outputs"
_resources = json.loads((PROJECT / "configs/resources.json").read_text())
DATASETS = (PROJECT / _resources["datasets"]).resolve()
HF_CACHE = (PROJECT / _resources["pretrained"]).resolve()
PICK_DATA = DATASETS / "pick_wide_rgbd"
