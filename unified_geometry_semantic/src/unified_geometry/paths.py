"""Project-owned outputs and explicit shared resource locations."""

from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parents[2]
ASSETS = PROJECT / "assets"
OUTPUTS = PROJECT / "outputs"
_resources = json.loads((PROJECT / "configs/resources.json").read_text())
DATASETS = (PROJECT / _resources["datasets"]).resolve()
HF_CACHE = (PROJECT / _resources["pretrained"]).resolve()
import hashlib
import random
import numpy as np
import torch

ROOT = OUTPUTS / "e178"
DATA = DATASETS / "e178_rgbd"
BENCHMARK = DATASETS / "e178_benchmark"
GRAPH = BENCHMARK / "graph"
POSES = BENCHMARK / "poses"
CODE = PROJECT / "src/unified_geometry"
BUNDLE = PROJECT
SPLITS = dict(
    train=1024,
    val=128,
    iid=128,
    positions=128,
    rotations=128,
    sizes=128,
    combinations=128,
)
N, K, BATCH, STEPS, DIM = (4096, 128, 8, 30000, 1024)


def setup():
    random.seed(170)
    np.random.seed(170)
    torch.manual_seed(170)
    torch.cuda.manual_seed_all(170)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance():
    sources = {str(p.relative_to(CODE)): digest(p) for p in sorted(CODE.rglob("*.py"))}
    identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return dict(
        git_commit=identity,
        identity_kind="source_sha256",
        original_commit="ed39d076aa330262ed0df6a91ce2ec5c60262886",
        source_sha256=sources,
        config_sha256=digest(PROJECT / "configs/vision.json"),
    )


def save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def attach_readme(folder, status):
    (folder / "README.md").write_text(
        f"# E178 {folder.name}\n\nStatus: {status}.\n\nDINOv3 block24 frozen; 4096 scene points to128 visible target points, four color tasks. Train30k/B8/EMD-only, BF16 scalar layers and FP32 geometry/loss. Actual configuration and metrics are in the JSON files beside this README.\n"
    )
