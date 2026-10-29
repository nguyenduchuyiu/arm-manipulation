"""Persist train/val1024Dfeatures; test splits temporary, identity config immutable."""

import json
import numpy as np
import torch
from tqdm import tqdm
from unified_geometry.paths import (
    ROOT,
    DATA,
    SPLITS,
    N,
    DIM,
    setup,
    provenance,
    write,
    digest,
    attach_readme,
)
from unified_geometry.data.vision import Features, WEIGHTS, RESOLUTION


def ensure(split):
    folder = ROOT / "features"
    selection = json.loads((ROOT / "feature_selection.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / f"{split}.npy"
    if path.exists():
        info = manifest["splits"][split]
        assert info["status"] == "complete" and digest(path) == info["sha256"]
        return
    cfg = json.loads((DATA / "data_config.json").read_text())
    extractor = Features(
        np.array(cfg["intrinsics"]),
        np.array(cfg["camera_to_base"]),
        selection["selected"],
    )
    cache = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(SPLITS[split], N, DIM)
    )
    for start in tqdm(
        range(0, SPLITS[split], 2),
        desc="DINO" + selection["selected"] + " " + split,
        mininterval=10,
    ):
        samples = []
        for scene in range(start, min(start + 2, SPLITS[split])):
            with np.load(DATA / "data" / split / f"{scene:05d}.npz") as d:
                samples.append({key: d[key] for key in ["rgb", "scene_points", "pixel_uv"]})
        value = extractor(samples)
        assert value.shape == (len(samples), N, DIM) and torch.isfinite(value).all()
        cache[start : start + len(samples)] = value.cpu().numpy()
    cache.flush()
    del cache
    assert extractor.max_projection_error < 0.001
    manifest["splits"][split] = dict(
        status="complete",
        sha256=digest(path),
        scenes=SPLITS[split],
        bytes=path.stat().st_size,
        projection_max_error_pixels=extractor.max_projection_error,
        kept=True,
    )
    write(folder / "manifest.json", manifest)
    del extractor
    torch.cuda.empty_cache()


def discard_test(split):
    assert split not in ["train", "val"]
    folder = ROOT / "features"
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / f"{split}.npy"
    assert path.exists()
    path.unlink()
    manifest["splits"][split]["kept"] = False
    write(folder / "manifest.json", manifest)


def main():
    setup()
    meta = provenance()
    selection = json.loads((ROOT / "feature_selection.json").read_text())
    assert (
        selection["status"] == "complete"
        and selection["selected"] == "block24"
        and (digest(WEIGHTS / "model.safetensors") == selection["weights_sha256"])
    )
    out = ROOT / "features"
    out.mkdir(exist_ok=False)
    attach_readme(out, "buildingselectedDINOfrozenfeatures")
    write(out / "manifest.json", dict(splits={}))
    ensure("train")
    ensure("val")
    write(
        out / "config.json",
        dict(
            status="complete",
            model=selection["model"],
            revision=selection["revision"],
            weights_sha256=selection["weights_sha256"],
            feature=selection["selected"],
            feature_dim=DIM,
            feature_grid=[72, 72],
            input_resize=[RESOLUTION, RESOLUTION],
            backbone_frozen=True,
            parameter_precision="BF16",
            cache_precision="FP16",
            patch_tokens_only=True,
            prefix_tokens_removed=5,
            GT_used_as_feature_input=False,
            selection_sha256=digest(ROOT / "feature_selection.json"),
            data_config_sha256=digest(DATA / "data_config.json"),
            provenance=meta,
        ),
    )
    attach_readme(out, "train/valfeaturecachecomplete")
    print("DINOtrain/val cache ready; heldouts extracted on demand", flush=True)


if __name__ == "__main__":
    main()
