"""Generate UV graphs, original fixed SE(3) transforms and physical OOD inputs."""

import json
import numpy as np
import torch
from tqdm import tqdm
from unified_geometry.paths import DATA, BENCHMARK, GRAPH, POSES, SPLITS, write, digest
from unified_geometry.data.graph import graph
from unified_geometry.geometry.transforms import sample_se3
from unified_geometry.simulation.shapes import Renderer as ShapeRenderer, FAMILIES
from unified_geometry.simulation.camera import Renderer as CameraRenderer, VIEWS


def main():
    GRAPH.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="building", splits={})
    for split, count in SPLITS.items():
        rows = []
        for scene in tqdm(range(count), desc="UV16 graph " + split, mininterval=5):
            with np.load(DATA / "data" / split / f"{scene:05d}.npz") as d:
                rows.append(graph(d["pixel_uv"]))
        path = GRAPH / f"{split}.npy"
        np.save(path, np.array(rows))
        manifest["splits"][split] = dict(sha256=digest(path), scenes=count)
    manifest["status"] = "complete"
    write(GRAPH / "manifest.json", manifest)
    (GRAPH / "README.md").write_text(
        "UV16 graphs from generated observations; all seven E178 splits. Status: complete.\n"
    )
    folder = POSES
    folder.mkdir(parents=True, exist_ok=False)
    for split in ["val", "iid"]:
        count = SPLITS[split]
        R = np.tile(np.eye(3, dtype=np.float32), (5, count, 1, 1))
        t = np.zeros((5, count, 3), np.float32)
        generator = torch.Generator().manual_seed(169000 + list(SPLITS).index(split))
        for view in range(1, 5):
            rr, tt = sample_se3(count, generator, device="cpu")
            R[view] = rr.numpy()
            t[view] = tt.numpy()
        generator = torch.Generator(device="cuda").manual_seed(137100 if split == "val" else 137200)
        for view in range(1, 2 if split == "val" else 5):
            rr, tt = sample_se3(count, generator)
            R[view] = rr.cpu().numpy()
            t[view] = tt.cpu().numpy()
        np.savez_compressed(folder / f"{split}.npz", R=R, t=t)
    (folder / "README.md").write_text(
        "Fixed five-view SE3 poses: canonical identity and four Haar SO(3)+translation views, original E137/E169 RNG seeds. Validation uses historical E137 view1. Status: complete.\n"
    )
    cfg = json.loads((DATA / "data_config.json").read_text())
    K = np.array(cfg["intrinsics"])
    T = np.array(cfg["camera_to_base"])
    for domain, renderer, names in [
        ("shapes", ShapeRenderer(K, T), FAMILIES),
        ("camera", CameraRenderer(K, T), VIEWS),
    ]:
        out = BENCHMARK / domain
        out.mkdir(parents=True, exist_ok=False)
        records = []
        calibration = {}
        for name in names:
            dest = out / "data" / name
            dest.mkdir(parents=True)
            (dest / "README.md").write_text(
                f"{domain}/{name}: 128 IID scenes, four visible targets, original seeded sampling. Status: generating.\n"
            )
            for scene in tqdm(range(SPLITS["iid"]), desc=f"Render {domain}/{name}", mininterval=5):
                with np.load(DATA / "data/iid" / f"{scene:05d}.npz") as d:
                    meta = json.loads(str(d["metadata"]))
                    uv = d["pixel_uv"]
                if domain == "shapes":
                    sample, info = renderer.render(meta, uv, name, 17563000 + scene)
                else:
                    sample, info = renderer.render(meta, name, scene)
                    calibration[name] = info["camera_to_base"]
                assert np.isfinite(sample["target_points"]).all(), (
                    f"Occluded target: {domain}/{name}/{scene}"
                )
                records.append(dict(info, scene=scene, condition=name))
                np.savez_compressed(
                    dest / f"{scene:05d}.npz",
                    **sample,
                    metadata=np.array(json.dumps(meta)),
                )
            (dest / "README.md").write_text(
                (dest / "README.md").read_text().replace("generating", "complete")
            )
        renderer.close()
        write(out / "render_manifest.json", records)
        write(out / "calibration.json", calibration)
        (out / "README.md").write_text(
            f"Generated {domain} RGB-D observations: 128 IID scenes x4 tasks per condition. No feature/model inputs used in rendering. Status: complete.\n"
        )


if __name__ == "__main__":
    main()
