"""Regenerate the original E135 poses and E157 physically resized RGB-D splits."""

import json
import hashlib
import numpy as np
from tqdm import tqdm
from unified_geometry.paths import DATA, SPLITS, write
from unified_geometry.simulation.renderer import Renderer, valid_dimensions
from unified_geometry.simulation.sampling import parameters


def main():
    DATA.mkdir(parents=True, exist_ok=False)
    (DATA / "README.md").write_text(
        "E157 procedural data: E135 seeded pose/visibility rejection, then E157 physical local XYZ resize. 1024 train,128 val,5x128 test; 4096 full-scene points/128 visible GT per task. No physics steps, no GT input. See data_config.json. Status: generating.\n"
    )
    renderer = Renderer()
    manifest = []
    rejected = {}
    config = dict(
        status="generating",
        splits=SPLITS,
        scene_points=4096,
        target_points=128,
        intrinsics=renderer.intrinsics.tolist(),
        camera_to_base=renderer.transform.tolist(),
        pose_seed="135000000 + split_index*100000 + scene_index",
        resize_seed="157000000 + split_index*100000 + scene_index",
        source="Procedural E135 sampling then physical E157 resize; no historical dataset read",
        static_scene=True,
        physics_stepped=False,
        GT_model_inputs=False,
    )
    write(DATA / "data_config.json", config)
    for si, (split, count) in enumerate(SPLITS.items()):
        folder = DATA / "data" / split
        folder.mkdir(parents=True)
        rejected[split] = dict(source=0, resize=0)
        for index in tqdm(range(count), desc="Generate " + split, mininterval=5):
            rng = np.random.default_rng(135000000 + si * 100000 + index)
            for attempt in range(1000):
                rows = parameters(rng, split)
                if rows is not None and renderer.render(rows, np.ones((4, 3)), rng) is not None:
                    break
                rejected[split]["source"] += 1
            else:
                raise RuntimeError(f"Cannot generate source {split}/{index}")
            fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            rng = np.random.default_rng(157000000 + si * 100000 + index)
            for attempt in range(1000):
                scales = rng.uniform(0.6, 1.4, size=(4, 3))
                rendered = (
                    renderer.render(rows, scales, rng) if valid_dimensions(rows, scales) else None
                )
                if rendered is not None:
                    break
                rejected[split]["resize"] += 1
            else:
                raise RuntimeError(f"Cannot resize {split}/{index}")
            sample, visible = rendered
            meta = dict(
                split=split,
                index=index,
                source_scene_fingerprint=fingerprint,
                cubes=rows,
                physical_scale_xyz=scales.tolist(),
                dimensions_m=(np.array([r["side_m"] for r in rows])[:, None] * scales).tolist(),
                visible_pixels=visible,
                input_points_per_cube=[
                    int(
                        (
                            sample["instance_labels"][
                                sample["pixel_uv"][:, 1], sample["pixel_uv"][:, 0]
                            ]
                            == t + 1
                        ).sum()
                    )
                    for t in range(4)
                ],
                attempts=attempt + 1,
                seed=157000000 + si * 100000 + index,
            )
            np.savez_compressed(
                folder / f"{index:05d}.npz",
                **sample,
                metadata=np.array(json.dumps(meta)),
            )
            manifest.append(meta)
    renderer.close()
    config.update(status="data_complete", rejections=rejected)
    write(DATA / "data_config.json", config)
    (DATA / "manifest.jsonl").write_text("".join((json.dumps(row) + "\n" for row in manifest)))
    (DATA / "README.md").write_text(
        (DATA / "README.md").read_text().replace("Status: generating.", "Status: complete.")
    )


if __name__ == "__main__":
    main()
