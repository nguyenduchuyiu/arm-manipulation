from sam3_pick.paths import PICK_DATA as DATA
from sam3_pick.paths import OUTPUTS
import json
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from sam3_pick.simulation.grasp import GraspFailure, IKFailure
from sam3_pick.simulation.scene import Scene, NAMES, PERMS
from sam3_pick.simulation.expert import execute
from sam3_pick.data.pick_labels import write
from sam3_pick.data.pick_setup import setup
from sam3_pick.data.pick_split import ROOT
from sam3_pick.data.pick_collect import LOW, HIGH

OUT = ROOT / "test_xy"
NOTE = OUTPUTS / "notes/e122_xy_README.md"


def train_xy(physics):
    rows = [json.loads(s) for s in (ROOT / "train_manifest.jsonl").read_text().splitlines()]
    points = []
    for layout in tqdm(sorted({r["layout_id"] for r in rows}), desc="Audit actual train XY"):
        row = next((r for r in rows if r["layout_id"] == layout))
        scene = Scene(physics)
        with np.load(DATA / row["path"] / "sim_state.npz") as z:
            scene.data.qpos[:] = z["initial_qpos"]
        mujoco.mj_forward(physics, scene.data)
        points.extend((scene.target_positions()[name][:2].copy() for name in NAMES[:4]))
    points = np.asarray(points)
    assert ((points >= LOW - 1e-05) & (points <= HIGH + 1e-05)).all()
    np.save(OUT / "training_xy.npy", points)
    write(
        OUT / "training_xy_audit.json",
        dict(
            layouts=32,
            centers=len(points),
            actual_min=points.min(0).tolist(),
            actual_max=points.max(0).tolist(),
            sampler_low=LOW.tolist(),
            sampler_high=HIGH.tolist(),
            frame="simulator world",
        ),
    )
    return points


def prepare(physics, visual, renderer, option):
    rng = np.random.default_rng(122031)
    rows, rejected, images = ([], [], [])
    display = mujoco.MjData(visual)
    for edge_index, (edge, axis, side) in enumerate((("x_low", 0, -1), ("x_high", 0, 1))):
        boundary = LOW[axis] if side == -1 else HIGH[axis]
        for target in tqdm(range(4), desc=f"Expert validation {edge}"):
            pair = edge_index * 4 + target
            for attempt in range(100):
                positions = rng.uniform(LOW + 0.065, HIGH - 0.065, (4, 2))
                tangent = rng.uniform(LOW[1 - axis] + 0.065, HIGH[1 - axis] - 0.065)
                positions[target, 1 - axis] = tangent
                yaw = rng.uniform(-np.pi, np.pi, 4)
                current, pictures = ([], [])
                try:
                    for group, direction in (("outside", 1), ("inside", -1)):
                        xy = positions.copy()
                        xy[target, axis] = boundary + side * direction * 0.03
                        if (
                            min(
                                (
                                    np.linalg.norm(a - b)
                                    for i, a in enumerate(xy)
                                    for b in xy[i + 1 :]
                                )
                            )
                            < 0.055
                        ):
                            raise ValueError("object center separation below55mm")
                        scene = Scene(physics)
                        initial = setup(scene, dict(positions=xy, yaw=yaw), 0)
                        actual = np.stack(list(scene.target_positions().values()))
                        delta = side * (actual[target, axis] - boundary)
                        if not (delta > 0.028 if group == "outside" else delta < -0.028):
                            raise ValueError("target drifted from planned boundary offset")
                        display.qpos[:] = initial[0]
                        mujoco.mj_forward(visual, display)
                        renderer.update_scene(display, camera="cam_front", scene_option=option)
                        image = renderer.render().copy()
                        renderer.enable_segmentation_rendering()
                        labels = renderer.render()
                        renderer.disable_segmentation_rendering()
                        geoms = np.flatnonzero(visual.geom_bodyid == visual.body(NAMES[target]).id)
                        pixels = int(
                            (
                                np.isin(labels[..., 0], geoms)
                                & (labels[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                            ).sum()
                        )
                        if pixels < 16:
                            raise ValueError(f"target visibility {pixels}px")
                        expert = execute(scene, NAMES[target], lambda phase: None)
                        row = dict(
                            path=f"{edge}/t{target}/{group}",
                            evaluation_group=group,
                            layout_id=pair,
                            permutation=0,
                            target_id=target,
                            target_slot=PERMS[0].index(target),
                            edge=edge,
                            pair=pair,
                            attempt=attempt,
                            target_xy=actual[target, :2].tolist(),
                            all_object_xyz=actual.tolist(),
                            visible_target_pixels=pixels,
                            outside_distance_m=max(0.0, float(delta)),
                            expert=expert,
                        )
                        current.append((row, initial))
                        pictures.append(image)
                except (GraspFailure, IKFailure, ValueError) as error:
                    rejected.append(
                        dict(
                            pair=pair,
                            edge=edge,
                            target=target,
                            attempt=attempt,
                            error=str(error),
                        )
                    )
                    write(OUT / "rejections.json", rejected)
                    continue
                for (row, initial), image in zip(current, pictures):
                    path = OUT / "states" / row["path"]
                    path.mkdir(parents=True)
                    np.savez_compressed(
                        path / "sim_state.npz",
                        **dict(
                            zip(
                                ("initial_qpos", "initial_qvel", "initial_ctrl"),
                                initial,
                            )
                        ),
                    )
                    Image.fromarray(image).save(path / "initial.png")
                    rows.append(row)
                    images.append(image)
                break
            else:
                raise RuntimeError(f"No feasible matched pair: {edge} target{target}")
    assert len(rows) == 16
    (OUT / "manifest.jsonl").write_text("".join((json.dumps(r) + "\n" for r in rows)))
    write(OUT / "rejections.json", rejected)
    sheet = Image.new("RGB", (640, len(rows) // 2 * 264), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (row, image) in enumerate(zip(rows, images)):
        x = i % 2 * 320
        y = i // 2 * 264
        draw.text(
            (x + 5, y + 4),
            f"{row['edge']} {NAMES[row['target_id']]} {row['evaluation_group']}",
            fill="black",
        )
        sheet.paste(Image.fromarray(image), (x, y + 24))
    sheet.save(OUT / "initial_samples.png")
    return rows
