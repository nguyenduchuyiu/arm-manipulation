"""Regenerate the original 16 expert-validated XY boundary cases, without E122 weights."""

from sam3_pick.paths import PICK_DATA as DATA
import shutil
import mujoco
from sam3_pick.simulation.scene import NAMES
from sam3_pick.data.rendering import clip_planes
from sam3_pick.simulation.render import visual_model
from sam3_pick.evaluation.xy_cases import OUT, NOTE, train_xy, prepare


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(NOTE, OUT / "README.md")
    physics = mujoco.MjModel.from_xml_path(str(DATA / "scene.xml"))
    visual = visual_model(DATA / "scene.xml", object_bodies=NAMES)
    clip_planes(visual)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(visual, height=240, width=320) as renderer:
        train_xy(physics)
        rows = prepare(physics, visual, renderer, option)
    assert len(rows) == 16


if __name__ == "__main__":
    main()
