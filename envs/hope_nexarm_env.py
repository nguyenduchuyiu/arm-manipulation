"""Random HOPE tabletop scene for the NexArm MuJoCo controller."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .nexarm_env import NexArmEnv


ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "assets/hope-dataset/meshes/eval"
SCENE_PATH = ROOT / "assets/robot/hope_scene.xml"
SLOTS = 3
SLOT_X = (0.54, 0.28, 0.80)
SPAWN_CLEARANCE = 0.02
GRASPABLE_TARGETS = ("Butter", "ChocolatePudding", "CreamCheese", "Popcorn", "Raisins")
HOPE_OBJECTS = GRASPABLE_TARGETS + ("Milk", "OrangeJuice", "Tuna")


def hope_objects() -> dict[str, np.ndarray]:
    """Return collision half extents in metres; HOPE OBJ coordinates are cm, Y-up."""
    sizes = {}
    for name in HOPE_OBJECTS:
        path = MESH_DIR / f"{name}.obj"
        texture = MESH_DIR / f"{name}.png"
        if not path.is_file() or not texture.is_file():
            raise FileNotFoundError(f"Missing HOPE mesh or texture for {name} in {MESH_DIR}")
        vertices = np.array(
            [list(map(float, line.split()[1:4])) for line in path.open() if line.startswith("v ")]
        )
        half = (vertices.max(axis=0) - vertices.min(axis=0)) * 0.005
        sizes[name] = half[[0, 2, 1]]
    return sizes


def build_hope_scene(path: Path = SCENE_PATH) -> Path:
    """Write a portable MJCF scene that includes the existing NexArm model."""
    sizes = hope_objects()
    root = ET.Element("mujoco", model="NexArm-HOPE")
    ET.SubElement(root, "include", file="robot.xml")
    asset = ET.SubElement(root, "asset")
    for name in sizes:
        texture = f"../hope-dataset/meshes/eval/{name}.png"
        mesh = f"../../hope-dataset/meshes/eval/{name}.obj"
        ET.SubElement(asset, "texture", name=f"hope_{name}_tex", type="2d", file=texture)
        ET.SubElement(asset, "material", name=f"hope_{name}_mat", texture=f"hope_{name}_tex")
        ET.SubElement(asset, "mesh", name=f"hope_{name}_mesh", file=mesh,
                      scale="0.01 0.01 0.01")

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.3 0.3 0.3",
                  diffuse="0.4 0.4 0.4", specular="0.15 0.15 0.15")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", pos="0 0 -0.75",
                  size="2 2 0.1", rgba="0.78 0.79 0.76 1")
    ET.SubElement(world, "geom", name="table_top", type="box", pos="0.54 -0.10 -0.025",
                  size="0.55 0.43 0.025", rgba="0.57 0.37 0.22 1", friction="1 0.01 0.001")
    for x in (0.06, 1.02):
        for y in (-0.48, 0.28):
            ET.SubElement(world, "geom", type="box", pos=f"{x} {y} -0.375",
                          size="0.025 0.025 0.35", rgba="0.25 0.25 0.27 1")
    ET.SubElement(world, "geom", name="chair_seat", type="box", pos="0.20 0.67 -0.35",
                  size="0.22 0.22 0.025", rgba="0.2 0.33 0.43 1")
    ET.SubElement(world, "geom", name="chair_back", type="box", pos="0.20 0.87 -0.05",
                  size="0.22 0.025 0.30", rgba="0.2 0.33 0.43 1")
    for x in (0.02, 0.38):
        for y in (0.50, 0.84):
            ET.SubElement(world, "geom", type="box", pos=f"{x} {y} -0.55",
                          size="0.018 0.018 0.18", rgba="0.15 0.18 0.21 1")
    ET.SubElement(world, "camera", name="front", pos="0.54 -0.78 0.55",
                  xyaxes="1 0 0 0 0.42 0.907", fovy="45")
    ET.SubElement(world, "light", pos="0.2 -0.5 1.4", dir="0.2 0.3 -1",
                  diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "light", pos="0.9 0.2 1.1", dir="-0.2 -0.2 -1",
                  diffuse="0.35 0.35 0.35")
    first = "Butter"
    # MuJoCo's collision broadphase bounds are compiled once. They must cover
    # every mesh size assigned to a slot later, or tall objects miss the table.
    collision_bound = np.max(np.stack(tuple(sizes.values())), axis=0)
    for slot in range(SLOTS):
        body_name = "cube" if slot == 0 else f"hope_object_{slot}"
        joint_name = "cube_joint" if slot == 0 else f"hope_joint_{slot}"
        body = ET.SubElement(world, "body", name=body_name,
                             pos=f"{SLOT_X[slot]} -0.21 {sizes[first][2] + SPAWN_CLEARANCE}")
        ET.SubElement(body, "freejoint", name=joint_name)
        # Each OBJ needs its own compiled geom transform; swapping geom_dataid
        # would retain Butter's mesh alignment and make other objects look sunk.
        for name in sizes:
            ET.SubElement(body, "geom", name=f"hope_visual_{slot}_{name}", type="mesh",
                          mesh=f"hope_{name}_mesh", material=f"hope_{name}_mat",
                          quat="0.70710678 0.70710678 0 0",
                          contype="0", conaffinity="0", group="2", mass="0",
                          rgba="1 1 1 1" if name == first else "1 1 1 0")
        ET.SubElement(body, "geom", name=f"hope_collision_{slot}", type="box",
                      size=" ".join(map(str, collision_bound)), mass="0.08",
                      rgba="0 0 0 0", friction="1.2 0.01 0.001")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)
    return path


class HopeNexArmEnv(NexArmEnv):
    """Pick the centre HOPE object and lift it 8 cm above the tabletop."""

    def __init__(self, **kwargs) -> None:
        self.hope_sizes = hope_objects()
        scene_path = build_hope_scene()
        super().__init__(scene_path=scene_path, **kwargs)
        self.hope_names = tuple(self.hope_sizes)
        self.slot_qpos = [self.model.jnt_qposadr[self.model.joint(
            "cube_joint" if i == 0 else f"hope_joint_{i}").id] for i in range(SLOTS)]
        self.slot_visuals = [
            {name: self.model.geom(f"hope_visual_{i}_{name}").id for name in self.hope_names}
            for i in range(SLOTS)
        ]
        self.slot_collisions = [self.model.geom(f"hope_collision_{i}").id for i in range(SLOTS)]
        self.object_names: tuple[str, ...] = ()
        camera = self.model.camera("wrist")
        camera.pos[:] = (0.539, -0.07, 0.32)
        forward = np.array((0.0, -0.13, -0.30))
        forward /= np.linalg.norm(forward)
        right = np.array((1.0, 0.0, 0.0))
        up = np.cross(-forward, right)
        rotation = np.column_stack((right, up, -forward))
        mujoco.mju_mat2Quat(camera.quat, rotation.flatten())

    def _reset_object(self) -> None:
        target = str(self.np_random.choice(GRASPABLE_TARGETS))
        distractors = self.np_random.choice(
            [name for name in self.hope_names if name != target],
            size=SLOTS - 1, replace=False,
        )
        self.object_names = (target, *map(str, distractors))
        for slot, name in enumerate(self.object_names):
            half = self.hope_sizes[name]
            collision = self.slot_collisions[slot]
            for mesh_name, visual in self.slot_visuals[slot].items():
                self.model.geom_rgba[visual, 3] = float(mesh_name == name)
            self.model.geom_size[collision] = half
            body = self.model.geom_bodyid[collision]
            self.model.body_mass[body] = 0.08
            self.model.body_inertia[body] = 0.08 / 3 * np.array(
                (half[1] ** 2 + half[2] ** 2,
                 half[0] ** 2 + half[2] ** 2,
                 half[0] ** 2 + half[1] ** 2)
            )
            qadr = self.slot_qpos[slot]
            x = SLOT_X[slot] + self.np_random.uniform(-0.015, 0.015)
            y = -0.21 + self.np_random.uniform(-0.025, 0.025)
            self.data.qpos[qadr:qadr + 3] = (x, y, half[2] + SPAWN_CLEARANCE)
            if slot == 0:
                yaw = self.np_random.uniform(np.pi / 2 - 0.15, np.pi / 2 + 0.15)
            else:
                yaw = self.np_random.uniform(-np.pi, np.pi)
            self.data.qpos[qadr + 3:qadr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        self.lift_height = float(self.hope_sizes[self.object_names[0]][2] + 0.08)

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["hope_objects"] = self.object_names
        info["target_object"] = self.object_names[0] if self.object_names else None
        return info
