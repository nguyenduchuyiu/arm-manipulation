"""Local MuJoCo scene assets and original rendering setup."""

from unified_geometry.paths import ASSETS
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import copy
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

BASE = np.array([0.539369022999, 0.0639727628473, 0.0])
NAMES = ("RedCube", "BlueCube", "GreenCube", "YellowCube", "cover_a", "cover_b")


def visual_model():
    root = ET.parse(ASSETS / "scene.xml").getroot()
    reference = ET.parse(ASSETS / "visual_reference.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSETS / "meshes"))
    root.append(copy.deepcopy(reference.find("asset")))
    for body in root.findall(".//body"):
        original = reference.find(f".//body[@name='{body.get('name')}']")
        if body.get("name") in NAMES or body.get("mocap") == "true":
            continue
        assert original is not None
        for geom in original.findall("geom[@class='visual']"):
            body.append(copy.deepcopy(geom))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def clip_planes(model):
    model.vis.map.znear = 0.001 / model.stat.extent
    model.vis.map.zfar = 10.0 / model.stat.extent
