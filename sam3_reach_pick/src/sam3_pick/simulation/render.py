"""Robot visuals and trajectory overlays shared by data rendering and rollouts."""

from sam3_pick.paths import ASSETS
import copy
import xml.etree.ElementTree as ET
import mujoco


def visual_model(scene_path, object_bodies=(), lighting_scene=None):
    root = ET.parse(scene_path).getroot()
    reference = ET.parse(ASSETS / "robot/visual_reference.xml").getroot()
    meshes = ASSETS / "robot/meshes"
    root.find("compiler").set("meshdir", str(meshes.resolve()))
    root.append(copy.deepcopy(reference.find("asset")))
    if lighting_scene is not None:
        lighting = ET.parse(lighting_scene).getroot()
        if root.find("visual") is not None:
            root.remove(root.find("visual"))
        root.append(copy.deepcopy(lighting.find("visual")))
        world = root.find("worldbody")
        for light in list(world.findall("light")):
            world.remove(light)
        for light in lighting.findall("worldbody/light"):
            world.append(copy.deepcopy(light))
    for body in root.findall(".//body"):
        original = reference.find(f".//body[@name='{body.get('name')}']")
        if body.get("name") in object_bodies or body.get("mocap") == "true":
            continue
        assert original is not None
        for geom in original.findall("geom[@class='visual']"):
            body.append(copy.deepcopy(geom))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
