"""Robot visuals and trajectory overlays shared by data rendering and rollouts."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

def visual_model(scene_path, object_bodies=(), lighting_scene=None):
    root = ET.parse(scene_path).getroot()
    reference = ET.parse(Path('/workspace/robotics_assets/robot/nexarm/visual_reference.xml')).getroot()
    meshes = Path('/workspace/robotics_assets/robot/nexarm/meshes')
    root.find('compiler').set('meshdir', str(meshes.resolve()))
    root.append(copy.deepcopy(reference.find('asset')))
    if lighting_scene is not None:
        lighting = ET.parse(lighting_scene).getroot()
        if root.find('visual') is not None:
            root.remove(root.find('visual'))
        root.append(copy.deepcopy(lighting.find('visual')))
        world = root.find('worldbody')
        for light in list(world.findall('light')):
            world.remove(light)
        for light in lighting.findall('worldbody/light'):
            world.append(copy.deepcopy(light))
    for body in root.findall('.//body'):
        original = reference.find(f".//body[@name='{body.get('name')}']")
        if body.get('name') in object_bodies or body.get('mocap') == 'true':
            continue
        assert original is not None
        for geom in original.findall("geom[@class='visual']"):
            body.append(copy.deepcopy(geom))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))

def add_marker(scene, point, rgba, radius=.004):
    assert scene.ngeom < scene.maxgeom
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, [radius]*3, point, np.eye(3).ravel(), rgba)
    scene.ngeom += 1


def path_overlay(scene, points, color, radius):
    for a, b in zip(points, points[1:]):
        if np.linalg.norm(a-b) < 1e-7:
            continue
        assert scene.ngeom < scene.maxgeom
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, [0.]*3, [0.]*3, np.eye(3).ravel(), [v/255 for v in color]+[1.])
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
        scene.ngeom += 1
