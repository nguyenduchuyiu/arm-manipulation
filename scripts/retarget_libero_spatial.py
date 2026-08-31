"""Retarget the first LIBERO-Spatial bowl task to NexArm."""
from __future__ import annotations

import json
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import imageio.v3 as iio
import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from retarget_libero_drawer import (
    CONTROL_STEPS_PER_RECORD,
    MAX_POSITION_DELTA,
    RECORD_HZ,
    arm_addresses,
    object_id,
    render,
    site_position,
)

SOURCE = Path("data/libero_spatial")
OUTPUT = Path("data/nexarm_libero_spatial")
ROBOT = Path("assets/robot/robot.xml")
VIDEO_DIR = Path("outputs/nexarm_libero_spatial")
IMAGE_SIZE = 128
VIDEO_SIZE = 768
OBJECT_SCALE = np.full(3, 0.25 / 0.7)
TABLE_TOP = 0.9
HELD_BOWL_OFFSET = np.array([0.012, 0.002])
LIBERO_ASSETS = Path("LIBERO/libero/libero/assets").resolve()
ROBOSUITE_ASSETS = Path(
    "/opt/homebrew/Caskroom/miniforge/base/envs/libero-replay/lib/python3.10/"
    "site-packages/robosuite/models/assets"
)


def numbers(value):
    return np.fromstring(value, sep=" ")


def remap_assets(xml):
    return xml.replace(
        "/Users/yifengz/workspace/libero-dev/chiliocosm/assets", str(LIBERO_ASSETS)
    ).replace(
        "/Users/yifengz/workspace/robosuite-master/robosuite/models/assets",
        str(ROBOSUITE_ASSETS),
    )


def scale_free_objects(root):
    """Scale movable LIBERO objects while retaining their original collision meshes."""
    mesh_names = set()
    free_bodies = []
    for body in root.findall(".//worldbody//body"):
        joint = body.find("joint[@type='free']")
        if joint is None:
            continue
        free_bodies.append(joint.attrib["name"])
        for element in body.iter():
            if "mesh" in element.attrib:
                mesh_names.add(element.attrib["mesh"])
            for attribute in ("pos", "size", "fromto"):
                if attribute in element.attrib:
                    values = numbers(element.attrib[attribute])
                    if len(values) == 1:
                        scaled = values * OBJECT_SCALE[0]
                    else:
                        scaled = (values.reshape(-1, 3) * OBJECT_SCALE).ravel()
                    element.set(
                        attribute,
                        " ".join(f"{x:.9g}" for x in scaled),
                    )
            if "density" in element.attrib:
                element.set(
                    "density",
                    str(float(element.attrib["density"]) / np.prod(OBJECT_SCALE)),
                )
    for mesh in root.findall("./asset/mesh"):
        if mesh.get("name") in mesh_names:
            scale = numbers(mesh.get("scale", "1 1 1")) * OBJECT_SCALE
            mesh.set("scale", " ".join(map(str, scale)))
    return free_bodies


def build_model(source):
    """Replace Panda in the exact per-demo LIBERO MJCF with NexArm."""
    original_xml = remap_assets(source.attrs["model_file"])
    original = mujoco.MjModel.from_xml_string(original_xml)
    state_qpos = source["states"][0, 1 : 1 + original.nq]
    root = ET.fromstring(original_xml)
    world = root.find("worldbody")
    world.remove(world.find("body[@name='robot0_base']"))
    root.remove(root.find("actuator"))
    sensor = root.find("sensor")
    if sensor is not None:
        root.remove(sensor)

    free_bodies = scale_free_objects(root)
    poses = {}
    for joint_name in free_bodies:
        joint = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        address = original.jnt_qposadr[joint]
        pose = state_qpos[address : address + 7].copy()
        pose[2] = TABLE_TOP + (pose[2] - TABLE_TOP) * OBJECT_SCALE[2]
        poses[joint_name] = pose

    robot = ET.parse(ROBOT).getroot()
    for mesh in robot.findall("./asset/mesh"):
        mesh.set("file", str((ROBOT.parent / "meshes" / mesh.attrib["file"]).resolve()))
    root.find("asset").extend(copy.deepcopy(list(robot.find("asset"))))
    root.find("default").extend(copy.deepcopy(list(robot.find("default"))))
    base = copy.deepcopy(robot.find("worldbody/body[@name='base_link']"))
    base_center = np.array([-0.2, 0.25])
    grasp_vector = poses["akita_black_bowl_1_joint0"][:2] - base_center
    if np.linalg.norm(grasp_vector) > 0.3:
        base_center = poses["akita_black_bowl_1_joint0"][:2] - 0.3 * (
            grasp_vector / np.linalg.norm(grasp_vector)
        )
    base_pos = np.r_[base_center - np.array([0.53937, 0.06397]), TABLE_TOP]
    base.set("pos", " ".join(map(str, base_pos)))
    world.append(base)
    for section_name in ("equality", "contact", "actuator"):
        section = copy.deepcopy(robot.find(section_name))
        if section_name == "actuator":
            gripper = section.find("position[@name='gripper_control']")
            gripper.set("kp", "300")
            gripper.set("kv", "5")
        root.append(section)
    root.find("option").set("timestep", "0.002")
    root.find("option").set("integrator", "implicitfast")
    global_visual = root.find("visual/global")
    if global_visual is None:
        global_visual = ET.SubElement(root.find("visual"), "global")
    global_visual.set("offwidth", str(VIDEO_SIZE))
    global_visual.set("offheight", str(VIDEO_SIZE))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, poses


def joint_path(model, arm_qpos, arm_dofs, site, targets):
    clone = mujoco.MjData(model)
    joints = model.actuator_trnid[:5, 0]
    low, high = model.jnt_range[joints, 0], model.jnt_range[joints, 1]
    path = []
    jacobian_pos = np.zeros((3, model.nv))
    jacobian_rot = np.zeros((3, model.nv))
    fallbacks = 0

    def forward_kinematics():
        mujoco.mj_kinematics(model, clone)
        mujoco.mj_comPos(model, clone)

    def error(value, target, previous):
        clone.qpos[arm_qpos] = value
        forward_kinematics()
        approach = np.asarray(clone.site_xmat[site]).reshape(3, 3)[:, 1]
        return np.r_[
            100 * (site_position(clone, site) - target),
            2 * (approach - np.array([0.0, 0.0, 1.0])),
            0.05 * (value - previous),
        ]

    q = np.zeros(5)
    for index, target in enumerate(targets):
        previous = q.copy()
        if index == 0:
            q = least_squares(
                error, q, args=(target, previous), bounds=(low, high), max_nfev=200
            ).x
        else:
            for _ in range(3):
                clone.qpos[arm_qpos] = q
                forward_kinematics()
                position_error = target - site_position(clone, site)
                approach = np.asarray(clone.site_xmat[site]).reshape(3, 3)[:, 1]
                rotation_error = np.cross(approach, np.array([0.0, 0.0, 1.0]))
                mujoco.mj_jacSite(model, clone, jacobian_pos, jacobian_rot, site)
                jacobian = np.vstack(
                    (jacobian_pos[:, arm_dofs], 0.02 * jacobian_rot[:, arm_dofs])
                )
                residual = np.r_[position_error, 0.02 * rotation_error]
                delta = np.linalg.solve(
                    jacobian.T @ jacobian + 1e-5 * np.eye(5),
                    jacobian.T @ residual,
                )
                q = np.clip(q + np.clip(delta, -0.05, 0.05), low, high)
            if np.linalg.norm(error(q, target, previous)[:3]) > 0.8:
                fallbacks += 1
                q = least_squares(
                    error, q, args=(target, previous), bounds=(low, high), max_nfev=50
                ).x
        if np.linalg.norm(error(q, target, previous)[:3]) > 0.8:
            raise RuntimeError(f"IK unreachable: {target.round(3)}")
        path.append(q.copy())
    print(f"IK: {len(path)} targets, {fallbacks} optimizer fallbacks", flush=True)
    return np.asarray(path)


def bodies_touch(model, data, body_a, body_b):
    for contact in data.contact:
        a = int(model.geom_bodyid[contact.geom1])
        b = int(model.geom_bodyid[contact.geom2])
        if (a == body_a and b == body_b) or (a == body_b and b == body_a):
            return True
    return False


def retarget(source: h5py.Group, model, data, renderer, video_renderer, ids, poses):
    xyz = source["obs/ee_pos"][()].copy()
    xyz[:, 2] = np.clip(xyz[:, 2], TABLE_TOP + 0.014, TABLE_TOP + 0.29)
    source_grip = source["actions"][:, 6]
    close = np.flatnonzero((source_grip[:-1] < 0) & (source_grip[1:] > 0)) + 1
    release = np.flatnonzero((source_grip[:-1] > 0) & (source_grip[1:] < 0)) + 1
    pairs = [(int(next((r for r in release if r > c), len(xyz) - 1)), int(c)) for c in close]
    if not pairs:
        raise RuntimeError("demonstration never closes the gripper")
    r, c = max(pairs, key=lambda pair: pair[0] - pair[1])
    arm_act, arm_qpos, arm_dofs, grip_act, grip_qpos, site, bowl_qpos, plate_qpos = ids
    bowl = poses["akita_black_bowl_1_joint0"][:3].copy()
    plate = poses["plate_1_joint0"][:3].copy()
    grasp_shift = np.r_[bowl[:2] - xyz[c, :2], 0.0]
    place_shift = np.r_[plate[:2] - HELD_BOWL_OFFSET - xyz[r, :2], 0.0]
    xyz[: c + 1] += np.linspace(0.0, grasp_shift, c + 1)
    xyz[c + 1 : r + 1] += np.linspace(grasp_shift, place_shift, r - c + 1)[1:]
    xyz[r + 1 :] += place_shift
    xyz = np.r_[xyz[:c], np.repeat(xyz[c : c + 1], 15, axis=0), xyz[c:]]
    grip = np.ones(len(xyz))
    grip[c + 5 : r + 15] = -1.0
    time20 = np.arange(len(xyz)) / RECORD_HZ
    time100 = np.arange((len(xyz) - 1) * CONTROL_STEPS_PER_RECORD + 1) / 100
    xyz100 = PchipInterpolator(time20, xyz, axis=0)(time100)
    q100 = joint_path(model, arm_qpos, arm_dofs, site, xyz100)
    q_start, q100 = q100[0], q100[1:]

    mujoco.mj_resetData(model, data)
    for joint_name, pose in poses.items():
        joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        address = int(model.jnt_qposadr[joint])
        data.qpos[address : address + 7] = pose
    data.qpos[bowl_qpos : bowl_qpos + 3] = bowl
    data.qpos[plate_qpos : plate_qpos + 3] = plate
    data.qpos[arm_qpos] = q_start
    data.ctrl[arm_act] = data.qpos[arm_qpos]
    data.ctrl[grip_act] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(100):
        mujoco.mj_step(model, data)

    records = {name: [] for name in ("actions", "front_rgb", "eye_in_hand_rgb", "joint_states", "ee_pos", "ee_ori", "object_pos")}
    video_frames = []
    for step in range(len(xyz) - 1):
        target_slice = q100[step * CONTROL_STEPS_PER_RECORD : (step + 1) * CONTROL_STEPS_PER_RECORD]
        current = site_position(data, site)
        grip_action = float(grip[step])
        records["actions"].append(np.r_[np.clip((xyz[step + 1] - current) / MAX_POSITION_DELTA, -1, 1), np.zeros(3), grip_action])
        records["front_rgb"].append(render(renderer, data, "agentview"))
        records["eye_in_hand_rgb"].append(render(renderer, data, "wrist"))
        if video_renderer is not None:
            video_frames.append(render(video_renderer, data, "agentview"))
        records["joint_states"].append(np.r_[data.qpos[arm_qpos], data.qpos[grip_qpos]])
        records["ee_pos"].append(current)
        records["ee_ori"].append(Rotation.from_matrix(np.asarray(data.site_xmat[site]).reshape(3, 3)).as_rotvec())
        records["object_pos"].append(data.qpos[bowl_qpos : bowl_qpos + 3].copy())
        close_fraction = np.clip((step - c - 5) / 5, 0.0, 1.0)
        data.ctrl[grip_act] = -0.0255 * close_fraction if grip_action < 0 else 0.0
        for target_q in target_slice:
            data.ctrl[arm_act] = target_q
            for _ in range(5):
                mujoco.mj_step(model, data)

    final = data.qpos[bowl_qpos : bowl_qpos + 3].copy()
    plate_final = data.qpos[plate_qpos : plate_qpos + 3].copy()
    bowl_body = int(
        model.jnt_bodyid[
            object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "akita_black_bowl_1_joint0")
        ]
    )
    plate_body = int(
        model.jnt_bodyid[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_1_joint0")]
    )
    contact = bodies_touch(model, data, bowl_body, plate_body)
    success = (
        contact
        and final[2] >= plate_final[2]
        and np.linalg.norm(final[:2] - plate_final[:2]) < 0.03
    )
    return records, video_frames, bool(success), bowl, plate, final, plate_final, contact


def main(paths=None, limit=None):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in paths or sorted(SOURCE.glob("*.hdf5")):
        with h5py.File(path) as src, h5py.File(OUTPUT / path.name, "w") as dst:
            out = dst.create_group("data")
            info = json.loads(src["data"].attrs["problem_info"])
            out.attrs.update(num_demos=0, total=0, successes=0, problem_info=json.dumps(info), env_args=json.dumps({"robot": "NexArm", "action": "ee_delta", "control_freq": RECORD_HZ, "controller_freq": 100}))
            demos = sorted(src["data"], key=lambda name: int(name[5:]))[:limit]
            for name in demos:
                try:
                    source = src[f"data/{name}"]
                    model, poses = build_model(source)
                    data = mujoco.MjData(model)
                    renderer = mujoco.Renderer(model, IMAGE_SIZE, IMAGE_SIZE)
                    approval_video = name == "demo_0"
                    video_renderer = (
                        mujoco.Renderer(model, VIDEO_SIZE, VIDEO_SIZE)
                        if approval_video else None
                    )
                    arm_act, arm_qpos, arm_dofs = arm_addresses(model)
                    grip_act = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_control")
                    grip_joint = int(model.actuator_trnid[grip_act, 0])
                    ids = (
                        arm_act, arm_qpos, arm_dofs, grip_act,
                        int(model.jnt_qposadr[grip_joint]),
                        object_id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"),
                        int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "akita_black_bowl_1_joint0")]),
                        int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_1_joint0")]),
                    )
                    records, video_frames, success, bowl, plate, final, plate_final, contact = retarget(
                        source, model, data, renderer, video_renderer, ids, poses
                    )
                except RuntimeError as error:
                    print(path.stem, name, "ERROR", error, flush=True)
                    continue
                demo = out.create_group(name)
                n = len(records["actions"])
                demo.attrs.update(num_samples=n, success=success)
                for key, values in records.items():
                    demo.create_dataset(("obs/" if key != "actions" else "") + key, data=np.asarray(values), compression="gzip" if "rgb" in key else None)
                done = np.zeros(n, dtype=np.uint8)
                done[-1] = 1
                demo.create_dataset("dones", data=done)
                demo.create_dataset("rewards", data=done * success)
                out.attrs.modify("num_demos", int(out.attrs["num_demos"]) + 1)
                out.attrs.modify("total", int(out.attrs["total"]) + n)
                out.attrs.modify("successes", int(out.attrs["successes"]) + success)
                print(
                    path.stem, name, success, "bowl", bowl.round(3),
                    "plate", plate.round(3), "final", final.round(3),
                    "plate_final", plate_final.round(3), "contact", contact,
                    flush=True,
                )
                if approval_video:
                    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
                    video_path = VIDEO_DIR / f"{path.stem}_{name}_hq.mp4"
                    iio.imwrite(
                        video_path,
                        np.asarray(video_frames),
                        fps=RECORD_HZ,
                        codec="libx264",
                        quality=9,
                        pixelformat="yuv420p",
                    )
                renderer.close()
                if video_renderer is not None:
                    video_renderer.close()


if __name__ == "__main__":
    main(paths=[sorted(SOURCE.glob("*.hdf5"))[0]], limit=1)
