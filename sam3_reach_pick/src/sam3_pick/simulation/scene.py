"""Generate e36-style context + expert with balanced unseen permutations/layouts."""

from collections import Counter
from itertools import permutations
import json
import mujoco
import numpy as np
from sam3_pick.simulation.pose import ARM_JOINTS, rotation_6d

NAMES = ("RedCube", "BlueCube", "GreenCube", "YellowCube", "cover_a", "cover_b")
BASE = np.array([0.539369022999, 0.0639727628473, 0.0])
PERMS = tuple(permutations(range(4)))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def permutation_split():
    remaining = set(PERMS)
    orbits = []
    while remaining:
        p = min(remaining)
        orbit = {p[k:] + p[:k] for k in range(4)}
        orbits.append(orbit)
        remaining -= orbit
    held = set.union(orbits[1], orbits[4])
    unseen = [i for i, p in enumerate(PERMS) if p in held]
    seen = [i for i in range(24) if i not in unseen]
    for ids in (seen, unseen):
        counts = Counter(((color, slot) for i in ids for slot, color in enumerate(PERMS[i])))
        assert len(counts) == 16 and set(counts.values()) == {len(ids) // 4}
        assert all((PERMS.index(PERMS[i][2:] + PERMS[i][:2]) in ids for i in ids))
    return (seen, unseen)


SEEN, UNSEEN = permutation_split()


def split(group, permutation):
    if group < 32:
        return ("train", "train") if permutation in SEEN else ("test", "composition")
    if group < 36:
        assert permutation in SEEN
        return ("val", "val")
    return ("test", "spatial" if permutation in SEEN else "combined")


class Scene:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.target_qadr = {n: int(model.joint(n + "_joint").qposadr[0]) for n in NAMES[:4]}
        self.cover_qadr = {n: int(model.joint(n + "_joint").qposadr[0]) for n in NAMES[4:]}
        self.qadr = np.array(
            [model.jnt_qposadr[model.joint(n).id] for n in (*ARM_JOINTS, "right_jaw_slide_joint")]
        )
        self.dofs = np.array(
            [model.jnt_dofadr[model.joint(n).id] for n in (*ARM_JOINTS, "right_jaw_slide_joint")]
        )
        self.site = model.site("grasp_site").id

    def target_positions(self):
        return {n: self.data.xpos[self.model.body(n).id].copy() for n in NAMES[:4]}

    def query(self, target):
        self.query_target = target

    def choose_cover(self, cover):
        self.selected_cover = cover
        assert cover == self.assignment[self.query_target]


def snapshot(scene, store, phase, valid):
    model, data = (scene.model, scene.data)
    mujoco.mj_forward(model, data)
    for key, value in (
        ("qpos", data.qpos),
        ("qvel", data.qvel),
        ("ctrl", data.ctrl),
        ("joint_position", data.qpos[scene.qadr]),
        ("joint_velocity", data.qvel[scene.dofs]),
        ("ee_position_base", data.site_xpos[scene.site] - BASE),
        ("ee_rotation_6d", rotation_6d(data.site_xmat[scene.site].reshape(3, 3))),
        ("entity_xyz_world", np.stack([data.xpos[model.body(n).id] for n in NAMES])),
        (
            "entity_rotation_world",
            np.stack([data.xmat[model.body(n).id].reshape(3, 3) for n in NAMES]),
        ),
    ):
        store.setdefault(key, []).append(value.copy())
    store.setdefault("phase", []).append(phase)
    store.setdefault("action_valid", []).append(valid)


def save_segment(directory, store, offset=0, initial=None):
    directory.mkdir(parents=True, exist_ok=True)
    values = {k: np.array(v) for k, v in store.items()}
    n = len(values["phase"])
    times = (offset + np.arange(n)) / 25
    state = dict(qpos=values["qpos"], qvel=values["qvel"], ctrl=values["ctrl"], timestamp_s=times)
    if initial is not None:
        state.update(initial_qpos=initial[0], initial_qvel=initial[1], initial_ctrl=initial[2])
    np.savez_compressed(directory / "sim_state.npz", **state)
    np.savez_compressed(
        directory / "observation.npz",
        joint_position=values["joint_position"],
        joint_velocity=values["joint_velocity"],
        ee_position_base_m=values["ee_position_base"],
        ee_rotation_6d=values["ee_rotation_6d"],
        timestamp_s=times,
    )
    np.savez_compressed(
        directory / "supervision.npz",
        joint_command=values["ctrl"],
        phase=values["phase"],
        action_valid=values["action_valid"],
        timestamp_s=times,
    )
    np.savez_compressed(
        directory / "labels.npz",
        entity_xyz_world_m=values["entity_xyz_world"],
        entity_xyz_base_m=values["entity_xyz_world"] - BASE,
        entity_rotation_world=values["entity_rotation_world"],
        entity_names=np.array(NAMES),
        timestamp_s=times,
    )
    return n
