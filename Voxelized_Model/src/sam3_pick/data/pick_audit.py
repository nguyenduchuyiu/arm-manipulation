"""Audit the rendered wide-pick dataset; optionally retire the explicitly replaced E102 data/cache."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import mujoco
import numpy as np
from tqdm import tqdm
from sam3_pick.simulation.scene import BASE, write_json
from sam3_pick.simulation.wrist_camera import INTRINSICS
from sam3_pick.data.pick_collect import DATA, ROOT


def arrays(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def replay(model, stored):
    data = mujoco.MjData(model)
    mujoco.mj_setState(
        model, data, stored["integration_state"][0], mujoco.mjtState.mjSTATE_INTEGRATION
    )
    qpos = qvel = 0.0
    for i in range(len(stored["qpos"])):
        data.ctrl[:] = stored["ctrl"][i]
        mujoco.mj_forward(model, data)
        qpos = max(qpos, float(np.abs(data.qpos - stored["qpos"][i]).max()))
        qvel = max(qvel, float(np.abs(data.qvel - stored["qvel"][i]).max()))
        if i + 1 < len(stored["qpos"]):
            mujoco.mj_step(model, data, 20)
    assert qpos < 1e-07 and qvel < 1e-06, (qpos, qvel)
    return dict(max_qpos_error=qpos, max_qvel_error=qvel)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA)
    args = p.parse_args()
    root = args.data
    config = json.loads((root / "config.json").read_text())
    assert config["status"] in ("render_complete", "complete")
    rows = [json.loads(s) for s in (root / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == config["expected_episodes"] and len({r["path"] for r in rows}) == len(rows)
    groups = Counter((r["evaluation_group"] for r in rows))
    if not config["pilot"]:
        assert groups == dict(train=2048, val=256, composition=1024, spatial=256, combined=128)
    assert config["views"] == ["front", "wrist"] and config["depth_views"] == ["front"]
    model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
    data = mujoco.MjData(model)
    assert model.ncam == 2
    jaws = {model.geom(f"link_6_{side}_jaw_collision_0").id for side in ("left", "right")}
    initial = {}
    checked_replays = []
    joints = []
    frames = 0
    start = time.perf_counter()
    for row in tqdm(rows, desc="Audit wide-pick media/calibration/contact/state", mininterval=5):
        directory = root / row["path"]
        stored = arrays(directory / "sim_state.npz")
        n = len(stored["qpos"])
        frames += n
        assert row["frames"] == n and row["views"] == ["front", "wrist"] and row["success"]
        key = (row["layout_id"], row["permutation"])
        vector = np.r_[stored["initial_qpos"], stored["initial_qvel"], stored["initial_ctrl"]]
        if key in initial:
            np.testing.assert_array_equal(vector, initial[key])
        else:
            initial[key] = vector
        if config["pilot"] or row["permutation"] == 0:
            checked_replays.append(dict(path=row["path"], **replay(model, stored)))
        render = json.loads((directory / "render_complete.json").read_text())
        assert (
            render["frames"] == n
            and render["rgb_frames_verified"] == n
            and render["depth_codec_lossless"]
        )
        assert render["quantization_max_mm"] <= 1.1 and (
            not (directory / "depth_wrist.mkv").exists()
        )
        c = arrays(directory / "calibration.npz")
        assert c["projection_model"].tolist() == ["pinhole", "fisheye_equidistant"]
        assert c["depth_views"].tolist() == ["front"]
        np.testing.assert_allclose(c["intrinsics"][1], INTRINSICS)
        np.testing.assert_array_equal(c["timestamp_s"], stored["timestamp_s"])
        np.testing.assert_allclose(
            c["camera_to_world"][..., :3, 3] - BASE,
            c["camera_to_base"][..., :3, 3],
            atol=1e-12,
        )
        for i in [0, n // 2, n - 1]:
            data.qpos[:] = stored["qpos"][i]
            mujoco.mj_forward(model, data)
            for vi, name in enumerate(["cam_front", "wrist"]):
                cid = model.camera(name).id
                np.testing.assert_allclose(
                    c["camera_to_world"][i, vi, :3, 3], data.cam_xpos[cid], atol=1e-12
                )
                np.testing.assert_allclose(
                    c["camera_to_world"][i, vi, :3, :3],
                    data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]),
                    atol=1e-12,
                )
        m = arrays(directory / "masks.npz")
        assert m["instance"].shape == (n, 2, 240, 320)
        prompt = arrays(directory / "prompt.npz")["query_mask"]
        np.testing.assert_array_equal(prompt, m["instance"][0, 0] == row["target_id"] + 1)
        assert prompt.sum() > 20
        sup = arrays(directory / "supervision.npz")
        assert sup["action_valid"][:-1].all() and (not sup["action_valid"][-1])
        labels = arrays(directory / "labels.npz")["entity_xyz_world_m"]
        t = row["target_id"]
        run = longest = 0
        body = model.body(row["target_object_id"]).id
        for i in range(max(0, n - 20), n):
            data.qpos[:] = stored["qpos"][i]
            data.qvel[:] = stored["qvel"][i]
            mujoco.mj_forward(model, data)
            touching = {
                j
                for contact in data.contact
                for j in jaws
                if contact.geom1 == j
                and model.geom_bodyid[contact.geom2] == body
                or (contact.geom2 == j and model.geom_bodyid[contact.geom1] == body)
            }
            run = run + 1 if data.xpos[body, 2] - labels[0, t, 2] > 0.04 and touching == jaws else 0
            longest = max(longest, run)
        assert longest >= 10 and labels[-1, t, 2] - labels[0, t, 2] > 0.04
        others = [j for j in range(4) if j != t]
        assert np.all(labels[:, others, 2] - labels[0, others, 2] <= 0.04)
        if row["split"] == "train":
            joints.append(arrays(directory / "observation.npz")["joint_position"][:-1])
    joint = np.concatenate(joints)
    write_json(
        root / "normalization.json",
        dict(
            applied=False,
            measured_joint_train_mean=joint.mean(0).tolist(),
            measured_joint_train_std=joint.std(0).tolist(),
            measured_joint_train_frames=len(joint),
        ),
    )
    result = dict(
        status="passed",
        episodes=len(rows),
        layouts=len({r["layout_id"] for r in rows}),
        frames=frames,
        groups=dict(groups),
        replay=checked_replays,
        all_terminal_bilateral_lifts_passed=True,
        initial_states_query_independent=True,
        elapsed_s=time.perf_counter() - start,
    )
    write_json(root / "audit.json", result)
    config["status"] = "complete"
    write_json(root / "config.json", config)
    with (root / "README.md").open("a") as f:
        f.write(
            f"\nActual status: COMPLETE; {len(rows)}episodes,{frames}frames; audit.json PASSED.\n"
        )
    shutil.copyfile(
        root / "audit.json",
        ROOT / ("full_audit.json" if not config["pilot"] else "preflight_rgbd_audit.json"),
    )
    print("AUDIT PASSED", len(rows), "episodes", frames, "frames", flush=True)


if __name__ == "__main__":
    main()
