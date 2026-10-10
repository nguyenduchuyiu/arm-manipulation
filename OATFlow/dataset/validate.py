"""Check aligned RGB, context labels and the short physical expert trajectory."""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from OATFlow.task import TARGETS, TASKS


def validate(directory):
    meta = json.loads((directory / "episode.json").read_text())
    inputs = json.loads((directory / "input.json").read_text())
    n = meta["frames"]
    if (not meta["success"] or meta["fps"] != 25 or meta["resolution"] != [320, 320]
            or meta["action_representation"] != "absolute_joint"
            or meta["objective"] != "cover_drop_then_target_lift"):
        raise ValueError("expected successful 25Hz RGB absolute-joint lift episode")
    if set(inputs) != {"rgb_video", "wrist_rgb_video", "observation", "query_rgb", "decision_frames"}:
        raise ValueError("unexpected model input fields")
    if inputs["decision_frames"] != meta["decision_frames"] or not (directory / meta["query_rgb"]).is_file():
        raise ValueError("missing query reference or inconsistent decisions")
    if meta["object_to_occluder"][meta["target_object_id"]] != meta["selected_cover"]:
        raise ValueError("target and selected cover disagree")
    task = TASKS[meta["task_id"]]
    if (meta["target_id"] != TARGETS.index(meta["target_object_id"])
            or meta["target_object_id"] not in task["objects"]
            or set(meta["task"]["objects"]) != set(task["objects"])
            or meta["task"]["cover_side"] != task["cover_side"]
            or meta["correct_cover_id"] != task["cover_side"]
            or meta["cover_ids_at_t_occ"][task["cover_side"]] != meta["selected_cover"]):
        raise ValueError("post-shuffle task/target IDs disagree with episode metadata")
    with np.load(directory / "observation.npz") as obs, np.load(directory / "supervision.npz") as gt:
        if set(obs.files) != {"joint_position", "timestamp_s"} or not np.allclose(obs["timestamp_s"], np.arange(n) / 25):
            raise ValueError("observations are not aligned RGB/joint frames")
        for array in (obs["joint_position"], gt["expert_action"]):
            if array.shape != (n, 6) or not np.isfinite(array).all() or np.any(array[:, :5] < -1) or np.any(array > 1) or np.any(array[:, 5] < 0):
                raise ValueError("invalid absolute joints or gripper")
        occ, obj = (meta["decision_frames"][k] for k in ("t_occ", "t_obj"))
        cover_grasp, target_grasp = (meta["grasp_frames"][k] for k in ("cover", "object"))
        valid, phases = gt["action_valid"], gt["phase"]
        if (valid.shape != (n,) or not 0 < occ < cover_grasp < obj < target_grasp < n - 1
                or valid[:occ].any() or not valid[occ:-1].all() or valid[-1]):
            raise ValueError("context/action boundaries or grasp order differ")
        events = gt["grasp_event"]
        if not np.array_equal(np.flatnonzero(events), [cover_grasp, target_grasp]) or events[cover_grasp] != 1 or events[target_grasp] != 2:
            raise ValueError("grasp events disagree with frames")
        for body, stages in ((meta["selected_cover"], ("transport", "release", "settle")),
                             (meta["target_object_id"], ("approach", "engage", "close", "lift", "hold"))):
            if any(not np.any(phases == body + "_" + stage) for stage in stages):
                raise ValueError("missing expert phase")
        if any(np.char.endswith(phases, "_" + stage).any() for stage in ("lower", "retreat", "home")):
            raise ValueError("obsolete long expert phases")
        if any(np.any(phases == meta["target_object_id"] + "_" + stage) for stage in ("transport", "release", "settle")):
            raise ValueError("target placement in lift-only task")
        if meta["bilateral_lift_streak_frames"][meta["target_object_id"]] < 10:
            raise ValueError("target lift lacks ten bilateral-contact frames")
        intervals = meta["phase_intervals"]
        cursor = 0
        for item in intervals:
            start, end = item["start_frame"], item["end_frame_exclusive"]
            if start != cursor or not start < end <= n or not np.all(phases[start:end] == item["phase"]):
                raise ValueError("phase intervals do not align")
            cursor = end
        if cursor != n:
            raise ValueError("phase intervals do not cover the episode")
        if meta["collection_mode"] == "context":
            expected_split = "test" if meta["seed"] % 10 == 9 else "val" if meta["seed"] % 10 == 8 else "train"
            plan = meta["context_plan"]
            expected_occ = plan["reveal_frames"] + plan["occlude_frames"] + plan["hold_frames"] + sum(s["frames"] + s["pause_frames"] for s in plan["swaps"])
            if meta["split"] != expected_split or occ != expected_occ or len(plan["swaps"]) != meta["swaps"]:
                raise ValueError("scene split or context plan differs")
            hidden = meta["semantic_transition_frames"]["fully_occluded"]
            released = meta["semantic_transition_frames"]["cover_released"]
            complete = meta["semantic_transition_frames"]["task_complete"]
            if not 0 < hidden < occ < cover_grasp < released == obj < target_grasp < complete == n - 1:
                raise ValueError("semantic event order differs")
            entity = np.full(n, 2, dtype=np.uint8)
            entity[hidden:released], entity[complete:] = 1, 0
            masks = gt["mask"]
            if masks.shape != (n, 240, 320) or not np.array_equal(gt["semantic_entity"], entity) or np.any((masks != 0) & (masks != entity[:, None, None])):
                raise ValueError("relevance masks disagree with task stages")
            with np.load(directory / "tcow_labels.npz") as labels:
                amodal = labels["mask"]
                if amodal.shape != (n, 3, 240, 320) or amodal.dtype != np.bool_:
                    raise ValueError("TCOW labels must be three binary mask channels")
                if list(labels["channel_names"]) != ["target_amodal", "frontmost_occluder", "outermost_container"] or not np.allclose(labels["timestamp_s"], obs["timestamp_s"]):
                    raise ValueError("TCOW channels/timestamps differ")
                if not amodal[occ, 2].any() or not amodal[obj, 0].any():
                    raise ValueError("missing covered or exposed target masks")
            query = iio.imread(directory / "query_mask.png") > 0
            if not query.any() or not np.array_equal(query, masks[0] == 2):
                raise ValueError("query mask differs from the first visible target")
    with np.load(directory / "sim_state.npz") as state:
        if state["qpos"].shape[0] != n or state["qvel"].shape[0] != n:
            raise ValueError("simulation states disagree with video length")
        if not np.array_equal(state["initial_qpos"], state["qpos"][occ]):
            raise ValueError("expert initial state differs from post-shuffle t_occ")
    for name in ("rgb.mp4", "wrist_rgb.mp4"):
        video = iio.immeta(directory / name, plugin="pyav")
        shape = iio.improps(directory / name, plugin="pyav", index=0).shape
        if video["fps"] != 25 or abs(video["duration"] * 25 - n) > 1 or shape != (320, 320, 3):
            raise ValueError("camera video disagrees with observations")
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    for path in sorted(parser.parse_args().dataset.rglob("episode.json")):
        print(path.parent, validate(path.parent), "aligned frames", flush=True)
