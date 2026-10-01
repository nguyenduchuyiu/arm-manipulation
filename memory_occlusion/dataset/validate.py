"""Check full-video alignment and event-driven dense relevance masks."""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def validate(directory):
    metadata = json.loads((directory / "episode.json").read_text())
    model_input = json.loads((directory / "input.json").read_text())
    if set(model_input) != {"rgb_video", "observation", "query_rgb", "decision_frames"}:
        raise ValueError("model input contains unexpected metadata")
    with np.load(directory / "observation.npz") as obs, np.load(directory / "supervision.npz") as gt:
        n = metadata["frames"]
        expected_split = "test" if metadata["seed"] % 10 == 9 else "val" if metadata["seed"] % 10 == 8 else "train"
        if metadata["split"] != expected_split:
            raise ValueError("split does not match layout seed")
        if set(metadata["object_to_occluder"]) != set(metadata["object_ids"]):
            raise ValueError("object assignment is incomplete")
        if set(metadata["object_to_occluder"].values()) != set(metadata["occluder_ids"]):
            raise ValueError("occluder IDs do not match assignments")
        if metadata["object_to_occluder"][metadata["target_object_id"]] != metadata["correct_cover_body"]:
            raise ValueError("target assignment disagrees with correct cover")
        if metadata["cover_ids_at_t_occ"][metadata["correct_cover_id"]] != metadata["correct_cover_body"]:
            raise ValueError("cover ID disagrees with left/right order")
        if not metadata["success"] or metadata["fps"] != 25:
            raise ValueError("unsuccessful episode or wrong FPS")
        height, width = metadata["resolution"]
        if obs["depth_m"].shape != (n, height, width) or gt["mask"].shape != (n, height, width):
            raise ValueError("depth/mask shape mismatch")
        if not np.isfinite(obs["depth_m"]).all() or not (obs["depth_m"] > 0).all():
            raise ValueError("invalid metric depth")
        if not np.allclose(np.diff(obs["timestamp_s"]), .04, atol=1e-9):
            raise ValueError("timestamps are not 25 Hz")
        for array in (obs["joint_position"], gt["expert_action"]):
            if array.shape != (n, 6) or not np.isfinite(array).all():
                raise ValueError("invalid proprio/action shape")
            if np.any(array[:, :5] < -1) or np.any(array > 1) or np.any(array[:, 5] < 0):
                raise ValueError("normalization out of range")
        occ, obj = (metadata["decision_frames"][key] for key in ("t_occ", "t_obj"))
        if "context_plan" in metadata:
            plan = metadata["context_plan"]
            if len(plan["swaps"]) != metadata["swaps"] or occ != (
                    plan["reveal_frames"] + plan["occlude_frames"] +
                    sum(path["frames"] + path["pause_frames"] for path in plan["swaps"]) +
                    plan["hold_frames"]):
                raise ValueError("context plan differs from recorded phases")
        cover_grasp, object_grasp = (metadata["grasp_frames"][key]
                                     for key in ("cover", "object"))
        mask, valid = gt["mask"], gt["action_valid"]
        transition = metadata["semantic_transition_frames"]
        hidden, cover_released, object_released = (transition[key] for key in
            ("fully_occluded", "cover_released", "object_released"))
        if (not 0 < occ < cover_grasp < obj < object_grasp < n - 1
                or valid[:occ].any() or not valid[occ:-1].all() or valid[-1]):
            raise ValueError("bad decision boundaries/action validity")
        if not 0 < hidden < occ < cover_grasp < cover_released <= obj < object_grasp < object_released < n:
            raise ValueError("bad semantic transition order")
        expected = np.full(n, 2, dtype=np.uint8)
        expected[hidden:cover_released] = 1
        expected[object_released:] = 0
        if not np.array_equal(gt["semantic_entity"], expected):
            raise ValueError("semantic entity does not follow task events")
        if np.any((mask != 0) & (mask != expected[:, None, None])):
            raise ValueError("mask assigned to the wrong entity")
        if mask[0].sum() == 0 or mask[hidden].sum() == 0 or mask[cover_released].sum() == 0:
            raise ValueError("missing mask at semantic transition")
        if not (mask[occ] == 1).any() or not (mask[obj] == 2).any():
            raise ValueError("decision mask is empty")
        for key, frame in (("cover", cover_grasp), ("object", object_grasp)):
            if not (directory / f"t_{key}_grasp.png").is_file():
                raise ValueError(f"missing {key} grasp snapshot")
        event = gt["grasp_event"]
        if event.shape != (n,) or not np.array_equal(np.flatnonzero(event),
                                                     [cover_grasp, object_grasp]):
            raise ValueError("grasp events do not align with video")
        if event[cover_grasp] != 1 or event[object_grasp] != 2:
            raise ValueError("grasp event IDs are incorrect")
        if metadata["tcow_labels"]:
            with np.load(directory / metadata["tcow_labels"]) as labels:
                masks = labels["mask"]
                if masks.shape != (n, 3, height, width) or not np.isin(masks, (0, 1)).all():
                    raise ValueError("invalid TCOW mask shape or values")
                if list(labels["channel_names"]) != ["target_amodal", "frontmost_occluder",
                                                      "outermost_container"]:
                    raise ValueError("wrong TCOW channel order")
                if not np.allclose(labels["timestamp_s"], obs["timestamp_s"]):
                    raise ValueError("TCOW mask timestamps differ from RGB")
                if np.any((gt["mask"][0] == 2) & ~masks[0, 0]):
                    raise ValueError("visible query target is outside its amodal mask")
                if not masks[occ, 2].any() or not masks[cover_released, 0].any():
                    raise ValueError("missing container or released target mask")
                query_path = directory / "query_mask.png"
                if not query_path.is_file() or not np.array_equal(
                        iio.imread(query_path) > 0, gt["mask"][0] == 2):
                    raise ValueError("TCOW query mask differs from visible first-frame target")
        phase_intervals = metadata["phase_intervals"]
        if phase_intervals[0]["start_frame"] != 0 or phase_intervals[-1]["end_frame_exclusive"] != n:
            raise ValueError("phase intervals do not span the video")
        for interval in phase_intervals:
            start, end = interval["start_frame"], interval["end_frame_exclusive"]
            if not 0 <= start < end <= n or not np.all(gt["phase"][start:end] == interval["phase"]):
                raise ValueError("phase interval disagrees with frame labels")
            if interval["start_s"] != start / 25 or interval["end_s"] != end / 25:
                raise ValueError("phase interval timestamps disagree with frames")
        if any(phase_intervals[i]["end_frame_exclusive"] != phase_intervals[i + 1]["start_frame"]
               for i in range(len(phase_intervals) - 1)):
            raise ValueError("phase intervals have a gap")
        if any(metadata["semantic_transition_times_s"][key] != frame / 25
               for key, frame in transition.items()):
            raise ValueError("semantic event times disagree with frames")
    with np.load(directory / "debug_poses.npz") as debug:
        if debug["poses"].shape != (n, 7 * len(metadata["pose_body_order"])):
            raise ValueError("debug pose shape does not match entity IDs")
        if list(debug["body_names"]) != metadata["pose_body_order"]:
            raise ValueError("debug pose body order does not match metadata")
        if not np.allclose(debug["timestamp_s"], np.arange(n) / 25):
            raise ValueError("debug pose timestamps do not align")
    video = iio.immeta(directory / "rgb.mp4", plugin="pyav")
    if video["fps"] != 25 or abs(video["duration"] * 25 - n) > 1:
        raise ValueError("RGB video does not align with observations")
    if not (directory / metadata["query_rgb"]).is_file():
        raise ValueError("missing query image")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("outputs/memory_occlusion/samples"))
    root = parser.parse_args().dataset
    episodes = sorted(root.glob("episode_*/episode.json"))
    if not episodes:
        raise ValueError("no completed episodes")
    for path in episodes:
        print(path.parent.name, validate(path.parent), "aligned frames", flush=True)


if __name__ == "__main__":
    main()
