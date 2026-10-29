"""Run one self-contained reach/pick pipeline stage."""

from pathlib import Path
import argparse
import os
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT / "src")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from sam3_pick.paths import DATASETS

commands = {
    "fetch": [["-m", "sam3_pick.inference.fetch_weights"]],
    "voxel": [
        [
            "-m",
            "sam3_pick.data.voxel.generate",
            "--output",
            str(DATASETS / "reach_voxel_grid"),
            "--edge",
            ".02",
            "--workers",
            "8",
        ]
    ],
    "export": [
        [
            "-m",
            "sam3_pick.data.voxel.audit",
            str(DATASETS / "reach_voxel_grid"),
            "--workers",
            "8",
            "--replay",
            "16",
        ],
        [
            "-m",
            "sam3_pick.data.voxel.export_actions",
            str(DATASETS / "reach_voxel_grid"),
            str(DATASETS / "reach_voxel_grid/ee_delta_25hz_h25_s2"),
            "--workers",
            "8",
        ],
        [
            "-m",
            "sam3_pick.data.voxel.export_joints",
            "--chunks",
            str(DATASETS / "reach_voxel_grid/ee_delta_25hz_h25_s2"),
        ],
    ],
    "reach_metadata": [["-m", "sam3_pick.data.reach_metadata"]],
    "reach_clouds": [["-m", "sam3_pick.data.reach_clouds"]],
    "reach_train": [
        ["-m", "sam3_pick.training.reach"],
        ["-m", "sam3_pick.training.select_reach"],
    ],
    "pick_data": [["-m", "sam3_pick.data.pick_collect", "--workers", "8"]],
    "pick_render": [["-m", "sam3_pick.data.pick_render", "--workers", "8"]],
    "pick_audit": [["-m", "sam3_pick.data.pick_audit"]],
    "pick_labels": [["-m", "sam3_pick.data.prepare_pick"]],
    "pick_train": [["-m", "sam3_pick.training.pick"]],
    "evaluate": [["-m", "sam3_pick.evaluation.pick"]],
    "xy": [["-m", "sam3_pick.evaluation.prepare_xy"]],
    "ood": [["-m", "sam3_pick.evaluation.ood"]],
    "sam3_prepare": [["-m", "sam3_pick.inference.pipeline", "prepare"]],
    "sam3_run": [["-m", "sam3_pick.inference.pipeline", "run"]],
    "video": [["-m", "sam3_pick.evaluation.video"]],
    "test": [["-m", "unittest", "discover", "-s", "tests", "-v"]],
}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", *commands, "full"])
    stage = parser.parse_args().stage
    if stage != "test":
        from sam3_pick.runtime import main as prepare

        prepare()
    if stage == "prepare":
        return
    stages = list(commands)[: list(commands).index("video")] if stage == "full" else [stage]
    for name in stages:
        for args in commands[name]:
            subprocess.run([sys.executable, "-u", *args], cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
