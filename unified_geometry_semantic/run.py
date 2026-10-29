"""Run E178 generation, training, evaluation or the core regression suite."""

from pathlib import Path
import argparse
import json
import os
import shutil
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
commands = {
    "fetch": "data.fetch_weights",
    "generate": "data.generate",
    "benchmarks": "data.generate_benchmarks",
    "audit": "data.audit",
    "features": "data.feature_cache",
    "train": "training.train",
    "evaluate": "evaluation.canonical",
    "ood": "evaluation.ood",
    "controls": "evaluation.correspondence",
    "report": "evaluation.report",
}


def prepare():
    from unified_geometry.paths import ROOT

    ROOT.mkdir(parents=True, exist_ok=True)
    source = PROJECT / "configs/vision.json"
    target = ROOT / "feature_selection.json"
    if not target.exists():
        shutil.copy2(source, target)
    actual = json.loads(target.read_text())
    expected = json.loads(source.read_text())
    for key in ["selected", "revision", "weights_sha256"]:
        assert actual[key] == expected[key], f"Conflicting vision config: {key}"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", *commands, "test", "full"])
    stage = parser.parse_args().stage
    if stage == "test":
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=PROJECT,
            check=True,
        )
        return
    prepare()
    for name in list(commands) if stage == "full" else ([] if stage == "prepare" else [stage]):
        subprocess.run(
            [sys.executable, "-u", "-m", "unified_geometry." + commands[name]],
            cwd=PROJECT,
            check=True,
        )


if __name__ == "__main__":
    main()
