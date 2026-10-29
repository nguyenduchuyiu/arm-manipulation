"""Prepare local output metadata without copying data or pretrained models."""

import shutil
from sam3_pick.paths import PROJECT, ASSETS, OUTPUTS


def main():
    assert shutil.which("ffmpeg"), "Install ffmpeg on PATH"
    reference = OUTPUTS / "e118_reach/metadata"
    reference.mkdir(parents=True, exist_ok=True)
    if not (reference / "config.json").exists():
        shutil.copy2(ASSETS / "reach_metadata/config.json", reference / "config.json")
    reach = OUTPUTS / "e118_reach"
    if not (reach / "README.md").exists():
        (reach / "README.md").write_text(
            "# E118 reach\n\nTrain120k/B256/BF16 from voxel clouds. Actual configuration and status are saved beside each output.\n"
        )
    (OUTPUTS / "data_collection").mkdir(parents=True, exist_ok=True)
    notes = OUTPUTS / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    template = (PROJECT / "configs/run_readme.md").read_text()
    for name in [
        "e117_README",
        "e122_xy_README",
        "e131_README",
        "e131_ood_README",
        "e131_test_README",
        "e131_xy_README",
        "e133_README",
        "e133_video_README",
        "REPORT",
    ]:
        if not (notes / (name + ".md")).exists():
            (notes / (name + ".md")).write_text("# " + name + "\n\n" + template)


if __name__ == "__main__":
    main()
