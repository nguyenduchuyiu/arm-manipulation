"""Full train/validation/test split for point-cloud picking."""

from sam3_pick.paths import PICK_DATA as DATA
from sam3_pick.paths import DATASETS
from collections import Counter
import json
from sam3_pick.data.pick_labels import write
from sam3_pick.utils.hashing import sha256_file as sha

ROOT = DATASETS / "pick_cloud_labels"


def partition():
    rows = [json.loads(s) for s in (DATA / "manifest.jsonl").read_text().splitlines()]
    manifests = {k: [] for k in ("train", "val", "test")}
    for row in rows:
        group = row["evaluation_group"]
        split = group if group in ("train", "val") else "test"
        manifests[split].append(dict(row, split=split))
    assert [len(manifests[k]) for k in ("train", "val", "test")] == [2048, 256, 1408]
    assert {r["layout_id"] for r in manifests["train"]} == set(range(32))
    assert Counter((r["target_id"] for r in manifests["train"])) == {k: 512 for k in range(4)}
    assert Counter((r["evaluation_group"] for r in manifests["test"])) == dict(
        composition=1024, spatial=256, combined=128
    )
    path_sets = [{r["path"] for r in manifests[k]} for k in ("train", "val", "test")]
    assert all((not path_sets[i] & path_sets[j] for i in range(3) for j in range(i + 1, 3)))
    for split, rows in manifests.items():
        (ROOT / f"{split}_manifest.jsonl").write_text("".join((json.dumps(r) + "\n" for r in rows)))
    write(
        ROOT / "partitions.json",
        dict(
            status="complete",
            source_manifest_sha256=sha(DATA / "manifest.jsonl"),
            train_layouts=list(range(32)),
            train_episodes=2048,
            val_episodes=256,
            test_episodes=1408,
            test_groups=dict(Counter((r["evaluation_group"] for r in manifests["test"]))),
            selection="original E117 full split; no original test/validation episode enters training",
        ),
    )
    return manifests
