"""Build E131 cloud/action labels directly, without RGB/ViT features or E122 training."""

from sam3_pick.paths import PICK_DATA as DATA
import time
import av
import numpy as np
from tqdm import tqdm
from sam3_pick.data.pick_split import partition
from sam3_pick.data.pick_labels import jobs_and_labels, write
from sam3_pick.data.target_cloud import CACHE, SOURCE, extract_cloud
from sam3_pick.utils.hashing import sha256_file as sha


def main():
    CACHE.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifests = partition()
    jobs, labels = jobs_and_labels(manifests)
    n = len(labels["is_train"])
    clouds = np.zeros((n, 128, 3), np.float32)
    good = np.zeros(n, bool)
    pixels = np.zeros(n, np.int32)
    seen = np.zeros(n, bool)
    for row, starts, offset in tqdm(jobs, desc="Depth/mask to E131 cloud labels", mininterval=5):
        directory = DATA / row["path"]
        with np.load(directory / "masks.npz") as z:
            masks = z["instance"][:, 0] == row["target_id"] + 1
        with np.load(directory / "calibration.npz") as z:
            K = z["intrinsics"][0]
            Ts = z["camera_to_base"][:, 0]
            unit = float(z["depth_unit_m"])
        rng = np.random.default_rng(
            np.random.SeedSequence([119, row["layout_id"], row["permutation"], row["target_id"]])
        )
        wanted = {int(frame): offset + i for i, frame in enumerate(starts)}
        with av.open(str(directory / "depth_front.mkv")) as stream:
            stream.streams.video[0].thread_count = 1
            for frame, item in enumerate(stream.decode(video=0)):
                if frame not in wanted:
                    continue
                i = wanted[frame]
                clouds[i], good[i], pixels[i] = extract_cloud(
                    item.to_ndarray(format="gray16le"),
                    masks[frame],
                    K,
                    Ts[frame],
                    rng,
                    unit,
                )
                seen[i] = True
    assert seen.all() and good.all() and np.isfinite(clouds).all()
    labels.update(cloud=clouds, cloud_valid=good)
    np.savez_compressed(CACHE / "labels.npz", **labels, visible_pixels=pixels)
    write(
        CACHE / "cache_jobs.json",
        [dict(path=r["path"], starts=s.tolist(), offset=o) for r, s, o in jobs],
    )
    train = np.flatnonzero(labels["is_train"])
    val = np.random.default_rng(102).choice(np.flatnonzero(~labels["is_train"]), 512, replace=False)
    (CACHE / "train").mkdir()
    np.savez(CACHE / "train/indices.npz", train=train, validation=val)
    write(
        CACHE / "cache_config.json",
        dict(
            status="complete",
            frames=n,
            train_frames=len(train),
            val_frames=int((~labels["is_train"]).sum()),
            source_manifest_sha256=sha(DATA / "manifest.jsonl"),
            source_checkpoint_sha256=sha(SOURCE),
            RGB_or_ViT_cache=False,
            elapsed_s=time.perf_counter() - started,
        ),
    )
    (CACHE / "README.md").write_text(
        "# Point-cloud pick cache\n\nGenerated directly from front optical-Z depth, GT target masks, calibration and expert actions. E131 training only; GT masks are replaced by SAM3 at E133 inference. No RGB/ViT features or E122 policy. Status: complete.\n"
    )


if __name__ == "__main__":
    main()
