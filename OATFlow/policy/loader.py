"""Action-chunk DataLoader with LeRobot decoding and shared compressed MP4s."""
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets.video_utils import VideoDecoderCache, decode_video_frames_torchcodec


_decoders = VideoDecoderCache(max_size=8)


def cache_videos(root, rows, wrist, cache_root=Path("/dev/shm")):
    """Store compressed MP4s once in tmpfs so DataLoader workers share RAM files."""
    names = ("rgb.mp4", "wrist_rgb.mp4") if wrist else ("rgb.mp4",)
    paths = list(dict.fromkeys((root / row["path"] / name).resolve()
                              for row in rows for name in names))
    size = sum(path.stat().st_size for path in paths)
    if size > 16 * 2**30:
        raise MemoryError("compressed train videos exceed 16 GiB RAM budget")
    if not cache_root.is_dir() or shutil.disk_usage(cache_root).free < size + 4 * 2**30:
        raise MemoryError("video RAM filesystem needs compressed videos plus 4 GiB for DataLoader IPC")
    destination = Path(tempfile.mkdtemp(prefix="oatflow_", dir=cache_root))
    videos = {}
    for index, path in enumerate(tqdm(paths, desc="cache compressed videos", unit="video",
                                      mininterval=5, file=sys.stdout)):
        cached = destination / f"{index}.mp4"
        shutil.copyfile(path, cached)
        videos[path] = str(cached)
    return videos, size / 2**30


class LeRobotVideoFrames:
    """Adapt LeRobot's timestamp decoder to existing NumPy episode indexing."""

    def __init__(self, url, count, resolution):
        self.url = url
        self.count = count
        decoder = _decoders.get_decoder(url)
        meta = decoder.metadata
        if (meta.average_fps != 25 or len(decoder) != count or
                (meta.height, meta.width) != tuple(resolution)):
            raise ValueError("video metadata must match 25 Hz frame count and resolution")

    def __len__(self):
        return self.count

    def __getitem__(self, indices):
        indices = np.asarray(indices)
        if indices.dtype.kind not in "iu" or indices.ndim > 1 or not indices.size:
            raise ValueError("video indices must be integers or a nonempty 1D integer array")
        if np.any(indices < 0) or np.any(indices >= self.count):
            raise IndexError("video frame outside episode")
        frames = decode_video_frames_torchcodec(
            self.url, timestamps=(indices.reshape(-1) / 25).tolist(), tolerance_s=1e-3,
            decoder_cache=_decoders, return_uint8=True)
        rgb = frames.permute(0, 2, 3, 1).numpy()
        return rgb[0] if indices.ndim == 0 else rgb


def release_videos(videos):
    _decoders.clear()
    directory = Path(next(iter(videos.values()))).parent
    if any(Path(path).parent != directory for path in videos.values()):
        raise ValueError("video cache must have one dedicated directory")
    shutil.rmtree(directory)


def prepare_samples(episodes, items, wrist):
    from OATFlow.policy.data import training_clip

    def decode_episode(episode_id):
        episode = episodes[episode_id][1]
        if not isinstance(episode["rgb"], LeRobotVideoFrames):
            return episode_id, episode
        ends = np.unique([end for index, end, _active in items if index == episode_id])
        indices = np.unique(np.rint(np.linspace(0, ends, 30, axis=1)).astype(np.int64))
        episode = episode.copy()
        # Decode each episode's entire batch in one LeRobot call, sharing GOP work.
        episode["rgb"] = episode["rgb"][indices]
        episode["rgb_frames"] = indices
        if wrist:
            episode["wrist_rgb"] = episode["wrist_rgb"][ends]
            episode["wrist_frames"] = ends
        return episode_id, episode

    ids = sorted({episode_id for episode_id, _end, _active in items})
    decoded = dict(map(decode_episode, ids))
    return [tuple(tensor[0] for tensor in training_clip(
        decoded[episode_id], end, active, "cpu", include_wrist=wrist))
        for episode_id, end, active in items]


class ActionChunkDataset(Dataset):
    """LeRobot decoding with task-specific history and fixed-anchor EE targets."""

    def __init__(self, root, rows, samples, videos, flow_only, wrist):
        self.root, self.rows, self.samples = root, rows, samples
        self.videos, self.flow_only, self.wrist = videos, flow_only, wrist

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        from OATFlow.policy.data import load_joint_episode, load_policy_episode
        items = [self.samples[index] for index in indices]
        episodes = {}
        load = load_policy_episode if self.flow_only else load_joint_episode
        for episode_id in sorted({item[0] for item in items}):
            row = self.rows[episode_id]
            episode = load(self.root / row["path"], require_wrist=self.wrist,
                           training_cache=True, videos=self.videos)
            episodes[episode_id] = row, episode
        return prepare_samples(episodes, items, self.wrist)


def make_dataloader(dataset, batches, seed, pin_memory, num_workers=2, prefetch_factor=2):
    # These are the standard DataLoader controls used by LeRobot's trainer.
    return DataLoader(dataset, batch_sampler=batches, num_workers=num_workers,
                      prefetch_factor=prefetch_factor, persistent_workers=True, pin_memory=pin_memory,
                      multiprocessing_context="spawn",
                      generator=torch.Generator().manual_seed(seed))


def training_batches(plans, epochs, batch_size, rng, smoke_steps=0):
    """Retain the existing cluster/phase shuffle and short final batches."""
    samples, groups = [], []
    episode_offset = 0
    for cluster in plans:
        start = len(samples)
        samples.extend((episode_offset + episode_id, end, active)
                       for episode_id, plan in enumerate(cluster) for end, active in plan)
        groups.append(list(range(start, len(samples))))
        episode_offset += len(cluster)
    batches, epoch_ends = [], []
    for _epoch in range(epochs):
        for group in groups:
            cluster_batches = []
            for phase in (False, True):
                phase_items = [index for index in group if samples[index][2] == phase]
                rng.shuffle(phase_items)
                phase_batches = [phase_items[start:start+batch_size]
                                 for start in range(0, len(phase_items), batch_size)]
                if smoke_steps:
                    if len(phase_batches) < 2:
                        raise ValueError("smoke requires two batches per phase; increase cluster size")
                    phase_batches = phase_batches[:2]
                cluster_batches.extend(phase_batches)
            rng.shuffle(cluster_batches)
            batches.extend(cluster_batches)
        epoch_ends.append(len(batches))
    return samples, batches, epoch_ends
