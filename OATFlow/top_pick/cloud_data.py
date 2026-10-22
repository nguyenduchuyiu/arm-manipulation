"""Balanced layout partitions and synchronized front RGB/depth target-cloud cache."""
from collections import Counter
import json

import av
import numpy as np
from scipy.ndimage import binary_erosion,median_filter
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

from OATFlow.simulation.pose import rotation_matrix, rotation_6d
from OATFlow.top_pick.cloud import ROOT, DATA
from OATFlow.top_pick.features import sha


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def partition():
    rows = [json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    parts = np.random.default_rng(119).permutation(32).reshape(8, 4)
    selected = set(parts[0].tolist())
    manifests = {k: [] for k in ('train', 'val', 'test')}
    for row in rows:
        layout = row['layout_id']; seen = row['evaluation_group'] in ('train', 'val', 'spatial')
        if 32 <= layout < 36:
            split, group = 'val', 'val'
        elif layout in selected:
            split, group = ('train', 'train') if seen else ('test', 'composition')
        else:
            split, group = 'test', 'spatial' if seen else 'combined'
        manifests[split].append(dict(row, original_evaluation_group=row['evaluation_group'],
                                    split=split, evaluation_group=group))
    assert [len(manifests[k]) for k in ('train', 'val', 'test')] == [256, 256, 3200]
    assert Counter(r['evaluation_group'] for r in manifests['test']) == dict(spatial=2048, composition=128, combined=1024)
    train = manifests['train']
    assert Counter(r['target_id'] for r in train) == {k: 64 for k in range(4)}
    assert len({r['permutation'] for r in train}) == 16
    for split, current in manifests.items():
        (ROOT/f'{split}_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in current))
    for number, layouts in enumerate(parts):
        current = [r for r in rows if r['evaluation_group'] == 'train' and r['layout_id'] in layouts]
        assert len(current) == 256
        (ROOT/f'part_{number+1:02d}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in current))
    config = dict(status='complete', seed=119, parts=parts.tolist(), selected_part=1,
        train_layouts=sorted(selected), train_episodes=256, val_episodes=256, test_episodes=3200,
        test_groups=dict(Counter(r['evaluation_group'] for r in manifests['test'])),
        source_manifest_sha256=sha(DATA/'manifest.jsonl'),
        balance='4whole layouts per part,16seen permutations,4queries;64episodes per target')
    write(ROOT/'partitions.json', config)
    return manifests


def action_labels(row, starts):
    directory = DATA/row['path']
    with np.load(directory/'observation.npz') as z:
        joints = z['joint_position']; ee = z['ee_position_base_m']; rotation = rotation_matrix(z['ee_rotation_6d'])
        timestamps = z['timestamp_s']
    with np.load(directory/'supervision.npz') as z:
        command = z['joint_command']; valid = z['action_valid']
        np.testing.assert_array_equal(timestamps, z['timestamp_s'])
    end = len(valid)-1
    assert valid[:end].all() and not valid[end]
    indices = starts[:, None]+np.arange(25); future = np.minimum(indices+1, end)
    actions = np.empty((len(starts), 25, 10), np.float32)
    actions[..., :3] = ee[future]-ee[starts, None]
    relative = rotation[future] @ rotation[starts, None].swapaxes(-1, -2)
    actions[..., 3:9] = rotation_6d(relative)
    actions[..., 9] = command[np.minimum(indices, end-1), 5]
    np.testing.assert_allclose(rotation_matrix(actions[..., 3:9])@rotation[starts, None],
                               rotation[future], rtol=0, atol=2e-6)
    return dict(state=joints[starts].astype(np.float32), action=actions, mask=indices < end,
        target=np.full(len(starts), row['target_id'], np.int64), raw_frame=starts,
        is_train=np.full(len(starts), row['split'] == 'train'))


def jobs_and_labels(manifests):
    rng = np.random.default_rng(102); jobs = []; values = {}; offset = 0
    for row in tqdm(manifests['train']+manifests['val'], desc='Align subset EE labels', mininterval=5):
        n = row['frames']-1
        starts = np.arange(n) if row['split'] == 'train' else np.sort(rng.choice(n, min(8, n), replace=False))
        labels = action_labels(row, starts)
        for k, v in labels.items(): values.setdefault(k, []).append(v)
        jobs.append((row, starts, offset)); offset += len(starts)
    return jobs, {k: np.concatenate(v) for k, v in values.items()}


class Clips(IterableDataset):
    def __init__(self, jobs, include_depth=False, clean_depth_labels=False):
        self.jobs = jobs
        self.include_depth = include_depth
        self.clean_depth_labels = clean_depth_labels

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        jobs = self.jobs[worker.id::worker.num_workers] if worker else self.jobs
        for row, starts, offset in jobs:
            directory = DATA/row['path']
            wanted = {int(frame): offset+j for j, frame in enumerate(starts)}
            with np.load(directory/'masks.npz') as z:
                masks = z['instance'][:, 0] == row['target_id']+1
                timestamps = z['timestamp_s']
            with np.load(directory/'calibration.npz') as z:
                K = z['intrinsics'][0]; transforms = z['camera_to_base'][:, 0]
                unit = float(z['depth_unit_m'])
                assert str(z['projection_model'][0]) == 'pinhole'
                np.testing.assert_array_equal(timestamps, z['timestamp_s'])
            # Independent geometry check, never an observation or point-cloud filter.
            with np.load(directory/'labels.npz') as z:
                centers = z['entity_xyz_base_m'][:, row['target_id']]
                rotations = z['entity_rotation_world'][:, row['target_id']]
            rng = np.random.default_rng(np.random.SeedSequence([119, row['layout_id'], row['permutation'], row['target_id']]))
            rgb_batch = []; indices = []; clouds = []; valid_cloud = []; counts = []; scenes = []
            with av.open(str(directory/'rgb_front.mp4')) as rgb, av.open(str(directory/'depth_front.mkv')) as depth:
                rgb.streams.video[0].thread_count = depth.streams.video[0].thread_count = 1
                frames = 0
                for frame, (image, distance) in enumerate(zip(rgb.decode(video=0), depth.decode(video=0), strict=True)):
                    frames += 1
                    if frame not in wanted: continue
                    codes = distance.to_ndarray(format='gray16le')
                    if self.include_depth:
                        assert np.isclose(unit,.002)
                        scenes.append((codes,K,transforms[frame]))
                    mask = binary_erosion(masks[frame], iterations=1) & (codes > 0)
                    if mask.any():
                        # Isolated raster artifacts can carry a foreground robot depth
                        # inside a target label. A32mm cube spans at most55.5mm in depth.
                        median = np.median(codes[mask])*unit
                        mask &= np.abs(codes.astype(np.float64)*unit-median) <= .060
                        if self.clean_depth_labels:
                            # Reject isolated raster depth spikes; never use GT3D pose
                            # to select points or modify the policy's depth input.
                            smooth=median_filter(codes,size=3,mode='nearest')
                            mask &= np.abs(codes.astype(np.int32)-smooth.astype(np.int32))*unit <= .010
                    v, u = np.nonzero(mask); count = len(u); cloud = np.zeros((128, 3), np.float32)
                    valid = count >= 8
                    if valid:
                        chosen = rng.choice(count, 128, replace=count < 128); u = u[chosen]; v = v[chosen]
                        d = codes[v, u].astype(np.float64)*unit
                        camera = np.c_[(u-K[0,2])*d/K[0,0], (v-K[1,2])*d/K[1,1], d]
                        cloud = (camera@transforms[frame, :3, :3].T+transforms[frame, :3, 3]).astype(np.float32)
                        local = (cloud-centers[frame])@rotations[frame]
                        assert np.max(abs(local)) < .023, f'Cloud geometry misaligned: {row["path"]} frame{frame}'
                    rgb_batch.append(image.to_ndarray(format='rgb24')); indices.append(wanted[frame])
                    clouds.append(cloud); valid_cloud.append(valid); counts.append(count)
                    if len(indices) == 64:
                        packed=self.pack(rgb_batch, indices, clouds, valid_cloud, counts)
                        yield (*packed, *[np.stack([s[k] for s in scenes]) for k in range(3)]) if self.include_depth else packed
                        rgb_batch = []; indices = []; clouds = []; valid_cloud = []; counts = []
                        scenes=[]
                assert frames == row['frames']
            if indices:
                packed=self.pack(rgb_batch, indices, clouds, valid_cloud, counts)
                yield (*packed, *[np.stack([s[k] for s in scenes]) for k in range(3)]) if self.include_depth else packed

    @staticmethod
    def pack(rgb, ids, clouds, valid, counts):
        return (torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2), np.array(ids),
                np.array(clouds), np.array(valid), np.array(counts, np.int32))
