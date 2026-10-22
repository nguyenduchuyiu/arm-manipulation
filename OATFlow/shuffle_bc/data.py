"""Past/current top clips and measured H25 labels from the original shuffle data."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.simulation.pose import rotation_matrix, rotation_6d
from OATFlow.tracking.point_data import load_points
from OATFlow.tracking.data import read_segment, sample_indices

DATA = Path('/workspace/robotics_assets/datasets/shuffle_pick_multiview')
ROOT = Path('/workspace/robotics_assets/experiments/e105_tcow')


def splits():
    rows = [json.loads(line) for line in (DATA / 'manifest.jsonl').read_text().splitlines()]
    return {group: [r for r in rows if r['evaluation_group'] == group]
            for group in ('train', 'val', 'composition', 'spatial', 'combined')}


def read_context(ref):
    segment = read_segment(DATA, ref, 1, views=('top',), center_crop=True)
    count = len(segment['raw_index'])
    keep = np.unique(np.r_[np.arange(0, count, 3), np.arange(0, count, 10), count - 1])
    segment['rgb'] = segment['rgb'][:, keep]
    for key in ('xyz', 'time', 'phase', 'raw_index'):
        segment[key] = segment[key][keep]
    return segment


def select_indices(raw_indices, end):
    desired = sample_indices(int(raw_indices[end]))
    selected = np.searchsorted(raw_indices, desired, side='right') - 1
    assert selected[0] == 0 and selected[-1] == end and (selected <= end).all()
    return selected


def clip(episode, end):
    indices = select_indices(episode['raw_index'], end)
    nc = len(episode['context']['time'])
    prefix = indices < nc
    rgb = torch.empty((30, 3, 240, 320), dtype=torch.uint8)
    for source, chosen, offset in ((episode['context'], prefix, 0), (episode['expert'], ~prefix, nc)):
        if chosen.any():
            rgb[torch.from_numpy(np.flatnonzero(chosen))] = source['rgb'][0, torch.from_numpy(indices[chosen] - offset)]
    query = torch.zeros(1, 1, 30, 240, 320)
    x, y = np.rint(episode['query_uv']).astype(int)
    yy, xx = torch.meshgrid(torch.arange(max(0, y-3), min(240, y+4)),
                            torch.arange(max(0, x-3), min(320, x+4)), indexing='ij')
    query[0, 0, 0, yy, xx] = ((xx-x).square() + (yy-y).square() <= 9).float()
    return rgb.permute(1, 0, 2, 3)[None], query, torch.from_numpy(episode['xyz'][indices].copy())[None]


def load(rows, segments=None):
    segments = {} if segments is None else segments
    contexts = {r['context_path'] for r in rows}
    refs = sorted((contexts | {r['path'] for r in rows}) - segments.keys())
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(read_context, ref) if ref in contexts else
                pool.submit(read_segment, DATA, ref, 3, views=('top',), center_crop=True): ref for ref in refs}
        for job in tqdm(as_completed(jobs), total=len(jobs), desc='Decode top context/expert RGB', mininterval=5):
            segments[jobs[job]] = job.result()
    episodes, segments = load_points(DATA, rows, segments=segments)
    samples = []
    for i, ep in enumerate(tqdm(episodes, desc='Align BC labels', mininterval=5)):
        path = DATA / ep['row']['path']
        with np.load(path / 'observation.npz') as z:
            ep['joints'] = z['joint_position'].astype(np.float32)
            ep['ee'] = z['ee_position_base_m'].copy()
            ep['rotation'] = rotation_matrix(z['ee_rotation_6d'])
            times = z['timestamp_s'].copy()
        with np.load(path / 'supervision.npz') as z:
            ep['jaw'] = z['joint_command'][:, 5].copy()
            valid = z['action_valid'].copy()
            np.testing.assert_array_equal(times, z['timestamp_s'])
        assert valid[:-1].all() and not valid[-1]
        np.testing.assert_array_equal(times[ep['expert']['raw_index']].astype(np.float32), ep['expert']['time'])
        nc = len(ep['context']['time'])
        last = ep['row']['context_frames'] - 1
        context_ends = np.unique(np.r_[np.arange(0, last + 1, 10), last])
        for raw in context_ends:
            end = int(np.searchsorted(ep['context']['raw_index'], raw))
            assert ep['context']['raw_index'][end] == raw
            samples.append((i, end, False))
        samples.extend((i, nc + j, True) for j, raw in enumerate(ep['expert']['raw_index']) if valid[raw])
    return episodes, segments, samples


def training_batches(episodes, points, rng, batch_size=4):
    groups = {}
    for index, (i, end, active) in enumerate(points):
        groups.setdefault(episodes[i]['row']['context_path'], {False: [], True: []})[active].append(index)
    names = list(groups)
    rng.shuffle(names)
    batches = []
    for name in names:
        for active in (False, True):
            ids = groups[name][active]
            ids.sort(key=lambda idx: (episodes[points[idx][0]]['raw_index'][points[idx][1]],
                                      episodes[points[idx][0]]['row']['target_id']))
            batches.extend(np.array(ids[start:start+batch_size], dtype=np.int64)
                           for start in range(0, len(ids), batch_size))
    assert sum(map(len, batches)) == len(points)
    return batches


def batch(episodes, points, device='cuda'):
    values = [[] for _ in range(6)]
    for i, end, active in points:
        ep = episodes[i]
        raw = int(ep['raw_index'][end] - ep['row']['context_frames'])
        rgb, query, xyz = clip(ep, end)
        action = np.zeros((25, 10), np.float32)
        mask = np.zeros(25, bool)
        state = np.zeros((1, 6), np.float32)
        if active:
            assert 0 <= raw < len(ep['ee']) - 1
            index = raw + np.arange(25)
            last = len(ep['ee']) - 1
            future = np.minimum(index + 1, last)
            action[:, :3] = ep['ee'][future] - ep['ee'][raw]
            action[:, 3:9] = rotation_6d(ep['rotation'][future] @ ep['rotation'][raw].T)
            action[:, 9] = ep['jaw'][np.minimum(index, last - 1)]
            mask = index < last
            state = ep['joints'][raw:raw+1]
        else:
            assert raw < 0
        for bucket, value in zip(values, (rgb, query, torch.from_numpy(state),
                torch.from_numpy(action[None]), torch.from_numpy(mask[None]), xyz)):
            bucket.append(value)
    # Six policy/loss inputs; target IDs and hidden geometry never become observations.
    return tuple(torch.cat(v).to(device) for v in values[:6])
