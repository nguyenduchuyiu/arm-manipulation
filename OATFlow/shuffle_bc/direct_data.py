"""Expert-only FM labels and full-history frozen TCOW/auxiliary-camera RAM cache."""
import time

import av
import numpy as np
import torch
from tqdm import tqdm

from OATFlow.shuffle_bc.data import DATA
from OATFlow.shuffle_bc.direct_model import FEATURE_BLOCK, pack_top, unpack_top
from OATFlow.simulation.pose import rotation_matrix, rotation_6d
from OATFlow.tracking.data import sample_indices
from OATFlow.tracking.mask_data import read_segment
from OATFlow.tracking.utils import write_json


def grouped(rows):
    result = {}
    for i, row in enumerate(rows):
        result.setdefault(row['context_path'], []).append(i)
    return list(result.values())


def allocate(rows, validation=False):
    starts = []
    slices = []
    offset = 0
    for row in rows:
        raw = np.arange(0, row['expert_frames']-1, 3, dtype=np.int32)
        if validation:
            raw = raw[np.unique(np.rint(np.linspace(0, len(raw)-1, 8)).astype(int))]
        starts.append(raw)
        slices.append((offset, offset + len(raw)))
        offset += len(raw)
    data = dict(top=torch.empty(offset, 300, 768, dtype=torch.int8),
        scale=torch.empty(offset, 300, 768 // FEATURE_BLOCK, dtype=torch.float16),
        side=torch.empty(offset, 64, 768, dtype=torch.float16),
        joints=torch.empty(offset, 6), actions=torch.empty(offset, 25, 10),
        mask=torch.empty(offset, 25, dtype=torch.bool), filled=np.zeros(offset, bool),
        rows=rows, starts=starts, slices=slices)
    for i, row in enumerate(tqdm(rows, desc='Align expert H25 labels', mininterval=5)):
        begin, end = slices[i]
        raw = starts[i]
        with np.load(DATA / row['path'] / 'observation.npz') as obs:
            ee = obs['ee_position_base_m']
            rotation = rotation_matrix(obs['ee_rotation_6d'])
            joints = obs['joint_position'][raw].astype(np.float32)
            times = obs['timestamp_s']
        with np.load(DATA / row['path'] / 'supervision.npz') as sup:
            np.testing.assert_array_equal(times, sup['timestamp_s'])
            valid = sup['action_valid']
            jaw = sup['joint_command'][:, 5]
        last = len(ee)-1
        assert valid[:-1].all() and not valid[-1] and (raw < last).all()
        index = raw[:, None] + np.arange(25)
        future = np.minimum(index+1, last)
        actions = np.empty((len(raw), 25, 10), np.float32)
        actions[..., :3] = ee[future] - ee[raw, None]
        actions[..., 3:9] = rotation_6d(rotation[future] @ rotation[raw, None].swapaxes(-1, -2))
        actions[..., 9] = jaw[np.minimum(index, last-1)]
        data['joints'][begin:end] = torch.from_numpy(joints)
        data['actions'][begin:end] = torch.from_numpy(actions)
        data['mask'][begin:end] = torch.from_numpy(index < last)
    return data


def memory_gib(data):
    return sum(v.numel()*v.element_size() for v in data.values() if isinstance(v, torch.Tensor))/2**30


def inputs(images, prompt, ends):
    indices = np.stack([sample_indices(int(end), 30) for end in ends])
    assert (indices <= np.asarray(ends)[:, None]).all() and (indices[:, 0] == 0).all()
    rgb = images[torch.from_numpy(indices)].permute(0, 2, 1, 3, 4).contiguous()
    query = torch.zeros(len(ends), 1, 30, 240, 320, dtype=torch.bool)
    query[:, 0, 0] = torch.from_numpy(prompt)
    return rgb.cuda(), query.cuda()


def read_side(row, frames):
    images = np.empty((len(frames), 3, 320, 320), np.uint8)
    cursor = 0
    with av.open(str(DATA / row['path'] / 'rgb_left.mp4')) as video:
        video.streams.video[0].thread_count = 1
        for raw, frame in enumerate(video.decode(video=0)):
            if cursor < len(frames) and raw == frames[cursor]:
                images[cursor] = frame.to_ndarray(format='rgb24').transpose(2, 0, 1)
                cursor += 1
    assert cursor == len(frames) and raw+1 == row['expert_frames']
    return torch.from_numpy(images)


@torch.inference_mode()
def cache_group(tracker, camera, data, group, size, out=None, name='train', timer=None):
    row = data['rows'][group[0]]
    context = read_segment(DATA, row['context_path'], True)
    count = 0
    for i in group:
        row = data['rows'][i]
        expert = read_segment(DATA, row['path'], False)
        images = torch.cat((context['rgb'], expert['rgb']))
        raw = data['starts'][i]
        local = np.searchsorted(expert['raw_index'], raw)
        np.testing.assert_array_equal(expert['raw_index'][local], raw)
        ends = len(context['raw_index']) + local
        prompt = context['prompt'][row['target_id']]
        side = read_side(row, raw)
        base = data['slices'][i][0]
        for start in tqdm(range(0, len(raw), size), desc=f'{name} frozen vision {row["path"]}', mininterval=5, leave=False):
            stop = min(start+size, len(raw))
            rgb, query = inputs(images, prompt, ends[start:stop])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = tracker.features(rgb, query)[:, -1].contiguous()
                codes, scales = pack_top(hidden)
                secondary = camera(side[start:stop].cuda()).half()
            assert torch.isfinite(scales).all() and torch.isfinite(secondary).all()
            dest = slice(base+start, base+stop)
            data['top'][dest].copy_(codes.cpu())
            data['scale'][dest].copy_(scales.cpu())
            data['side'][dest].copy_(secondary.cpu())
            data['filled'][dest] = True
            count += stop-start
            if out is not None:
                done = int(data['filled'].sum())
                elapsed = time.perf_counter()-timer
                write_json(out / 'cache_progress.json', dict(status='caching frozen vision', split=name,
                    completed=done, total=len(data['top']), episode=row['path'], elapsed_seconds=elapsed,
                    eta_seconds=elapsed/max(done, 1)*(len(data['top'])-done),
                    cache_gib=memory_gib(data), native_history='frame0 context through current expert frame'))
            del rgb, query, hidden, codes, scales, secondary
        del expert, images, side
    return count


def batch(data, ids):
    assert data['filled'][ids].all(), 'Unfilled frozen-feature cache entry'
    codes = data['top'][ids].cuda()
    scales = data['scale'][ids].cuda()
    return dict(top=unpack_top(codes, scales), side=data['side'][ids].cuda(),
        joints=data['joints'][ids].cuda(), actions=data['actions'][ids].cuda(), mask=data['mask'][ids].cuda())
