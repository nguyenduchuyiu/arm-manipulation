"""One top camera, one initial visible mask, amodal mask targets over causal clips."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import av
import numpy as np
import torch
from tqdm import tqdm

from OATFlow.tracking.data import sample_indices, subset


def splits(root):
    rows = [json.loads(line) for line in (root / 'manifest.jsonl').read_text().splitlines()]
    chosen = subset(root)
    chosen['train'] = [r for r in rows if r['evaluation_group'] == 'train']
    assert len(chosen['train']) == 2048
    return chosen


def read_segment(root, ref, context):
    path = root / ref
    with np.load(path / 'supervision.npz') as labels:
        count = len(labels['timestamp_s'])
        indices = np.unique(np.r_[np.arange(0, count, 3), count - 1])
        phase = labels['phase'][indices].copy()
        times = labels['timestamp_s'][indices].copy()
    rgb = np.empty((len(indices), 3, 240, 320), np.uint8)
    cursor = 0
    with av.open(str(path / 'rgb_top.mp4')) as video:
        video.streams.video[0].thread_count = 1
        for raw, frame in enumerate(video.decode(video=0)):
            if cursor < len(indices) and raw == indices[cursor]:
                rgb[cursor] = frame.to_ndarray(format='rgb24')[40:280].transpose(2, 0, 1)
                cursor += 1
    assert raw + 1 == count and cursor == len(indices), ref
    with np.load(path / 'masks.npz') as masks:
        assert str(masks['bitorder']) == 'little'
        np.testing.assert_array_equal(masks['timestamp_s'][indices], times)
        if context:
            assert masks['amodal_entity_ids'].tolist() == [1, 2, 3, 4, 5, 6]
            packed = masks['amodal_packed'][indices, 0, :4, 40:280].copy()
            instance = masks['instance'][indices, 0, 40:280]
            visible = np.stack([instance == target + 1 for target in range(4)], axis=1)
            prompt = visible[0].copy()
        else:
            packed = masks['tcow_packed'][indices, 0, :1, 40:280].copy()
            target = int(masks['target_ids'][0])
            visible = (masks['instance'][indices, 0, 40:280] == target + 1)[:, None]
            prompt = None
        occlusion = masks['occlusion_fraction'][indices, 0].copy()
        visible = np.packbits(visible, axis=-1, bitorder='little')
    assert all(packed.sum((-1, -2)).flatten() > 0), ref
    return dict(rgb=torch.from_numpy(rgb), packed=packed, visible=visible, prompt=prompt,
                phase=phase, time=times, raw_index=indices, hidden=occlusion >= .95, occlusion=occlusion)


def load(root, rows, segments=None, progress=None):
    segments = {} if segments is None else segments
    contexts = {r['context_path'] for r in rows}
    refs = sorted((contexts | {r['path'] for r in rows}) - segments.keys())
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(read_segment, root, ref, ref in contexts): ref for ref in refs}
        for done, job in enumerate(tqdm(as_completed(jobs), total=len(jobs), desc='Cache top RGB + packed masks in RAM', mininterval=5), 1):
            segments[jobs[job]] = job.result()
            if progress is not None and (done % 32 == 0 or done == len(jobs)):
                progress(done, len(jobs))
    episodes = []
    for row in rows:
        first, last = segments[row['context_path']], segments[row['path']]
        target = row['target_id']
        assert first['prompt'][target].any()
        phase = np.r_[first['phase'], last['phase']]
        nc = len(first['time'])
        categories = [np.flatnonzero(phase == 'reveal'), np.flatnonzero(phase == 'shuffle'),
                      np.flatnonzero(phase == 'hold'), np.arange(nc, len(phase))]
        assert all(len(x) for x in categories), row['path']
        episodes.append(dict(row=row, context=first, expert=last, phase=phase,
            raw_index=np.r_[first['raw_index'], row['context_frames'] + last['raw_index']],
            hidden=np.r_[first['hidden'][:, target], last['hidden'][:, 0]], categories=categories))
    return episodes, segments


def clip(episode, end):
    indices = sample_indices(end, 30)
    nc = len(episode['context']['time'])
    rgb = np.empty((30, 3, 240, 320), np.uint8)
    packed = np.empty((30, 240, 40), np.uint8)
    visible = np.empty_like(packed)
    occlusion = np.empty(30, np.float32)
    for segment, selected, offset, target in (
            (episode['context'], indices < nc, 0, episode['row']['target_id']),
            (episode['expert'], indices >= nc, nc, 0)):
        selected_indices = indices[selected] - offset
        rgb[selected] = segment['rgb'][selected_indices].numpy()
        packed[selected] = segment['packed'][selected_indices, target]
        visible[selected] = segment['visible'][selected_indices, target]
        occlusion[selected] = segment['occlusion'][selected_indices, target]
    truth = np.unpackbits(packed, axis=-1, count=320, bitorder='little').astype(bool)
    visible = np.unpackbits(visible, axis=-1, count=320, bitorder='little').astype(bool)
    query = np.zeros((1, 30, 240, 320), bool)
    query[0, 0] = episode['context']['prompt'][episode['row']['target_id']]
    assert not (visible & ~truth).any()
    return (torch.from_numpy(rgb.transpose(1, 0, 2, 3)), torch.from_numpy(query), torch.from_numpy(truth),
            torch.from_numpy(visible), torch.from_numpy(occlusion))


def batch(episodes, points):
    values = [clip(episodes[i], end) for i, end in points]
    return tuple(torch.stack([x[k] for x in values]).cuda() for k in range(5))


def train_plan(episodes, rng):
    plan = []
    for i in rng.permutation(len(episodes)):
        category = rng.choice(4, p=[.15, .35, .2, .3])
        plan.append((int(i), int(rng.choice(episodes[i]['categories'][category]))))
    return plan
