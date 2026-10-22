"""RAM-cached e96 RGB segments and strictly past/current localization clips."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import av
import numpy as np
import torch
from tqdm import tqdm


VIEWS = ('top', 'left', 'right')


def subset(root):
    rows = [json.loads(line) for line in (root / 'manifest.jsonl').read_text().splitlines()]
    assert json.loads((root / 'config.json').read_text())['status'] == 'complete'
    val_permutations = (0, 9, 16, 18)
    selected = dict(
        train=[r for r in rows if r['evaluation_group'] == 'train' and r['layout_id'] < 8],
        val=[r for r in rows if r['evaluation_group'] == 'val' and r['layout_id'] in (32, 33)
             and r['permutation'] in val_permutations],
        composition=[r for r in rows if r['evaluation_group'] == 'composition' and r['layout_id'] in (0, 1)],
        spatial=[r for r in rows if r['evaluation_group'] == 'spatial' and r['layout_id'] in (36, 37)
                 and r['permutation'] in val_permutations],
        combined=[r for r in rows if r['evaluation_group'] == 'combined' and r['layout_id'] in (36, 37)])
    assert {k: len(v) for k, v in selected.items()} == dict(train=512, val=32, composition=64, spatial=32, combined=64)
    assert not ({r['path'] for r in selected['train']} & {r['path'] for k, v in selected.items() if k != 'train' for r in v})
    return selected


def read_segment(root, ref, stride, evaluation_labels=False, views=VIEWS, center_crop=False):
    path = root / ref
    with np.load(path / 'labels.npz') as z:
        count = len(z['timestamp_s'])
        indices = np.unique(np.r_[np.arange(0, count, stride), count - 1])
        xyz = z['entity_xyz_base_m'][indices].astype(np.float32)
        times = z['timestamp_s'][indices].astype(np.float32)
    with np.load(path / 'supervision.npz') as z:
        phase = z['phase'][indices]
    hidden = None
    if evaluation_labels:
        with np.load(path / 'masks.npz') as z:
            view_ids = [VIEWS.index(name) for name in views]
            hidden = (z['occlusion_fraction'][indices][:, view_ids] >= .95).all(axis=1)
    rgb = np.empty((len(views), len(indices), 3, 240, 320), np.uint8)
    for view, name in enumerate(views):
        cursor = decoded = 0
        with av.open(str(path / f'rgb_{name}.mp4')) as video:
            stream = video.streams.video[0]
            stream.thread_count = 1
            for index, frame in enumerate(video.decode(stream)):
                if cursor < len(indices) and index == indices[cursor]:
                    pixels = (frame.to_ndarray(format='rgb24')[40:280] if center_crop else
                              frame.reformat(width=320, height=240, format='rgb24').to_ndarray())
                    rgb[view, cursor] = pixels.transpose(2, 0, 1)
                    cursor += 1
                decoded += 1
        assert decoded == count and cursor == len(indices), ref
    return dict(rgb=torch.from_numpy(rgb), xyz=xyz, time=times, phase=phase,
                hidden=hidden, raw_index=indices)


def load_episodes(root, rows, stride=3, workers=8, segments=None, evaluation_labels=False, views=VIEWS,
                  center_crop=False):
    segments = {} if segments is None else segments
    refs = sorted({ref for row in rows for ref in (row['context_path'], row['path'])} - segments.keys())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(read_segment, root, ref, stride, evaluation_labels, views, center_crop): ref for ref in refs}
        for job in tqdm(as_completed(jobs), total=len(jobs), desc='Decode RGB segments to RAM', mininterval=5):
            segments[jobs[job]] = job.result()
    episodes = []
    for row in rows:
        first, last = segments[row['context_path']], segments[row['path']]
        target = row['target_id']
        phase = np.concatenate((first['phase'], last['phase']))
        hidden = None
        if evaluation_labels:
            assert first['hidden'] is not None and last['hidden'] is not None
            hidden = np.concatenate((first['hidden'][:, target], last['hidden'][:, 0]))
        nc = len(first['time'])
        episode = dict(row=row, context=first, expert=last,
            xyz=np.concatenate((first['xyz'][:, target], last['xyz'][:, target])),
            time=np.concatenate((first['time'], last['time'])), phase=phase, hidden=hidden,
            raw_index=np.r_[first['raw_index'], row['context_frames'] + last['raw_index']])
        episode['categories'] = [np.flatnonzero(phase == 'reveal'),
            np.flatnonzero(phase == 'shuffle'),
            np.flatnonzero(phase == 'hold'), np.arange(nc, len(phase) - 1)]
        assert all(len(indices) for indices in episode['categories'])
        episodes.append(episode)
    return episodes, segments


def sample_indices(end, frames=30):
    if frames < 1:
        raise ValueError('frames must be positive')
    indices = np.rint(np.linspace(0, end, frames)).astype(np.int64) if frames > 1 else np.array([end])
    assert (frames == 1 or indices[0] == 0) and indices[-1] == end and (indices <= end).all()
    return indices


def evaluation_plan(episodes):
    plan = []
    for i, ep in enumerate(episodes):
        phase = ep['phase']
        points = [('visible', int(np.flatnonzero(phase == 'reveal')[-1]))]
        shuffle = np.flatnonzero(phase == 'shuffle')
        runs = np.split(shuffle, np.flatnonzero(np.diff(shuffle) > 1) + 1)
        points += [('shuffle', int(run[len(run) // 2])) for run in runs if len(run)]
        points += [('post_shuffle', len(ep['context']['time']) - 1),
                   ('exposed', min(int(np.searchsorted(ep['raw_index'], ep['row']['decision_frames']['t_obj'])), len(phase) - 1)),
                   ('lift', len(phase) - 1)]
        plan.extend((i, end, phase_name) for phase_name, end in points)
    return plan
