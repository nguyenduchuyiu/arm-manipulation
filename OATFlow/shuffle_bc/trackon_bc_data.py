"""Causal frozen-vision RAM cache and measured H25 labels for Track-On BC."""
import json
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.shuffle_bc.data import DATA
from OATFlow.shuffle_bc.train_trackon_xyz import (
    encode_side, grouped, sampled_frames, store, validation_points,
)
from OATFlow.simulation.pose import rotation_matrix, rotation_6d
from OATFlow.tracking.data import read_segment
from OATFlow.tracking.point_data import top_camera, project
from OATFlow.tracking.utils import write_json


def allocate(rows, validation=False):
    points = []
    records = []
    if validation:
        plan = validation_points(rows)
        for i, row in enumerate(rows):
            for raw, phase in plan[row['path']].items():
                points.append((i, raw, False))
                records.append(dict(episode=row['path'], raw_frame=int(raw), phase=phase))
        n = len(points)
        expert = [(i, row['context_frames'] + raw, True) for i, row in enumerate(rows)
                  for raw in range(0, row['expert_frames'] - 1, 3)]
        rng = np.random.default_rng(105000)
        points.extend(expert[j] for j in np.sort(rng.choice(len(expert), 64, replace=False)))
    else:
        for i, row in enumerate(rows):
            points.extend((i, int(raw), False) for raw in sampled_frames(row['context_frames']))
            points.extend((i, row['context_frames'] + raw, True)
                          for raw in range(0, row['expert_frames'] - 1, 3))
        n = 0
    requests = {}; locations = {r['path']: {} for r in rows}
    for index, (i, raw, _) in enumerate(points):
        row = rows[i]
        locations[row['path']].setdefault(raw, []).append(index)
        ref, local = ((row['context_path'], raw) if raw < row['context_frames']
                      else (row['path'], raw - row['context_frames']))
        requests.setdefault(ref, set()).add(local)
    offsets = {}; unique = 0
    for ref, frames in sorted(requests.items()):
        offsets[ref] = dict(zip(sorted(frames), range(unique, unique + len(frames))))
        unique += len(frames)
    data = dict(top=torch.empty(len(points), 256, dtype=torch.float16),
                wrist=torch.empty(unique, 64, 768, dtype=torch.float16),
                joints=torch.empty(len(points), 6), xyz=torch.empty(len(points), 3),
                actions=torch.zeros(len(points), 25, 10), mask=torch.zeros(len(points), 25, dtype=torch.bool),
                side_index=torch.empty(len(points), dtype=torch.int64), points=points, records=records,
                xyz_count=n, requests=requests, offsets=offsets, locations=locations, filled=0, side_filled=0)
    # Actions are fixed-current-anchor EE trajectories, not differences between future steps.
    expert_ids = {i: [] for i in range(len(rows))}
    for j, (i, _, active) in enumerate(points):
        if active: expert_ids[i].append(j)
    for i, row in enumerate(tqdm(rows, desc='Align measured H25 labels', mininterval=5)):
        ids = expert_ids[i]
        raw = np.array([points[j][1] - row['context_frames'] for j in ids])
        if not len(raw): continue
        with np.load(DATA / row['path'] / 'observation.npz') as obs:
            ee = obs['ee_position_base_m']; rotation = rotation_matrix(obs['ee_rotation_6d'])
            times = obs['timestamp_s']
        with np.load(DATA / row['path'] / 'supervision.npz') as sup:
            jaw = sup['joint_command'][:, 5]; valid = sup['action_valid']
            np.testing.assert_array_equal(times, sup['timestamp_s'])
        last = len(ee) - 1
        assert valid[:-1].all() and not valid[-1] and (raw < last).all()
        index = raw[:, None] + np.arange(25); future = np.minimum(index + 1, last)
        value = np.empty((len(ids), 25, 10), dtype=np.float32)
        value[..., :3] = ee[future] - ee[raw, None]
        value[..., 3:9] = rotation_6d(rotation[future] @ rotation[raw, None].swapaxes(-1, -2))
        value[..., 9] = jaw[np.minimum(index, last - 1)]
        data['actions'][ids] = torch.from_numpy(value)
        data['mask'][ids] = torch.from_numpy(index < last)
    return data


def store_all(data, row, raw, hidden, segment, local):
    locations = data['locations'][row['path']].get(raw, [])
    for location in locations:
        # The shared XYZ/BC validation frame can appear in both evaluation subsets.
        view = dict(data, locations={row['path']: {raw: location}})
        store(view, row, raw, hidden, segment, local)
        data['filled'] += 1


def native_indices(row, data):
    extras = [raw - row['context_frames'] for raw in data['locations'][row['path']]
              if raw >= row['context_frames']]
    expert = np.unique(np.r_[np.arange(0, row['expert_frames'], 3), row['expert_frames'] - 1, extras])
    return np.r_[sampled_frames(row['context_frames']), row['context_frames'] + expert]


@torch.no_grad()
def cache(tracker, encoder, data, rows, out, name, source=None):
    camera = top_camera(DATA); groups = grouped(rows); started = time.perf_counter()
    root = source if source is not None else out
    total = sum(len(sampled_frames(g[0]['context_frames'])) +
                sum(len(native_indices(r, data)) - len(sampled_frames(r['context_frames'])) for r in g)
                for g in groups)
    progress = tqdm(total=total, desc=f'{name}: frozen Track-On/ViT once', mininterval=5)
    for done, group in enumerate(groups, 1):
        destinations = [(root / 'query_cache' / r['path']).with_suffix('.npz') for r in group]
        reuse = all(p.exists() for p in destinations)
        assert tracker is not None or reuse, 'Missing native query cache for resume'
        context = encode_side(encoder, data, group[0]['context_path'])
        states = []; queries = [[] for _ in group]
        if not reuse:
            top = read_segment(DATA, group[0]['context_path'], 10, views=('top',), center_crop=True)
            for j, raw in enumerate(top['raw_index']):
                with torch.autocast('cuda', dtype=torch.float16):
                    features = tracker.features(top['rgb'][0, j:j+1].cuda())
                    if raw == 0:
                        states = [tracker.initialize(features, torch.tensor(
                            project(top['xyz'][0, r['target_id']], camera), device='cuda', dtype=torch.float32))
                                  for r in group]
                    for q, (row, state) in enumerate(zip(group, states)):
                        hidden, uv, _ = tracker.step(features, state)
                        assert torch.isfinite(hidden).all() and torch.isfinite(uv).all()
                        queries[q].append(hidden.half().cpu())
                        store_all(data, row, int(raw), hidden, context, int(raw))
                progress.update()
            del top
        else:
            progress.update(len(sampled_frames(group[0]['context_frames'])))
        for q, (row, destination) in enumerate(zip(group, destinations)):
            side = encode_side(encoder, data, row['path'])
            indices = native_indices(row, data)
            if reuse:
                with np.load(destination) as saved:
                    np.testing.assert_array_equal(saved['raw_index'], indices)
                    features = saved['features']
                    assert features.shape == (len(indices), 256) and features.dtype == np.float16
                    assert np.isfinite(features).all()
                    for raw, hidden in zip(indices, features):
                        prefix = raw < row['context_frames']
                        store_all(data, row, int(raw), torch.from_numpy(hidden), context if prefix else side,
                                  int(raw if prefix else raw - row['context_frames']))
                progress.update(len(indices) - len(sampled_frames(row['context_frames'])))
            else:
                # Decode once at stride1, then retain the exact stride3/update endpoints.
                top = read_segment(DATA, row['path'], 1, views=('top',), center_crop=True)
                for raw in indices[indices >= row['context_frames']] - row['context_frames']:
                    with torch.autocast('cuda', dtype=torch.float16):
                        hidden, uv, _ = tracker.step(tracker.features(top['rgb'][0, raw:raw+1].cuda()), states[q])
                    assert torch.isfinite(hidden).all() and torch.isfinite(uv).all()
                    queries[q].append(hidden.half().cpu())
                    store_all(data, row, int(row['context_frames'] + raw), hidden, side, int(raw))
                    progress.update()
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez(destination, features=torch.stack(queries[q]).numpy(), raw_index=indices)
                del top
            del side
        del context
        elapsed = time.perf_counter() - started
        record = dict(status='caching frozen vision', split=name, completed_groups=done, groups=len(groups),
                      points=data['filled'], frame_updates=progress.n, total_frame_updates=total,
                      elapsed_s=elapsed, eta_s=elapsed / progress.n * (total - progress.n),
                      native_cache_reused=reuse, gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(out / 'progress.json', record)
        if done == 1 or done % 5 == 0: print(json.dumps(record), flush=True)
    progress.close()
    assert data['filled'] == len(data['top']) and data['side_filled'] == len(data['wrist'])
    data['truth'] = data['xyz'].numpy()
    np.save(out / f'{name}_xyz.npy', data['truth'])
    write_json(out / f'{name}_episodes.json', rows)
    if data['records']: write_json(out / f'{name}_points.json', data['records'])
    for key in ('locations', 'requests', 'offsets'): del data[key]
    return time.perf_counter() - started


def batch(data, ids):
    return {key: (data['wrist'][data['side_index'][ids]] if key == 'wrist' else data[key][ids]).cuda()
            for key in ('top', 'wrist', 'joints', 'xyz', 'actions', 'mask')}


def schedule(data, rows, order, size):
    by_context = {}
    for index, (i, raw, active) in enumerate(data['points']):
        by_context.setdefault(rows[i]['context_path'], {False: [], True: []})[active].append(index)
    groups = grouped(rows); result = []
    for group_index in order:
        phases = by_context[groups[group_index][0]['context_path']]
        for active in (False, True):
            ids = sorted(phases[active], key=lambda j: (data['points'][j][1], rows[data['points'][j][0]]['target_id']))
            result.extend((int(group_index), active, ids[start:start+size]) for start in range(0, len(ids), size))
    assert sum(len(ids) for _, _, ids in result) == len(data['points'])
    return result
