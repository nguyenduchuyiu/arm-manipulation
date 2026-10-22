"""Held-out order/layout XYZ evaluation of the frozen-vision context model."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.shuffle_bc.context_cache import read_left
from OATFlow.shuffle_bc.context_model import ContextPolicy
from OATFlow.shuffle_bc.data import DATA, splits
from OATFlow.shuffle_bc.train_context_xyz import metrics
from OATFlow.tracking.data import evaluation_plan
from OATFlow.tracking.point_data import load_points
from OATFlow.tracking.pretrain_xyz import inputs
from OATFlow.tracking.utils import write_json


def cases():
    config = json.loads((DATA / 'config.json').read_text()); groups = splits()
    result = []
    for name, layouts, permutations in [('composition', (0, 31), config['unseen_permutations'][::2]),
            ('spatial', (36, 39), config['train_permutations'][::4]),
            ('combined', (36, 39), config['unseen_permutations'][::2])]:
        chosen = [r for r in groups[name] if r['layout_id'] in layouts and r['permutation'] in permutations]
        assert len(chosen) == 32; result.extend(chosen)
    assert len(result) == 96
    return result


@torch.inference_mode()
def cache_parity(model, cache):
    top_root = Path(json.loads((cache / 'config.json').read_text())['top_source'])
    points = json.loads((cache / 'val_points.json').read_text())
    ids = np.rint(np.linspace(0, 23, 8)).astype(int); chosen = [points[i] for i in ids]
    names = list(dict.fromkeys(p['episode'] for p in chosen))
    rows = {r['path']: r for r in splits()['val']}
    episodes, _ = load_points(DATA, [rows[name] for name in names], stride=10)
    requests = {}; keys = []; ends = []
    for point in chosen:
        i = names.index(point['episode']); ep = episodes[i]; raw = point['raw_frame']; row = ep['row']
        end = int(np.searchsorted(ep['raw_index'], raw)); assert ep['raw_index'][end] == raw
        ends.append((i, end)); ref, frame = ((row['context_path'], raw) if raw < row['context_frames']
            else (row['path'], raw - row['context_frames']))
        keys.append((ref, frame)); requests.setdefault(ref, set()).add(frame)
    side = {ref: read_left(ref, frames) for ref, frames in requests.items()}
    rgb, query, _ = inputs(episodes, ends)
    images = torch.stack([side[ref]['rgb'][side[ref]['index'][frame]] for ref, frame in keys]).cuda()
    joints = torch.tensor(np.array([side[ref]['joints'][side[ref]['index'][frame]] for ref, frame in keys]), device='cuda')
    top_cached = torch.from_numpy(np.load(top_root / 'val_features.npy')[ids].reshape(-1, 300, 768)).cuda()
    wrist_cached = torch.from_numpy(np.load(cache / 'val_wrist.npy')[ids]).cuda()
    state_cached = torch.from_numpy(np.load(cache / 'val_joints.npy')[ids]).cuda()
    torch.testing.assert_close(joints, state_cached, rtol=0, atol=0)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        top, wrist = model.encode(rgb, query, images)
        online = model.xyz_from_features(top, wrist, joints) * model.xyz_std + model.xyz_mean
        offline = model.xyz_from_features(top_cached, wrist_cached, state_cached) * model.xyz_std + model.xyz_mean
    error = (online - offline).norm(dim=-1) * 1000
    return dict(points=len(chosen), mean_prediction_difference_mm=float(error.mean()),
        max_prediction_difference_mm=float(error.max()), joint_state_exact=True,
        top_token_mean_abs_difference=float((top.float() - top_cached.float()).abs().mean()),
        wrist_token_mean_abs_difference=float((wrist.float() - wrist_cached.float()).abs().mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e108_README.md', out / 'README.md')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = dict(status='running', checkpoint=str(args.checkpoint), epoch=saved['epoch'], step=saved['step'],
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), data=str(DATA),
        scenes='32unseen-order/32unseen-layout/32both; fixed e105 case identities; recorded full shuffle/expert histories',
        endpoint_stride=10, phases='visible/shuffle/post_shuffle/exposed/lift',
        inputs='causal top RGB + frame0 dot, current left RGB as surrogate wrist, measured joints; no ID/XYZ/masks',
        actual_wrist_camera=False, action_control_test=False, H_K_flow_noise_budget='not applicable; XYZ prediction only',
        batch=16, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path('OATFlow/shuffle_bc/context_model.py'), Path('OATFlow/shuffle_bc/context_cache.py'),
             Path('OATFlow/shuffle_bc/train_context_xyz.py'), Path('OATFlow/tracking/point.py'),
             Path('OATFlow/tracking/backbone.py'), Path('OATFlow/tracking/point_data.py'), Path('OATFlow/tracking/data.py')]})
    write_json(out / 'config.json', config); write_json(out / 'progress.json', dict(status='loading recorded test RGB'))
    for name in config['source_sha256']:
        p = Path(name).resolve(); dest = out / 'source' / p.relative_to(Path.cwd())
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    (out / 'source' / 'README.md').write_text('Held-out XYZ evaluation source; exact checkpoint, cases and settings in ../config.json and ../README.md. No action-control test.\n')
    rows = cases(); write_json(out / 'cases.json', rows)
    episodes, _ = load_points(DATA, rows, stride=10); plan = evaluation_plan(episodes)
    requests = {}; keys = []
    for i, end, _ in plan:
        row = rows[i]; frame = int(episodes[i]['raw_index'][end])
        ref, raw = ((row['context_path'], frame) if frame < row['context_frames']
                    else (row['path'], frame - row['context_frames']))
        keys.append((ref, raw)); requests.setdefault(ref, set()).add(raw)
    side = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(read_left, ref, indices): ref for ref, indices in requests.items()}
        for job in tqdm(as_completed(jobs), total=len(jobs), desc='Decode held-out left/state', mininterval=5):
            side[jobs[job]] = job.result()
    model = ContextPolicy(); model.load_context_state(saved['model']); model.eval().cuda()
    parity = cache_parity(model, Path(saved['config']['cache']))
    write_json(out / 'online_cache_parity.json', parity); print(json.dumps(dict(online_cache_parity=parity)), flush=True)
    predictions = []; truths = []; records = []; started = time.perf_counter()
    with torch.inference_mode():
        for start in tqdm(range(0, len(plan), config['batch']), desc='Held-out context XYZ', mininterval=5):
            selected = plan[start:start + config['batch']]
            rgb, query, xyz = inputs(episodes, [(i, end) for i, end, _ in selected])
            left = torch.stack([side[ref]['rgb'][side[ref]['index'][raw]] for ref, raw in keys[start:start + len(selected)]]).cuda()
            joints = torch.tensor(np.array([side[ref]['joints'][side[ref]['index'][raw]]
                for ref, raw in keys[start:start + len(selected)]]), device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                top, wrist = model.encode(rgb, query, left)
                predicted = model.xyz_from_features(top, wrist, joints) * model.xyz_std + model.xyz_mean
            predictions.extend(predicted.cpu().numpy()); truths.extend(xyz[:, -1].cpu().numpy())
            records.extend(dict(episode=rows[i]['path'], group=rows[i]['evaluation_group'],
                raw_frame=int(episodes[i]['raw_index'][end]), phase=phase) for i, end, phase in selected)
            write_json(out / 'progress.json', dict(status='predicting', completed=len(records), total=len(plan)))
    prediction = np.array(predictions); truth = np.array(truths)
    summary, details = metrics(prediction, truth, records)
    grouped = {}
    for name in ('composition', 'spatial', 'combined'):
        ids = [i for i, r in enumerate(records) if r['group'] == name]
        grouped[name], _ = metrics(prediction[ids], truth[ids], [records[i] for i in ids])
    result = dict(status='complete', checkpoint_epoch=saved['epoch'], checkpoint_step=saved['step'],
        metrics=summary, by_split=grouped, prediction_seconds=time.perf_counter() - started,
        episodes=len(rows), endpoints=len(records), action_control_test=False, online_cache_parity=parity)
    write_json(out / 'summary.json', result); write_json(out / 'predictions.json', details)
    np.savez(out / 'predictions.npz', prediction_base_m=prediction, truth_base_m=truth)
    config.update(status='complete', endpoints=len(records)); write_json(out / 'config.json', config)
    write_json(out / 'progress.json', result)
    with (out / 'README.md').open('a') as stream:
        stream.write(f'\nHeld-out XYZ test complete: checkpoint epoch{saved["epoch"]}/step{saved["step"]}, '
            f'{len(rows)}episodes/{len(records)}phase endpoints; macro{summary["macro_phase_mm"]:.3f}mm. '
            'Exact split/phase results in summary.json. Recorded expert motion, no learned robot-control rollout.\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
