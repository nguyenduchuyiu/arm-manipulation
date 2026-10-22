"""Reuse e107 top tokens and cache aligned left-camera ViT features/state."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time

import av
import numpy as np
import torch
from tqdm import tqdm

from OATFlow.shuffle_bc.context_model import FrozenWristViT
from OATFlow.shuffle_bc.data import DATA, splits
from OATFlow.top_pick.data import VISION
from OATFlow.tracking.linear_probe import fingerprint
from OATFlow.tracking.utils import write_json

TOP = Path('/workspace/robotics_assets/experiments/e107_linear/train_3')


def read_left(ref, frames):
    path = DATA / ref; frames = sorted(frames)
    with np.load(path / 'observation.npz') as z:
        joints = z['joint_position'][frames].astype(np.float32); times = z['timestamp_s']
    with np.load(path / 'labels.npz') as z:
        xyz = z['entity_xyz_base_m'][frames].astype(np.float32)
        np.testing.assert_array_equal(times, z['timestamp_s'])
    images = np.empty((len(frames), 3, 320, 320), dtype=np.uint8); cursor = 0; decoded = 0
    with av.open(str(path / 'rgb_left.mp4')) as video:
        stream = video.streams.video[0]; stream.thread_count = 1
        for index, frame in enumerate(video.decode(stream)):
            if cursor < len(frames) and index == frames[cursor]:
                pixels = frame.to_ndarray(format='rgb24'); assert pixels.shape == (320, 320, 3)
                images[cursor] = pixels.transpose(2, 0, 1); cursor += 1
            decoded += 1
    assert decoded == len(times) and cursor == len(frames)
    return dict(index={frame: i for i, frame in enumerate(frames)}, rgb=torch.from_numpy(images), joints=joints, xyz=xyz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e108_README.md', out / 'README.md')
    source = json.loads((TOP / 'config.json').read_text())
    assert source['status'] == 'complete' and source['train_part'] == 1
    assert source['train_points'] == 11568 and source['validation_points'] == 184
    assert source['context_stride'] == source['expert_stride'] == 25 and source['validation_stride'] == 10
    rows = {r['path']: r for group in splits().values() for r in group}
    records = {name: json.loads((TOP / f'{name}_points.json').read_text()) for name in ('train', 'val')}
    truth = {name: np.load(TOP / f'{name}_xyz.npy') for name in records}
    source_keys = {}; requests = {}
    for name, points in records.items():
        source_keys[name] = []
        assert len(points) == len(truth[name])
        top = np.load(TOP / f'{name}_features.npy', mmap_mode='r')
        assert top.shape == (len(points), 230400) and top.dtype == np.float16
        for point in points:
            row = rows[point['episode']]; frame = point['raw_frame']
            ref, raw = ((row['context_path'], frame) if frame < row['context_frames']
                        else (row['path'], frame - row['context_frames']))
            assert 0 <= raw < (row['context_frames'] if ref == row['context_path'] else row['expert_frames'])
            source_keys[name].append((ref, raw))
            requests.setdefault(ref, set()).add(raw)
    meta = dict(status='decoding', top_source=str(TOP), data=str(DATA), train_points=len(records['train']),
        val_points=len(records['val']), surrogate_camera='left', actual_wrist_camera=False,
        side_images='original320x320 current-frame RGB, no TCOW crop; ImageNet normalization',
        side_tokens=[64, 768], side_features_dtype='float16 from BF16 frozen ImageNet ViT-B/16',
        states='measured joint_position[6] from exact same context/expert raw frame; no action/XYZ/mask input',
        references=len(requests), unique_frames=sum(map(len, requests.values())),
        tcow_source_config_sha256=hashlib.sha256((TOP / 'config.json').read_bytes()).hexdigest(),
        top_backbone_sha256=json.loads((TOP / 'frozen_checks.json').read_text())['backbone_sha256'],
        vision_weights=str(VISION), vision_weights_sha256=hashlib.sha256(VISION.read_bytes()).hexdigest(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path('OATFlow/shuffle_bc/context_model.py')]})
    write_json(out / 'config.json', meta); write_json(out / 'progress.json', dict(status='decoding left RGB'))
    started = time.perf_counter(); segments = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(read_left, ref, frames): ref for ref, frames in requests.items()}
        for job in tqdm(as_completed(jobs), total=len(jobs), desc='Decode aligned left RGB/state', mininterval=5):
            segments[jobs[job]] = job.result()
    for name, points in records.items():
        measured = []; expected = []
        for point, (ref, raw) in zip(points, source_keys[name]):
            segment = segments[ref]; index = segment['index'][raw]
            measured.append(segment['joints'][index])
            expected.append(segment['xyz'][index, rows[point['episode']]['target_id']])
        np.testing.assert_allclose(np.array(expected), truth[name], rtol=0, atol=1e-7)
        np.save(out / f'{name}_joints.npy', np.array(measured, dtype=np.float32))
    encoder = FrozenWristViT().eval().cuda(); before = fingerprint(encoder)
    keys = sorted({key for values in source_keys.values() for key in values})
    indices = {key: i for i, key in enumerate(keys)}
    features = np.empty((len(keys), 64, 768), dtype=np.float16)
    for start in tqdm(range(0, len(keys), 64), desc='Cache frozen left-camera ViT', mininterval=5):
        chosen = keys[start:start + 64]
        images = torch.stack([segments[ref]['rgb'][segments[ref]['index'][raw]] for ref, raw in chosen]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16): value = encoder(images).half()
        assert torch.isfinite(value).all(); features[start:start + len(chosen)] = value.cpu().numpy()
        write_json(out / 'progress.json', dict(status='caching left ViT', completed=start + len(chosen),
            total=len(keys), elapsed_s=time.perf_counter() - started,
            gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30))
    assert fingerprint(encoder) == before and all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    for name, points in records.items():
        aligned = np.lib.format.open_memmap(out / f'{name}_wrist.npy', mode='w+', dtype=np.float16,
                                          shape=(len(points), 64, 768))
        for start in tqdm(range(0, len(points), 256), desc=f'Align {name} camera tokens', mininterval=5):
            ids = [indices[key] for key in source_keys[name][start:start + 256]]
            aligned[start:start + len(ids)] = features[ids]
        aligned.flush(); del aligned
        write_json(out / f'{name}_points.json', records[name])
    meta.update(status='complete', elapsed_s=time.perf_counter() - started, vision_module_sha256=before,
        all_labels_aligned=True, frozen_vision_unchanged=True, source_tokens_recomputed=False)
    write_json(out / 'config.json', meta); write_json(out / 'progress.json', meta)
    with (out / 'README.md').open('a') as stream:
        stream.write(f'\nCache complete: {len(keys)}unique left images/{len(requests)}segments; '
            f'{len(records["train"])}train/{len(records["val"])}val records; {meta["elapsed_s"]:.2f}s. '
            'Top tokens referenced from e107, not duplicated/recomputed. Exact raw-frame joint/XYZ alignment and unchanged frozen ViT verified.\n')
    print(json.dumps(meta), flush=True)


if __name__ == '__main__': main()
