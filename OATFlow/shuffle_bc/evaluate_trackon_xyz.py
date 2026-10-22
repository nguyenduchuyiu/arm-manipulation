"""Held-out Track-On context XYZ evaluation and recorded GT/prediction videos."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import av
import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

from OATFlow.shuffle_bc.context_cache import read_left
from OATFlow.shuffle_bc.context_model import FrozenWristViT
from OATFlow.shuffle_bc.context_video import font, marker, GREEN, PINK, ORANGE
from OATFlow.shuffle_bc.data import DATA, splits
from OATFlow.shuffle_bc.evaluate_context_xyz import cases
from OATFlow.shuffle_bc.trackon import FrozenTrackOn
from OATFlow.shuffle_bc.trackon_model import TrackOnContext
from OATFlow.shuffle_bc.train_context_xyz import metrics
from OATFlow.shuffle_bc.train_trackon_xyz import allocate, cache, batch, grouped, validation_points
from OATFlow.tracking.data import read_segment
from OATFlow.tracking.linear_probe import fingerprint
from OATFlow.tracking.point_data import project, top_camera
from OATFlow.tracking.utils import write_json

ROOT = Path('/workspace/robotics_assets/experiments/e110_trackon')


@torch.inference_mode()
def predict(model, data):
    values = []
    for start in tqdm(range(0, len(data['top']), 64), desc='Predict XYZ trajectories', mininterval=5):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model.xyz_from_features(*batch(data, slice(start, start + 64)))
        values.append((prediction * model.xyz_std + model.xyz_mean).cpu().numpy())
    result = np.concatenate(values)
    assert result.shape == data['truth'].shape and np.isfinite(result).all()
    return result


def binding(rows, locations, data, prediction):
    records = []
    for group in grouped(rows):
        assert sorted(r['target_id'] for r in group) == [0, 1, 2, 3]
        ids = [locations[r['path']][r['context_frames'] - 1] for r in group]
        assert (data['side_index'][ids] == data['side_index'][ids[0]]).all()
        torch.testing.assert_close(data['joints'][ids], data['joints'][ids[:1]].expand(4, -1), rtol=0, atol=0)
        truth = data['truth'][ids]; predicted = prediction[ids]
        nearest = np.linalg.norm(predicted[:, None] - truth[None], axis=-1).argmin(1)
        records.append(dict(context=group[0]['context_path'], group=group[0]['evaluation_group'],
            correct=(nearest == np.arange(4)).tolist(), nearest=nearest.tolist(),
            errors_mm=(np.linalg.norm(predicted - truth, axis=-1) * 1000).tolist()))
    return dict(queries=4 * len(records), contexts=len(records),
        accuracy=float(np.mean([v for r in records for v in r['correct']])),
        by_split={name: float(np.mean([v for r in records if r['group'] == name for v in r['correct']]))
                  for name in ('composition', 'spatial', 'combined')},
        same_side_and_state_verified=True, records=records)


def canvas(row, context, expert, side, raw, predicted10, predicted_best, camera, click, best_epoch, last_raw):
    nc = row['context_frames']; segment, index = (context, raw) if raw < nc else (expert, raw - nc)
    truth = segment['xyz'][index, row['target_id']]
    pixels = segment['rgb'][0, index].permute(1, 2, 0).numpy()
    left = side[0 if raw < nc else 1]['rgb'][index].permute(1, 2, 0).numpy()
    image = Image.new('RGB', (1280, 900), '#f5f7fa'); draw = ImageDraw.Draw(image)
    draw.text((30, 16), 'Track-On-R | Object-center XYZ tracking', font=font(28, True), fill='#182638')
    draw.text((30, 59), f'{row["evaluation_group"].upper()} | {row["path"]} | target: {row["target_object_id"]}',
              font=font(19), fill='#43536a')
    for x, label, prediction, color in ((30, '10 EPOCHS', predicted10, ORANGE),
                                       (650, f'BEST AFTER 100 EPOCHS (epoch {best_epoch})', predicted_best, PINK)):
        draw.text((x, 98), label, font=font(19, True), fill=color)
        image.paste(Image.fromarray(pixels).resize((600, 450)), (x, 132))
        marker(draw, project(truth, camera) * 1.875 + [x, 132], GREEN)
        marker(draw, project(prediction, camera) * 1.875 + [x, 132], color, cross=True)
        error = np.linalg.norm(prediction - truth) * 1000
        draw.text((x, 592), f'XYZ error: {error:.1f} mm', font=font(23, True), fill=color)
    image.paste(Image.fromarray(left).resize((220, 220)), (30, 635))
    draw.text((275, 638), 'INITIAL POINT PROMPT', font=font(17, True), fill='#182638')
    initial = context['rgb'][0, 0].permute(1, 2, 0).numpy()
    image.paste(Image.fromarray(initial).resize((200, 150)), (275, 672))
    marker(draw, click * .625 + [275, 672], ORANGE, radius=4)
    draw.text((650, 643), f'Time: {raw / 25:.2f}s | Phase: {segment["phase"][index]}', font=font(18), fill='#182638')
    draw.text((650, 682), f'Prediction age: {(raw - last_raw) / 25:.2f}s | inference stride 10', font=font(18), fill='#43536a')
    draw.text((650, 723), 'Green: GT center', font=font(19, True), fill=GREEN)
    draw.text((650, 758), 'Orange: 10 epochs | Pink: selected best', font=font(19), fill='#43536a')
    draw.text((650, 799), 'Left camera substitutes for a wrist camera', font=font(17), fill='#43536a')
    draw.text((30, 869), 'Recorded expert motion | XYZ prediction only | causal hold between predictions', font=font(18), fill='#43536a')
    return image


def videos(out, chosen, trajectories, best_epoch, diagnostic=False):
    folder = out / ('failure' if diagnostic else 'videos'); folder.mkdir(exist_ok=False)
    shutil.copyfile('docs/research/e110_failure_README.md' if diagnostic else 'docs/research/e110_test_README.md', folder / 'README.md')
    camera = top_camera(DATA); result = []
    for name, row in chosen:
        context = read_segment(DATA, row['context_path'], 1, views=('top',), center_crop=True)
        expert = read_segment(DATA, row['path'], 1, views=('top',), center_crop=True)
        side = [read_left(row['context_path'], range(row['context_frames'])),
                read_left(row['path'], range(row['expert_frames']))]
        trajectory = trajectories[row['path']]
        raw_index = trajectory['raw_index']; count = row['context_frames'] + row['expert_frames']
        assert count == row['frames']
        held = np.searchsorted(raw_index, np.arange(count), side='right') - 1
        assert (held >= 0).all() and (raw_index[held] <= np.arange(count)).all()
        click = project(context['xyz'][0, row['target_id']], camera)
        path = folder / f'{name}.mp4'
        writer = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '1280x900', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
            '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
        errors10 = []; errors_best = []
        for raw in tqdm(range(count), desc=f'Encode {name} video', mininterval=5):
            index = held[raw]
            a = trajectory['epoch10'][index]; b = trajectory['best'][index]
            frame = canvas(row, context, expert, side, raw, a, b, camera, click, best_epoch, int(raw_index[index]))
            writer.stdin.write(np.asarray(frame).tobytes())
            segment, local = (context, raw) if raw < row['context_frames'] else (expert, raw - row['context_frames'])
            truth = segment['xyz'][local, row['target_id']]
            errors10.append(float(np.linalg.norm(a - truth) * 1000)); errors_best.append(float(np.linalg.norm(b - truth) * 1000))
            if raw in (0, row['context_frames'] - 1):
                frame.save(folder / f'{name}_{"prompt" if raw == 0 else "shuffle"}.png')
        writer.stdin.close(); assert writer.wait() == 0
        with av.open(str(path)) as video: decoded = sum(1 for _ in video.decode(video=0))
        assert decoded == count
        result.append(dict(name=name, episode=row['path'], group=row['evaluation_group'], frames=decoded,
            duration_s=decoded / 25, prediction_points=len(raw_index),
            dense_causal_hold_mean_mm=dict(epoch10=float(np.mean(errors10)), best=float(np.mean(errors_best)))))
        write_json(folder / 'progress.json', dict(status='rendering', completed=result))
    concat = folder / 'concat.txt'
    concat.write_text(''.join(f"file '{name}.mp4'\n" for name, _ in chosen))
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat),
                    '-c', 'copy', '-movflags', '+faststart', str(folder / 'comparison.mp4')], check=True)
    concat.unlink()
    with av.open(str(folder / 'comparison.mp4')) as video: total = sum(1 for _ in video.decode(video=0))
    assert total == sum(r['frames'] for r in result)
    write_json(folder / 'summary.json', dict(status='complete', cases=result, combined_frames=total,
        all_videos_fully_decoded=True, action_control_test=False))
    with (folder / 'README.md').open('a') as stream:
        stream.write(f'\nAll {len(chosen)} individual videos and comparison.mp4 fully decoded; exact frame counts verified.\n')
    return result


def failure_video(out):
    result = json.loads((out / 'summary.json').read_text()); assert result['status'] == 'complete'
    details = json.loads((out / 'best_predictions.json').read_text())
    worst = max((r for r in details if r['phase'] == 'post_shuffle'), key=lambda r: r['error_mm'])
    row = next(r for r in json.loads((out / 'cases.json').read_text()) if r['path'] == worst['episode'])
    with np.load((out / 'trajectories' / row['path']).with_suffix('.npz')) as z:
        trajectories = {row['path']: {key: z[key].copy() for key in z.files}}
    result['failure_video'] = videos(out, [('worst_post_shuffle', row)], trajectories, result['checkpoint_epoch'], diagnostic=True)
    shutil.copy2(Path(__file__), out / 'failure' / 'source.py')
    write_json(out / 'failure' / 'config.json', dict(selection='largest post-shuffle XYZ error among all96test episodes; diagnostic only',
        selected_endpoint=worst, row=row, checkpoint_epoch=result['checkpoint_epoch'], checkpoint_step=result['checkpoint_step'],
        inference_stride=10, output_fps=25, causal_hold=True, action_control_test=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    write_json(out / 'summary.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--failure-video', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    if args.failure_video:
        failure_video(args.output)
        return
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e110_test_README.md', out / 'README.md')
    checkpoints = dict(epoch10=ROOT / 'train/best.pt', best=ROOT / 'train_2/best.pt')
    saved = {name: torch.load(path, map_location='cpu', weights_only=False, mmap=True) for name, path in checkpoints.items()}
    assert saved['epoch10']['epoch'] == 10 and saved['best']['epoch'] == 79
    assert saved['epoch10']['config']['frozen_sha256'] == saved['best']['config']['frozen_sha256']
    rows = cases(); groups = splits()
    chosen = [('train', groups['train'][0]), ('validation', groups['val'][0])]
    chosen += [(f'test_{name}', next(r for r in rows if r['evaluation_group'] == name))
               for name in ('composition', 'spatial', 'combined')]
    all_rows = rows + [r for _, r in chosen[:2]]
    forbidden = {r['path'] for name in ('train', 'val') for r in groups[name]}
    assert not ({r['path'] for r in rows} & forbidden)
    config = dict(status='running', data=str(DATA), checkpoint={name: dict(path=str(path), epoch=saved[name]['epoch'],
        step=saved[name]['step'], sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in checkpoints.items()},
        primary='best selected on validation before this test; epoch10 is a fixed baseline',
        episodes=96, cases=rows, video_cases=[dict(name=name, row=row) for name, row in chosen],
        tracker_stride=10, frame0_and_segment_terminals=True, support_grid=20, memory=72,
        endpoint_protocol='same fixed phase endpoints as e108, 32new-order/32new-layout/32both',
        surrogate_camera='left', actual_wrist_camera=False, action_control_test=False,
        H_K_flow_noise_frame_budget='not applicable: recorded RGB/XYZ inference only',
        video_fps=25, video_size=[1280, 900], video_prediction='causal hold, no future interpolation',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path('OATFlow/shuffle_bc/train_trackon_xyz.py'), Path('OATFlow/shuffle_bc/trackon.py'),
             Path('OATFlow/shuffle_bc/trackon_model.py'), Path('OATFlow/shuffle_bc/context_cache.py'))})
    write_json(out / 'config.json', config); write_json(out / 'cases.json', rows)
    for name in config['source_sha256']:
        p = Path(name).resolve(); destination = out / 'source' / p.relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, destination)
    tracker = FrozenTrackOn().cuda(); encoder = FrozenWristViT().cuda()
    expected = saved['best']['config']['frozen_sha256']
    assert expected == dict(tracker=fingerprint(tracker), wrist=fingerprint(encoder))
    data = allocate(all_rows); locations = data['locations']
    started = time.perf_counter(); cache(tracker, encoder, data, all_rows, out, 'test')
    assert expected == dict(tracker=fingerprint(tracker), wrist=fingerprint(encoder))
    del tracker, encoder; torch.cuda.empty_cache()
    cache_seconds = time.perf_counter() - started
    predictions = {}
    for name in ('epoch10', 'best'):
        model = TrackOnContext(); model.load_context_state(saved[name]['model']); model.requires_grad_(False).eval().cuda()
        predictions[name] = predict(model, data)
        del model; torch.cuda.empty_cache()
    plan = validation_points(rows); ids = []; records = []
    for row in rows:
        for raw, phase in plan[row['path']].items():
            ids.append(locations[row['path']][raw])
            records.append(dict(episode=row['path'], group=row['evaluation_group'], raw_frame=raw, phase=phase))
    assert len(ids) == 540
    summary = {}; probes = {}
    for name, values in predictions.items():
        total, detail = metrics(values[ids], data['truth'][ids], records)
        by_split = {}
        for split in ('composition', 'spatial', 'combined'):
            chosen_ids = [i for i, record in enumerate(records) if record['group'] == split]
            by_split[split], _ = metrics(values[np.array(ids)[chosen_ids]], data['truth'][np.array(ids)[chosen_ids]],
                                        [records[i] for i in chosen_ids])
        summary[name] = dict(metrics=total, by_split=by_split)
        write_json(out / f'{name}_predictions.json', detail)
        probes[name] = binding(rows, locations, data, values)
    # Check the same validation case against its original cached predictions.
    reference = json.loads((ROOT / 'train_2/best_validation.json').read_text())['records']
    differences = []
    for point in reference:
        if point['episode'] == chosen[1][1]['path']:
            current = predictions['best'][locations[point['episode']][point['raw_frame']]]
            differences.append(float(np.linalg.norm(current - point['prediction_base_m']) * 1000))
    assert differences and max(differences) < .5
    parity = dict(points=len(differences), max_prediction_difference_mm=max(differences),
                  mean_prediction_difference_mm=float(np.mean(differences)))
    trajectories = {}
    for row in all_rows:
        raw = np.array(list(locations[row['path']]), dtype=np.int64)
        selected = np.array(list(locations[row['path']].values()), dtype=np.int64)
        trajectory = dict(raw_index=raw, truth=data['truth'][selected],
            epoch10=predictions['epoch10'][selected], best=predictions['best'][selected])
        trajectories[row['path']] = trajectory
        path = out / 'trajectories' / row['path']; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix('.npz'), **trajectory)
    result = dict(status='metrics_complete', episodes=96, endpoints=len(ids),
        checkpoint_epoch=saved['best']['epoch'], checkpoint_step=saved['best']['step'],
        results=summary, target_binding=probes, online_cache_parity=parity,
        feature_seconds=cache_seconds, action_control_test=False)
    write_json(out / 'summary.json', result); write_json(out / 'progress.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'target_binding'}), flush=True)
    del data, saved
    # Retain small prediction trajectories; discard the temporary native feature cache.
    assert (out / 'query_cache').is_dir()
    shutil.rmtree(out / 'query_cache')
    result['videos'] = videos(out, chosen, trajectories, result['checkpoint_epoch'])
    result['status'] = config['status'] = 'complete'
    write_json(out / 'summary.json', result); write_json(out / 'progress.json', result)
    write_json(out / 'config.json', config)
    with (out / 'README.md').open('a') as stream:
        stream.write('\nComplete: 96held-out episodes/540phase endpoints; both checkpoints evaluated on identical frozen features. '
                     'Five fixed demo videos and comparison.mp4 fully decoded. Temporary query features removed. Results in summary.json.\n')


if __name__ == '__main__': main()
