"""Recorded train/validation/both-unseen videos for e108 XYZ localization."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

from OATFlow.shuffle_bc.context_cache import read_left, TOP
from OATFlow.shuffle_bc.context_model import ContextPolicy
from OATFlow.shuffle_bc.data import DATA, splits
from OATFlow.shuffle_bc.evaluate_context_xyz import cases
from OATFlow.tracking.point_data import load_points, project, top_camera
from OATFlow.tracking.pretrain_xyz import inputs
from OATFlow.tracking.utils import write_json

ROOT = Path('/workspace/robotics_assets/experiments/e108_context')
GREEN, PINK, ORANGE = '#15975e', '#d32f94', '#df8700'


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/' + name, size)


def marker(draw, uv, color, cross=False, radius=7):
    x, y = uv
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=3)
    if cross:
        draw.line((x-radius-4, y, x+radius+4, y), fill=color, width=2)
        draw.line((x, y-radius-4, x, y+radius+4), fill=color, width=2)


def frame_canvas(ep, side, raw, prediction, dense, label, camera):
    row = ep['row']; nc = row['context_frames']
    segment, index = (ep['context'], raw) if raw < nc else (ep['expert'], raw-nc)
    ref = row['context_path'] if raw < nc else row['path']
    image = segment['rgb'][0, index].permute(1, 2, 0).numpy()
    canvas = Image.new('RGB', (1280, 800), '#f5f7fa'); draw = ImageDraw.Draw(canvas)
    draw.text((30, 18), 'e108 | Shared-context XYZ tracking', font=font(28, True), fill='#182638')
    draw.text((30, 61), f'{label}  |  {row["path"]}  |  target: {row["target_object_id"]}', font=font(19), fill='#43536a')
    draw.text((30, 98), 'TOP VIEW', font=font(18, True), fill='#182638')
    draw.text((850, 98), 'LEFT VIEW (surrogate wrist)', font=font(17, True), fill='#182638')
    canvas.paste(Image.fromarray(image).resize((800, 600)), (30, 130))
    left = side[ref]['rgb'][side[ref]['index'][index]].permute(1, 2, 0).numpy()
    canvas.paste(Image.fromarray(left).resize((360, 360)), (850, 130))
    for xyz, color, cross in ((dense['xyz'][raw], GREEN, False), (prediction, PINK, True)):
        uv = project(xyz, camera) * 2.5 + [30, 130]
        marker(draw, uv, color, cross)
    draw.text((850, 507), 'INITIAL POINT PROMPT', font=font(16, True), fill='#182638')
    initial = ep['context']['rgb'][0, 0].permute(1, 2, 0).numpy()
    canvas.paste(Image.fromarray(initial).resize((160, 120)), (850, 538))
    marker(draw, ep['query_uv'] * .5 + [850, 538], ORANGE, radius=4)
    error = np.linalg.norm(prediction-dense['xyz'][raw])*1000
    draw.text((1030, 542), f'{raw/25:.2f} s', font=font(23, True), fill='#182638')
    phase = str(dense['phase'][raw]).replace('_', ' ')
    draw.text((1030, 579), phase, font=font(17), fill='#43536a')
    draw.text((850, 680), f'XYZ error: {error:.1f} mm', font=font(25, True), fill=PINK)
    draw.text((30, 750), 'GT center', font=font(19, True), fill=GREEN)
    draw.text((175, 750), 'Predicted center', font=font(19, True), fill=PINK)
    draw.text((405, 750), 'Recorded expert motion | XYZ tracking only; FM not controlling robot', font=font(18), fill='#43536a')
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e108_video_README.md', out / 'README.md')
    train_name = json.loads((TOP / 'train_points.json').read_text())[0]['episode']
    groups = splits()
    rows = [next(r for r in groups['train'] if r['path'] == train_name), groups['val'][0],
            next(r for r in cases() if r['evaluation_group'] == 'combined')]
    labels = ['TRAIN (part1)', 'VALIDATION', 'TEST (unseen layout + order)']
    names = ['train', 'validation', 'test_unseen']
    checkpoint = ROOT / 'train' / 'best.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = dict(status='running', checkpoint=str(checkpoint), epoch=saved['epoch'], step=saved['step'],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), cases=rows,
        inference_stride=3, output_fps=25, causal_prediction_hold=True, future_interpolation=False,
        video_size=[1280, 800], actual_wrist_camera=False, action_control_test=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out / 'config.json', config)
    episodes, _ = load_points(DATA, rows, stride=3)
    dense_episodes, _ = load_points(DATA, rows, stride=1)
    side = {ref: read_left(ref, range(row[count])) for row in rows for ref, count in
            ((row['context_path'], 'context_frames'), (row['path'], 'expert_frames'))}
    model = ContextPolicy(); model.load_context_state(saved['model']); model.eval().cuda()
    camera = top_camera(DATA); summaries = []
    for i, ep in enumerate(episodes):
        values = []; row = ep['row']; dense = dense_episodes[i]
        for start in tqdm(range(0, len(ep['raw_index']), 8), desc=f'{names[i]} XYZ inference', mininterval=5):
            ends = list(range(start, min(start+8, len(ep['raw_index']))))
            rgb, query, _ = inputs(episodes, [(i, end) for end in ends])
            refs = [(row['context_path'], int(ep['raw_index'][end])) if ep['raw_index'][end] < row['context_frames']
                    else (row['path'], int(ep['raw_index'][end]-row['context_frames'])) for end in ends]
            left = torch.stack([side[ref]['rgb'][side[ref]['index'][raw]] for ref, raw in refs]).cuda()
            joints = torch.tensor(np.array([side[ref]['joints'][side[ref]['index'][raw]] for ref, raw in refs]), device='cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                top, wrist = model.encode(rgb, query, left)
                pred = model.xyz_from_features(top, wrist, joints) * model.xyz_std + model.xyz_mean
            values.extend(pred.cpu().numpy())
        prediction = np.array(values); assert np.isfinite(prediction).all()
        np.savez_compressed(out / f'{names[i]}.npz', prediction_base_m=prediction, raw_index=ep['raw_index'],
            truth_base_m=dense['xyz'], truth_raw_index=dense['raw_index'], query_uv=ep['query_uv'])
        held = np.searchsorted(ep['raw_index'], dense['raw_index'], side='right')-1
        assert (ep['raw_index'][held] <= dense['raw_index']).all()
        path = out / f'{names[i]}.mp4'
        writer = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '1280x800', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast',
            '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
        for raw in tqdm(range(row['frames']), desc=f'{names[i]} encode25FPS', mininterval=5):
            image = frame_canvas(dense, side, raw, prediction[held[raw]], dense, labels[i], camera)
            writer.stdin.write(np.asarray(image).tobytes())
            if raw == 0 or raw == row['context_frames']-1:
                image.save(out / f'{names[i]}_{"prompt" if raw == 0 else "shuffle"}.png')
        writer.stdin.close(); assert writer.wait() == 0
        with av.open(str(path)) as video:
            decoded = sum(1 for _ in video.decode(video=0))
        assert decoded == row['frames']
        errors = np.linalg.norm(prediction[held]-dense['xyz'], axis=-1)*1000
        summaries.append(dict(name=names[i], episode=row['path'], frames=decoded, duration_s=decoded/25,
            predictions=len(prediction), dense_causal_hold_mean_mm=float(errors.mean()),
            dense_causal_hold_median_mm=float(np.median(errors))))
        write_json(out / 'progress.json', dict(status='running', completed=summaries))
    concat = out / 'concat.txt'
    concat.write_text(''.join(f"file '{name}.mp4'\n" for name in names))
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat),
                    '-c', 'copy', '-movflags', '+faststart', str(out / 'comparison.mp4')], check=True)
    concat.unlink()
    with av.open(str(out / 'comparison.mp4')) as video:
        decoded = sum(1 for _ in video.decode(video=0))
    assert decoded == sum(row['frames'] for row in rows)
    result = dict(status='complete', cases=summaries, combined_frames=decoded,
                  all_videos_fully_decoded=True, action_control_test=False)
    write_json(out / 'summary.json', result); write_json(out / 'progress.json', result)
    config['status'] = 'complete'; write_json(out / 'config.json', config)
    with (out / 'README.md').open('a') as stream:
        stream.write('\nStatus: complete. All three MP4s and comparison.mp4 fully decoded and frame counts verified.\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
