"""Causal target-mask videos: GT, original TCOW, and fine-tuned TCOW."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
from tqdm import tqdm

from OATFlow.tracking.mask import TargetMaskTCOW, load_original
from OATFlow.tracking.mask_data import batch, load, splits
from OATFlow.tracking.train_mask import evaluate
from OATFlow.tracking.utils import write_json


DATA = Path('/workspace/robotics_assets/datasets/shuffle_pick_multiview')
TRAIN = Path('/workspace/robotics_assets/experiments/e114_mask/train_3')
WEIGHTS = Path('/workspace/robotics_assets/pretrained/vision')
COLORS = ((24, 171, 102), (229, 139, 23), (213, 45, 141))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def cases(groups):
    selected = []
    for group, target, name, label in (
            ('train', 0, 'train', 'TRAIN REFERENCE'),
            ('composition', 1, 'unseen_permutation', 'TEST: UNSEEN PERMUTATION'),
            ('spatial', 2, 'unseen_layout', 'TEST: UNSEEN LAYOUT'),
            ('combined', 3, 'both_unseen', 'TEST: UNSEEN LAYOUT + PERMUTATION')):
        row = next(row for row in groups[group] if row['target_id'] == target)
        selected.append(dict(name=name, label=label, row=row,
                             selection='first episode in split with predetermined target0/1/2/3'))
    records = json.loads((TRAIN / 'test_combined.json').read_text())['records']
    hidden = {}
    for record in records:
        if record['hidden']:
            hidden.setdefault(record['path'], []).append(record['iou'])
    worst = min(hidden, key=lambda path: np.mean(hidden[path]))
    row = next(row for row in groups['combined'] if row['path'] == worst)
    selected.append(dict(name='hard_case', label='TEST: HARDEST HIDDEN CASE IN EVALUATED COMBINED SUBSET',
                         row=row, selection='lowest mean hidden-probe IoU among64combined test episodes'))
    return selected


def dense_episode(row):
    images, masks, hidden, phases = [], [], [], []
    for ref, context in ((row['context_path'], True), (row['path'], False)):
        path = DATA / ref
        frames = []
        with av.open(str(path / 'rgb_top.mp4')) as video:
            video.streams.video[0].thread_count = 1
            for frame in video.decode(video=0):
                frames.append(frame.to_ndarray(format='rgb24')[40:280])
        images.append(np.stack(frames))
        with np.load(path / 'masks.npz') as labels:
            if context:
                packed = labels['amodal_packed'][:, 0, row['target_id'], 40:280].copy()
                fraction = labels['occlusion_fraction'][:, 0, row['target_id']]
            else:
                packed = labels['tcow_packed'][:, 0, 0, 40:280].copy()
                fraction = labels['occlusion_fraction'][:, 0, 0]
            masks.append(packed)
            hidden.append(fraction >= .95)
        with np.load(path / 'supervision.npz') as labels:
            phases.append(labels['phase'])
        assert len(frames) == len(packed) == len(phases[-1])
    result = dict(rgb=np.concatenate(images), packed=np.concatenate(masks),
                  hidden=np.concatenate(hidden), phase=np.concatenate(phases))
    assert len(result['rgb']) == row['frames']
    return result


@torch.inference_mode()
def predict(model, episode, description):
    outputs = []
    for start in tqdm(range(0, len(episode['raw_index']), 4), desc=description, mininterval=5):
        points = [(0, end) for end in range(start, min(start + 4, len(episode['raw_index'])))]
        rgb, query, _, _, _ = batch([episode], points)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(rgb, query)
        assert torch.isfinite(logits).all()
        binary = (logits[:, -1] > 0).cpu().numpy()
        outputs.append(np.packbits(binary, axis=-1, bitorder='little'))
    return np.concatenate(outputs)


def scores(dense, packed, held):
    values = []
    for raw in range(len(dense['packed'])):
        truth = np.unpackbits(dense['packed'][raw], axis=-1, count=320, bitorder='little').astype(bool)
        prediction = np.unpackbits(packed[held[raw]], axis=-1, count=320, bitorder='little').astype(bool)
        union = np.count_nonzero(truth | prediction)
        values.append(np.count_nonzero(truth & prediction) / union if union else 1.)
    return np.asarray(values, np.float32)


@lru_cache(maxsize=None)
def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/' + name, size)


def overlay(pixels, region, color):
    result = pixels.astype(np.float32).copy()
    result[region] = .65 * result[region] + .35 * np.asarray(color)
    eroded = np.asarray(Image.fromarray(region.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))) > 0
    result[region & ~eroded] = color
    return Image.fromarray(result.astype(np.uint8))


def canvas(case, dense, raw, baseline, tuned, original_iou, tuned_iou, prompt):
    row = case['row']
    truth = np.unpackbits(dense['packed'][raw], axis=-1, count=320, bitorder='little').astype(bool)
    pixels = dense['rgb'][raw]
    panels = [overlay(pixels, region, color) for region, color in zip((truth, baseline, tuned), COLORS)]
    image = Image.new('RGB', (1536, 864), '#f5f7fb')
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), 'e114 | Target-mask tracking', font=font(30, True), fill='#17283d')
    draw.text((24, 64), f'{case["label"]}  |  {row["path"]}  |  target: {row["target_object_id"]}',
              font=font(18), fill='#48596d')
    labels = ('GROUND TRUTH: AMODAL TARGET', 'ORIGINAL TCOW', 'FINE-TUNED TCOW: EPOCH 9')
    yy, xx = np.nonzero(truth)
    cx = int(np.clip(np.mean(xx), 48, 320-48))
    cy = int(np.clip(np.mean(yy), 48, 240-48))
    roi = (cx-48, cy-48, cx+48, cy+48)
    for column, (panel, label, color) in enumerate(zip(panels, labels, COLORS)):
        x = 24 + column * 504
        draw.text((x, 108), label, font=font(18, True), fill=color)
        image.paste(panel.resize((480, 360), Image.Resampling.BICUBIC), (x, 139))
        draw.rectangle((x, 139, x+479, 498), outline='#d6deea', width=1)
        value = ('Target extent behind occluders' if column == 0 else
                 f'Frame IoU: {(original_iou if column == 1 else tuned_iou):.3f}')
        draw.text((x, 515), value, font=font(21, True), fill=color)
        draw.text((x, 558), 'TARGET-AREA ZOOM', font=font(15, True), fill='#48596d')
        image.paste(panel.crop(roi).resize((192, 192), Image.Resampling.NEAREST), (x, 588))
        if column == 0:
            draw.text((x+212, 589), 'INITIAL MASK PROMPT', font=font(15, True), fill='#48596d')
            first = overlay(dense['rgb'][0], prompt, COLORS[0])
            image.paste(first.resize((240, 180), Image.Resampling.BICUBIC), (x+212, 615))
        elif column == 1:
            draw.text((x+213, 590), 'Official pretrained', font=font(18), fill='#48596d')
            draw.text((x+213, 620), 'weights unchanged', font=font(18), fill='#48596d')
        else:
            draw.text((x+213, 590), 'Mask threshold: 0.5', font=font(18), fill='#48596d')
            draw.text((x+213, 620), 'Frame-0 query only', font=font(18), fill='#48596d')
    visibility = 'FULLY OCCLUDED' if dense['hidden'][raw] else 'VISIBLE / PARTLY OCCLUDED'
    phase = str(dense['phase'][raw]).replace('_', ' ')
    draw.text((24, 806), f'{raw/25:5.2f} s  |  {phase}  |  {visibility}', font=font(18, True), fill='#17283d')
    draw.text((24, 837), 'Recorded expert motion; no FM control. RGB + initial mask only. 25 fps playback; causal predictions updated at stride 3.',
              font=font(16), fill='#48596d')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(114)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e114_test_README.md', out / 'README.md')
    checkpoint = TRAIN / 'best.pt'
    checkpoint_hash = digest(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert (saved['epoch'], saved['step']) == (9, 4608)
    model = TargetMaskTCOW()
    model.load_state_dict(saved['model'], strict=True)
    model.requires_grad_(False).eval().cuda()
    original = TargetMaskTCOW()
    load_original(original, WEIGHTS)
    original.requires_grad_(False).eval().cuda()
    groups = splits(DATA)
    selected = cases(groups)
    config = dict(status='running', checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        epoch=saved['epoch'], step=saved['step'], cases=selected, threshold=.5,
        inference_stride=3, output_fps=25, output_size=[1536, 864], causal_hold=True,
        future_interpolation=False, input='top RGB causal30frame prefix and visible target mask at frame0 only',
        supervision_only='all later GT masks; also used for metrics and display crop, never model input',
        baseline='original TCOW backbone and original target-mask head; no fine-tuning',
        action_control_test=False, source_sha256=digest(Path(__file__)))
    write_json(out / 'config.json', config)
    # Reproduce the selected checkpoint's validation before drawing videos.
    val, _ = load(DATA, groups['val'])
    result, _ = evaluate(model, val, 4, 'Replay best checkpoint validation')
    expected = json.loads((TRAIN / 'best_validation.json').read_text())['metrics']
    difference = abs(result['all']['mean_iou'] - expected['all']['mean_iou'])
    assert difference < 1e-7, (result, expected)
    write_json(out / 'validation_replay.json', dict(metrics=result, saved_mean_iou=expected['all']['mean_iou'], difference=difference))
    del val
    summaries = []
    for case in selected:
        row, name = case['row'], case['name']
        episodes, _ = load(DATA, [row])
        episode = episodes[0]
        dense = dense_episode(row)
        predicted = predict(model, episode, name + ' fine-tuned masks')
        baseline = predict(original, episode, name + ' original masks')
        held = np.searchsorted(episode['raw_index'], np.arange(row['frames']), side='right') - 1
        assert (episode['raw_index'][held] <= np.arange(row['frames'])).all()
        original_iou = scores(dense, baseline, held)
        tuned_iou = scores(dense, predicted, held)
        np.savez_compressed(out / f'{name}.npz', prediction_packed=predicted, original_packed=baseline,
            prediction_raw_index=episode['raw_index'], truth_packed=dense['packed'], hidden=dense['hidden'],
            phase=dense['phase'], original_iou=original_iou, fine_tuned_iou=tuned_iou,
            prompt=episode['context']['prompt'][row['target_id']], bitorder='little')
        writer = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '1536x864', '-r', '25', '-i', '-', '-an',
            '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '19',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / f'{name}.mp4')], stdin=subprocess.PIPE)
        prompt = episode['context']['prompt'][row['target_id']]
        snapshots = {0: 'prompt', row['context_frames']-1: 'shuffle', row['decision_frames']['t_obj']: 'exposed'}
        for raw in tqdm(range(row['frames']), desc=name + ' encode25FPS', mininterval=5):
            initial = np.unpackbits(baseline[held[raw]], axis=-1, count=320, bitorder='little').astype(bool)
            tuned = np.unpackbits(predicted[held[raw]], axis=-1, count=320, bitorder='little').astype(bool)
            image = canvas(case, dense, raw, initial, tuned, original_iou[raw], tuned_iou[raw], prompt)
            writer.stdin.write(np.asarray(image).tobytes())
            if raw in snapshots:
                image.save(out / f'{name}_{snapshots[raw]}.png')
        writer.stdin.close()
        assert writer.wait() == 0
        with av.open(str(out / f'{name}.mp4')) as video:
            decoded = sum(1 for _ in video.decode(video=0))
        assert decoded == row['frames']
        def score(values):
            return dict(all_mean_iou=float(values.mean()), hidden_mean_iou=float(values[dense['hidden']].mean()),
                        visible_mean_iou=float(values[~dense['hidden']].mean()))
        summaries.append(dict(name=name, episode=row['path'], frames=decoded, predictions=len(predicted),
                              duration_s=decoded/25, original=score(original_iou), fine_tuned=score(tuned_iou)))
        write_json(out / 'progress.json', dict(status='running', completed=summaries))
        print('VIDEO', json.dumps(summaries[-1]), flush=True)
    concat = out / 'concat.txt'
    concat.write_text(''.join(f"file '{case['name']}.mp4'\n" for case in selected))
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat),
                    '-c', 'copy', '-movflags', '+faststart', str(out / 'comparison.mp4')], check=True)
    concat.unlink()
    with av.open(str(out / 'comparison.mp4')) as video:
        combined_frames = sum(1 for _ in video.decode(video=0))
    assert combined_frames == sum(case['row']['frames'] for case in selected)
    assert digest(checkpoint) == checkpoint_hash
    result = dict(status='complete', cases=summaries, combined_frames=combined_frames,
                  all_videos_fully_decoded=True, checkpoint_unchanged=True,
                  validation_replay_difference=difference, action_control_test=False)
    write_json(out / 'summary.json', result)
    write_json(out / 'progress.json', result)
    config['status'] = 'complete'
    write_json(out / 'config.json', config)
    with (out / 'README.md').open('a') as stream:
        stream.write('\nStatus: complete. Five videos and comparison fully decoded; frame counts checked, checkpoint unchanged, saved validation reproduced. See summary.json for dense causal-hold IoU on each video.\n')
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
