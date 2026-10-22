"""Unmodified pretrained iSEE on four full recorded shuffle/expert top videos."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import timm
import torch
import torch.nn.functional as F
from tqdm import tqdm

from isee.config import build_model, build_walkers, grid_side, load_config, set_precision
from isee.model import constants as C
from isee.visualize import PALETTE, to_pixels
from OATFlow.shuffle_bc.data import DATA, splits
from OATFlow.simulation.io import encoder, write_json
from OATFlow.tracking.data import read_segment
from OATFlow.tracking.point_data import project, top_camera

REPO = Path('/workspace/iSEE')
CASES = [('train', 'g000/p00/t0'), ('unseen_order', 'g000/p01/t0'),
         ('unseen_layout', 'g036/p00/t0'), ('unseen_both', 'g036/p01/t0')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def marker(draw, xy, color):
    x, y = xy
    draw.ellipse((x-8, y-8, x+8, y+8), outline=color, width=3)
    draw.line((x-12, y, x+12, y), fill=color, width=2)
    draw.line((x, y-12, x, y+12), fill=color, width=2)


def canvas(rgb, segmentation, selected, pred, truth, held, name, row, t, phase, error):
    image = Image.new('RGB', (1280, 640), '#f4f6fa')
    draw = ImageDraw.Draw(image)
    draw.text((18, 10), 'iSEE | Original pretrained weights | No fine-tuning', font=font(25), fill='#17283f')
    draw.text((18, 46), f'{name} | {row["path"]} | RedCube | {t/25:.2f}s | {phase}',
              font=font(18), fill='#41546a')
    draw.text((18, 77), 'NATIVE 2D READOUT', font=font(17), fill='#17283f')
    draw.text((658, 77), f'OBJECT SLOTS | fixed query slot {selected}', font=font(17), fill='#17283f')
    image.paste(Image.fromarray(rgb).resize((640, 480)), (0, 106))
    overlay = (.48 * PALETTE[segmentation] + .52 * rgb).astype(np.uint8)
    region = segmentation == selected
    interior = region & np.roll(region, 1, 0) & np.roll(region, -1, 0)
    interior &= np.roll(region, 1, 1) & np.roll(region, -1, 1)
    overlay[region & ~interior] = 255
    image.paste(Image.fromarray(overlay).resize((640, 480)), (640, 106))
    for offset in (0, 640):
        marker(draw, truth * 2 + [offset, 106], '#00d885')
        marker(draw, pred * 2 + [offset, 106], '#ff39ae')
    draw.text((18, 598), 'Green: GT body center   Magenta: native slot center', font=font(17), fill='#17283f')
    draw.text((680, 598), f'{"HELD / WALKER" if held else "CORRECTOR"} | center error {error:.1f}px',
              font=font(17), fill='#17283f')
    return image


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert timm.__version__ == '1.0.30', 'Use the isolated official-version timm via PYTHONPATH.'
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    set_precision(True)
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e112_README.md', out / 'README.md')
    cfg_path = REPO / 'configs/lacater_static.yaml'
    cfg = load_config(cfg_path); side = grid_side(cfg)
    weights = [REPO / cfg['model']['checkpoint'],
               *[REPO / p for p in cfg['walker']['checkpoints']],
               REPO / 'checkpoints/nope_coco_dv2_vits14_reg4.pth']
    hashes = {str(p): sha(p) for p in weights}
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    original_sources = {str(p.relative_to(REPO)): sha(p) for p in (REPO / 'isee').rglob('*.py')}
    config = dict(status='running', upstream=str(REPO), commit=commit, original_config=cfg,
        weights_sha256=hashes, upstream_source_sha256=original_sources, seed=0, timm=timm.__version__,
        torch=torch.__version__, dtype='float32; TF32 enabled', rgb_stride=1, fps=25,
        crop_y=[40, 280], resize='bicubic antialias=True, 518x518, ImageNet normalization',
        encoder_batch=8, source_sha256=sha(__file__), trained=False, action_control=False,
        slot_selection='argmax bilinear-upsampled corrector grouping at frame0 click; fixed thereafter',
        readout='official infer.py decoder.pose; corrector visible, walker held', cases=CASES)
    write_json(out / 'config.json', config)
    shutil.copy2(__file__, out / 'evaluate_isee.py')
    shutil.copy2(cfg_path, out / 'lacater_static.yaml')
    model = build_model(cfg, 'cuda').eval().requires_grad_(False)
    walkers, dmax = build_walkers(cfg, 'cuda')
    assert not any(p.requires_grad for p in model.parameters())
    rows = {r['path']: r for group in splits().values() for r in group}
    camera = top_camera(DATA); results = []
    mean = torch.tensor(C.IMAGENET_MEAN, device='cuda')[None, :, None, None]
    std = torch.tensor(C.IMAGENET_STD, device='cuda')[None, :, None, None]
    for name, ref in CASES:
        row = rows[ref]; target = row['target_id']
        parts = [read_segment(DATA, p, 1, views=('top',), center_crop=True)
                 for p in (row['context_path'], ref)]
        rgb = torch.cat([p['rgb'][0] for p in parts])
        truth = project(np.concatenate([p['xyz'][:, target] for p in parts]), camera)
        phase = np.concatenate([p['phase'] for p in parts])
        T = len(rgb); assert T == row['frames']
        torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        chunks = []
        for start in tqdm(range(0, T, 8), desc=f'{name} DINO', mininterval=5):
            x = rgb[start:start+8].cuda().float() / 255
            x = F.interpolate(x, size=(518, 518), mode='bicubic', align_corners=False, antialias=True)
            chunks.append(model.encode(((x-mean)/std)[None], chunk=8))
        h = torch.cat(chunks, 1); del chunks, x
        assert h.shape == (1, T, side*side, 64)
        torch.cuda.synchronize(); encoder_seconds = time.perf_counter()-started
        bar = tqdm(total=T, desc=f'{name} slots/TEN/walkers', mininterval=5)
        hook = model.corrector.register_forward_hook(lambda *_: bar.update(1))
        started = time.perf_counter()
        values = model(h, walkers, dmax, decode=False)
        torch.cuda.synchronize(); recurrent_seconds = time.perf_counter()-started
        hook.remove(); bar.close(); del h
        # Decode each frame through the unchanged native decoder, without retaining feature reconstructions.
        masks = []
        for t in tqdm(range(T), desc=f'{name} native mask decoder', mininterval=5):
            masks.append(model.decoder(values['state_updated'][:, t])['masks'][0].half().cpu())
        masks = torch.stack(masks).numpy().reshape(T, 11, side, side)
        position = torch.where(values['held'][0, ..., None], values['position_walk'][0],
                               values['state_updated'][0, ..., C.APPEARANCE_DIM:])
        mu, _ = model.decoder.pose(position)
        centers = to_pixels(mu.cpu().numpy(), side, 240, 320)
        grouping = values['grouping'][0].float().cpu().reshape(T, 11, side, side)
        grouping_up = F.interpolate(grouping, (240, 320), mode='bilinear', align_corners=False)
        regions = grouping_up.argmax(1).numpy().astype(np.uint8)
        qx, qy = np.rint(truth[0]).astype(int)
        selected = int(regions[0, qy, qx])
        click_distribution = grouping_up[0, :, qy, qx].numpy()
        del grouping_up
        held = values['held'][0].cpu().numpy()
        prediction = centers[:, selected]
        errors = np.linalg.norm(prediction-truth, axis=-1)
        np.savez_compressed(out / f'{name}.npz', center_xy=centers, truth_xy=truth,
            held=held, position=position.cpu().numpy(), selected_slot=selected, query_uv=truth[0],
            grouping=grouping.half().numpy(), masks=masks,
            **{k: values[k][0].cpu().numpy() for k in ('state', 'state_updated', 'written')})
        coverage = []; iou = []; visible = []; owner = []
        for ref_part in (row['context_path'], ref):
            with np.load(DATA / ref_part / 'masks.npz') as z:
                modal = z['instance'][:, 0, 40:280] == target+1
                hidden = z['occlusion_fraction'][:, 0, target if ref_part == row['context_path'] else 0]
            offset = len(coverage)
            for j, mask in enumerate(modal):
                slot_mask = regions[offset+j] == selected
                intersection = (mask & slot_mask).sum()
                count = mask.sum()
                coverage.append(float(intersection / max(1, count)))
                iou.append(float(intersection / max(1, (mask | slot_mask).sum())))
                visible.append(bool(hidden[j] <= .5 and count >= 20))
                owner.append(int(np.bincount(regions[offset+j][mask], minlength=11).argmax()) if count else -1)
        visible = np.array(visible); owner = np.array(owner); coverage = np.array(coverage); iou = np.array(iou)
        assert visible.any()
        by_phase = {str(p): dict(frames=int((phase == p).sum()),
                    mean_center_error_px=float(errors[phase == p].mean()),
                    selected_slot_held_fraction=float(held[phase == p, selected].mean())) for p in np.unique(phase)}
        summary = dict(name=name, episode=ref, frames=T, selected_slot=selected,
            frame0_click_probabilities=click_distribution.tolist(), initial_center_error_px=float(errors[0]),
            mean_center_error_px=float(errors.mean()), median_center_error_px=float(np.median(errors)),
            final_center_error_px=float(errors[-1]),
            initial_target_coverage=float(coverage[0]), initial_slot_iou=float(iou[0]),
            visible_target_coverage=float(coverage[visible].mean()),
            visible_target_owner_same_slot_fraction=float((owner[visible] == selected).mean()),
            final_visible_owner=int(owner[np.flatnonzero(visible)[-1]]),
            selected_slot_held_fraction=float(held[:, selected].mean()),
            encoder_seconds=encoder_seconds, recurrent_seconds=recurrent_seconds,
            peak_vram_gib=torch.cuda.max_memory_allocated()/2**30, by_phase=by_phase)
        np.savez_compressed(out / f'{name}_metrics.npz', coverage=coverage, slot_iou=iou,
            at_least_half_visible=visible, majority_owner=owner, center_error_px=errors, phase=phase)
        writer = encoder(out / f'{name}.mp4', 1280, 640)
        shots = {0, row['context_frames']-1, T-1}
        for t in tqdm(range(T), desc=f'{name} video 25FPS', mininterval=5):
            image = canvas(rgb[t].permute(1, 2, 0).numpy(), regions[t], selected, prediction[t], truth[t],
                           held[t, selected], name, row, t, str(phase[t]), errors[t])
            writer.stdin.write(np.asarray(image).tobytes())
            if t in shots:
                image.save(out / f'{name}_{t:04d}.png')
        writer.stdin.close(); assert writer.wait() == 0
        with av.open(str(out / f'{name}.mp4')) as video:
            assert sum(1 for _ in video.decode(video=0)) == T
        results.append(summary)
        write_json(out / 'summary.json', dict(status='running', cases=results))
        print(json.dumps(summary), flush=True)
        del values, position, grouping, masks, rgb
        torch.cuda.empty_cache()
    assert hashes == {str(p): sha(p) for p in weights}, 'Weights changed.'
    assert original_sources == {str(p.relative_to(REPO)): sha(p) for p in (REPO / 'isee').rglob('*.py')}
    config.update(status='complete', weights_unchanged=True, upstream_code_unchanged=True)
    write_json(out / 'config.json', config)
    write_json(out / 'summary.json', dict(status='complete', weights_unchanged=True, cases=results))
    with (out / 'README.md').open('a') as stream:
        stream.write('\nCompleted four full clips; MP4 frame counts verified. All five weight files and upstream model sources SHA256 unchanged. See summary.json for actual results.\n')


if __name__ == '__main__':
    main()
