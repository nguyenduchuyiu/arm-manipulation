"""Front RGB-D, upright fisheye wrist RGB/masks and a physical-motion diagnostic."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import av
from matplotlib import colormaps
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES, write_json
from OATFlow.simulation.render import visual_model
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.tracking.mask_video import font


COLORS = [(230, 54, 44), (42, 120, 240), (36, 177, 98), (224, 185, 25)]
LUT = (colormaps['turbo_r'](np.linspace(0, 1, 256))[:, :3]*255).astype('uint8')


def decode(paths, depth=False):
    result = []
    for path in paths:
        with av.open(str(path)) as video:
            video.streams.video[0].thread_count = 1
            result.extend(f.to_ndarray(format='gray16le' if depth else 'rgb24') for f in video.decode(video=0))
    return result


def relative_panel(history, assignment):
    image = Image.new('RGB', (480, 360), '#f3f6fa'); draw = ImageDraw.Draw(image)
    draw.text((16, 15), 'GT positions in each cover frame', font=font(18, True), fill='#23364c')
    draw.text((16, 44), 'Colored trails: actual cube motion, not prediction', font=font(14), fill='#52647b')
    if not len(history):
        draw.text((40, 170), 'Waiting for shuffle to begin', font=font(20), fill='#52647b')
        return image
    for ci, cover in enumerate(NAMES[4:]):
        origin = np.array([125+230*ci, 205])
        draw.rectangle((origin[0]-89, origin[1]-71, origin[0]+89, origin[1]+71), outline='#8799ae', width=2)
        draw.text((origin[0]-36, 100), cover, font=font(17, True), fill='#23364c')
        for t, name in enumerate(NAMES[:4]):
            if assignment[name] != cover: continue
            points = origin+history[:, t, :2]*np.array([1200, -1200])
            if len(points) > 1: draw.line([tuple(p) for p in points], fill=COLORS[t], width=2)
            x, y = points[-1]; draw.ellipse((x-6, y-6, x+6, y+6), fill=COLORS[t])
    draw.text((16, 319), 'Cover outlines fixed; cubes are free physical bodies.', font=font(14), fill='#52647b')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode', help='Render only this manifest episode path.')
    args = parser.parse_args(); root, out = args.data, args.output
    assert json.loads((root/'config.json').read_text())['status'] == 'complete'
    out.mkdir(exist_ok=False)
    shutil.copyfile('docs/research/e116_README.md', out/'README.md')
    rows = [json.loads(l) for l in (root/'manifest.jsonl').read_text().splitlines()]
    selected = ([next(r for r in rows if r['path'] == args.episode)] if args.episode else
                [next(r for r in rows if r['split'] == split and r['target_id'] == t)
                 for split, t in [('train', 0), ('test', 3)]])
    model = visual_model(root/'scene.xml', object_bodies=NAMES)
    clip_planes(model)
    model.vis.global_.offwidth = 320; model.vis.global_.offheight = 240
    # Diagnostic cutaway ONLY; the dataset RGB-D remains opaque.
    for name in NAMES[4:]: model.geom_rgba[model.geom_bodyid == model.body(name).id, 3] = .15
    state = mujoco.MjData(model); option = mujoco.MjvOption(); option.geomgroup[3] = 0
    results = []
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        for row in selected:
            name = row['evaluation_group']; context = root/row['context_path']; expert = root/row['path']
            rgb = {v:decode([p/f'rgb_{v}.mp4' for p in (context, expert)]) for v in ('front', 'wrist')}
            depth = decode([p/'depth_front.mkv' for p in (context, expert)], True)
            wrist_masks = np.concatenate([np.load(p/'masks.npz')['instance'][:, 1] for p in (context, expert)])
            states = [np.load(p/'sim_state.npz') for p in (context, expert)]
            qpos = np.concatenate([s['qpos'] for s in states])
            phases = np.concatenate([np.load(p/'supervision.npz')['phase'] for p in (context, expert)])
            shuffle_start = int(np.flatnonzero(phases == 'shuffle')[0])
            labels = [np.load(p/'labels.npz') for p in (context, expert)]
            xyz = np.concatenate([z['entity_xyz_world_m'] for z in labels]); rotations = np.concatenate([z['entity_rotation_world'] for z in labels])
            assignment = json.loads((context/'context.json').read_text())['assignment']
            relative = np.stack([np.einsum('nji,nj->ni', rotations[:, NAMES.index(assignment[n])],
                                         xyz[:, t]-xyz[:, NAMES.index(assignment[n])]) for t,n in enumerate(NAMES[:4])], axis=1)
            n = row['frames']; assert len(qpos) == n and all(len(v) == n for v in [*rgb.values(), depth, wrist_masks])
            writer = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', '1440x840', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264',
                '-threads', '2', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/f'{name}.mp4')], stdin=subprocess.PIPE)
            for frame in tqdm(range(n), desc=f'Preview dynamic {name}', mininterval=5):
                image = Image.new('RGB', (1440, 840), 'white'); draw = ImageDraw.Draw(image)
                target_title = row.get('target_display_name', row['target_object_id'])
                draw.text((16, 8), f'Dynamic shuffle | {row["path"]} | {target_title} | {frame/25:.2f}s | {phases[frame]}', font=font(22, True), fill='#23364c')
                for vi, view in enumerate(('front', 'wrist')):
                    pixels = rgb[view][frame]
                    if view == 'wrist': pixels = np.rot90(pixels, 2)
                    image.paste(Image.fromarray(pixels).resize((480, 360)), (vi*480, 66))
                    title = 'FRONT RGB' if view == 'front' else 'WRIST RGB | fisheye | rotated180deg'
                    draw.text((16+480*vi, 42), title, font=font(17, True), fill='#23364c')
                z = depth[frame].astype('float32')*.002
                colors = LUT[np.rint(np.clip((z-.5)/.75, 0, 1)*255).astype(int)].copy()
                colors[z == 0] = [11, 20, 36]
                image.paste(Image.fromarray(colors).resize((480, 360), Image.Resampling.NEAREST), (0, 452))
                draw.text((16, 430), 'FRONT DEPTH | 2mm/unit | 0.5-1.25m', font=font(17, True), fill='#23364c')
                label_colors = np.array([[11, 20, 36], *COLORS, [150, 150, 150], [185, 185, 185], [100, 120, 145], [105, 70, 40]], dtype='uint8')
                wrist_gt = np.rot90(label_colors[wrist_masks[frame]], 2)
                image.paste(Image.fromarray(wrist_gt).resize((480, 360), Image.Resampling.NEAREST), (480, 452))
                draw.text((496, 430), 'WRIST VISIBLE MASKS | GT | rotated180deg', font=font(16, True), fill='#23364c')
                state.qpos[:] = qpos[frame]; mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera='cam_front', scene_option=option)
                image.paste(Image.fromarray(renderer.render()).resize((480, 360)), (960, 66))
                draw.text((976, 42), 'CUTAWAY | GT physics diagnostic only', font=font(17, True), fill='#b66315')
                upto = min(frame+1, row['context_frames'])
                image.paste(relative_panel(relative[shuffle_start:upto], assignment), (960, 452))
                draw.text((16, 817), 'Front RGB-D + wrist RGB only. Wrist rotated180deg for display. Masks/cutaway are GT diagnostics; no learned tracker or policy.', font=font(15), fill='#52647b')
                writer.stdin.write(np.asarray(image).tobytes())
                if frame in (0, row['context_frames']-1, row['decision_frames']['t_obj']): image.save(out/f'{name}_{frame:04d}.png')
            writer.stdin.close(); assert writer.wait() == 0
            with av.open(str(out/f'{name}.mp4')) as video: decoded = sum(1 for _ in video.decode(video=0))
            assert decoded == n
            results.append(dict(path=row['path'], video=f'{name}.mp4', frames=n))
            write_json(out/'summary.json', results)
    write_json(out/'config.json', dict(status='complete', source=str(root), diagnostic_only=True,
        depth_views=['front'], wrist_display_rotation_deg=180, cases=selected))


if __name__ == '__main__': main()
