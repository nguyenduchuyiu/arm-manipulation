"""Front-oblique RGB-D, causal TCOW masks and explicit geometry memory."""
import json
from pathlib import Path
import shutil
import subprocess

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES
from OATFlow.simulation.render import visual_model
from OATFlow.tracking.mask_video import DATA, dense_episode, digest, font, overlay
from OATFlow.tracking.point_data import top_camera, project
from OATFlow.tracking import pointcloud_front


SOURCE = Path('/workspace/robotics_assets/experiments/e114_mask/test_videos')
OUT = SOURCE.parent / 'test_pointcloud_3'
VIEW_AZIMUTH = -145
VIEW_ELEVATION = 18
CYAN = (0, 170, 181)
ORANGE = (230, 143, 34)
PINK = (213, 45, 141)


def erode(mask):
    return np.asarray(Image.fromarray(mask.astype('uint8') * 255).filter(ImageFilter.MinFilter(3))) > 0


def chroma(rgb):
    values = rgb.astype(float)
    return values / np.maximum(values.sum(-1, keepdims=True), 1)


def backproject(depth, camera):
    yy, xx = np.indices(depth.shape)
    rays = np.stack(((xx-159.5)/camera['focal'], (yy+40-159.5)/camera['focal'], np.ones_like(xx)), -1)
    return (rays * depth[..., None]) @ camera['rotation'].T + camera['position'] - camera['base']


def on_plane(mask, height, camera):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    ray = np.array([(xx.mean()-159.5)/camera['focal'], (yy.mean()+40-159.5)/camera['focal'], 1.])
    ray = camera['rotation'] @ ray
    origin = camera['position'] - camera['base']
    return origin + ray * ((height-origin[2])/ray[2])


def scatter_panel(scene, cloud, center, title, color, anchor):
    """Fixed orthographic 3D visualization, all coordinates in robot base metres."""
    image = Image.new('RGB', (740, 410), '#f4f7fb')
    draw = ImageDraw.Draw(image)
    az, el = np.deg2rad(VIEW_AZIMUTH), np.deg2rad(VIEW_ELEVATION)
    right = np.array([np.cos(az), -np.sin(az), 0])
    up = np.array([np.sin(az)*np.sin(el), np.cos(az)*np.sin(el), np.cos(el)])
    def uv(points):
        p = np.asarray(points)-anchor
        return np.stack((365+980*(p@right), 275-980*(p@up)), -1)
    # Fixed metric grid; never recenter on the GT target.
    for a in np.arange(-.3, .31, .05):
        for points in (np.array([[a, -.3, -.02], [a, .3, -.02]]),
                       np.array([[-.3, a, -.02], [.3, a, -.02]])):
            draw.line([tuple(x) for x in uv(points+anchor)], fill='#dde4ed', width=1)
    p = uv(scene)
    for x, y in p:
        if 4 < x < 736 and 45 < y < 385:
            draw.ellipse((x-.7, y-.7, x+.7, y+.7), fill='#a6b3c4')
    p = uv(cloud)
    for x, y in p:
        if 4 < x < 736 and 45 < y < 385:
            draw.ellipse((x-1.6, y-1.6, x+1.6, y+1.6), fill=color)
    if center is not None:
        x, y = uv(center)
        draw.ellipse((x-6, y-6, x+6, y+6), outline=color, width=2)
    draw.text((18, 12), title, font=font(18, True), fill=color)
    for axis, name, tint in (([.05, 0, 0], 'X', '#bd5555'), ([0, .05, 0], 'Y', '#399466'), ([0, 0, .05], 'Z', '#527eb7')):
        p = np.array([[70., 348.], [70+980*np.dot(axis, right), 348-980*np.dot(axis, up)]])
        draw.line([tuple(x) for x in p], fill=tint, width=3)
        draw.text(tuple(p[1]), name, font=font(13, True), fill=tint)
    if center is not None and len(cloud):
        draw.rounded_rectangle((550, 202, 726, 376), radius=8, fill='#ffffff', outline='#cbd5e1')
        draw.text((560, 211), 'LOCAL SURFACE x3', font=font(12, True), fill=color)
        local = cloud-center
        pixels = np.stack((638+2940*(local@right), 300-2940*(local@up)), -1)
        for x, y in pixels:
            if 554 < x < 722 and 235 < y < 372:
                draw.ellipse((x-1.3, y-1.3, x+1.3, y+1.3), fill=color)
    draw.text((18, 385), 'Robot-base coordinates | grid: 5 cm | gray: observed scene', font=font(14), fill='#56667b')
    return image


def main():
    OUT.mkdir(exist_ok=False)
    shutil.copyfile('docs/research/e114_front_README.md', OUT/'README.md')
    source = json.loads((SOURCE/'config.json').read_text())
    selected = [c for c in source['cases'] if c['name'] in ('unseen_permutation', 'both_unseen')]
    config = dict(status='running', source=str(SOURCE), checkpoint=source['checkpoint'],
                  epoch=source['epoch'], step=source['step'], cases=selected, fps=25,
                  visualization=dict(azimuth_deg=VIEW_AZIMUTH, elevation_deg=VIEW_ELEVATION,
                                     previous_output=str(SOURCE.parent/'test_pointcloud')),
                  source_sha256=digest(Path(__file__)), checkpoint_sha256=source['checkpoint_sha256'],
                  source_camera=dict(position_world_m=pointcloud_front.POSITION.tolist(),
                                     look_at_world_m=pointcloud_front.LOOK_AT.tolist(), fovy_deg=pointcloud_front.FOVY),
                  depth='new front-oblique ideal MuJoCo RGB-D at recorded qpos; no sensor noise',
                  inference='new causal TCOW predictions on front-oblique RGB, unchanged top-trained checkpoint; frame0 visible-mask prompt',
                  hidden_prior='rigid translation on initial surface-height plane from amodal-mask centroid',
                  appearance_gate='L2 chromaticity distance <0.16, eroded predicted mask; support >=25% initial pixels',
                  gt='used only to render sensor observations and score; never passed to estimator',
                  playback='all frames, cloud updated at cached prediction endpoints, causal hold')
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    shutil.copyfile(__file__, OUT/'source.py')
    shutil.copyfile(pointcloud_front.__file__, OUT/'front_source.py')
    model = visual_model(DATA/'scene.xml', object_bodies=NAMES)
    model.vis.global_.offwidth = model.vis.global_.offheight = 320
    state = mujoco.MjData(model)
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    camera = pointcloud_front.configure(model, state, top_camera(DATA)['base'])
    tracker = pointcloud_front.load_tracker(source['checkpoint'])
    summaries = []
    with mujoco.Renderer(model, height=320, width=320) as renderer:
        renderer.enable_depth_rendering()
        for case in selected:
            row, name = case['row'], case['name']
            dense = dense_episode(row)
            with np.load(SOURCE/f'{name}.npz') as z:
                endpoints = z['prediction_raw_index'].copy()
            qpos = np.concatenate([np.load(DATA/ref/'sim_state.npz')['qpos'] for ref in (row['context_path'], row['path'])])
            assert len(qpos) == row['frames']
            body = model.body(row['target_object_id']).id
            rgb_frames, depths, prompt, truths, hidden = pointcloud_front.render(model, state, qpos, endpoints, body, name)
            predictions = pointcloud_front.predict(tracker, rgb_frames, endpoints, prompt, name)
            packed = np.packbits(predictions, axis=-1, bitorder='little')
            held = np.searchsorted(endpoints, np.arange(row['frames']), side='right')-1
            dense['rgb'] = rgb_frames
            dense['hidden'] = hidden[held]
            ious = (predictions & truths).sum((1, 2))/np.maximum((predictions | truths).sum((1, 2)), 1)
            np.savez_compressed(OUT/f'{name}_masks.npz', prediction_packed=packed,
                                prediction_raw_index=endpoints, prompt=prompt,
                                gt_amodal_packed=np.packbits(truths, axis=-1, bitorder='little'),
                                hidden=hidden, iou=ious)
            writer = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', '1440x1000', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
                '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(OUT/f'{name}.mp4')], stdin=subprocess.PIPE)
            records, clouds, offsets, naive_clouds, naive_offsets = [], [], [0], [], [0]
            snapshots = set()
            for i, raw in enumerate(tqdm(endpoints, desc=f'{name} RGB-D replay', mininterval=5)):
                state.qpos[:] = qpos[raw]; mujoco.mj_forward(model, state)
                depth = depths[i]
                xyz = backproject(depth, camera)
                mask = np.unpackbits(packed[i], axis=-1, count=320, bitorder='little').astype(bool)
                rgb = dense['rgb'][raw]
                if i == 0:
                    initial = erode(prompt)
                    color_ref = np.median(chroma(rgb)[initial], axis=0)
                    initial &= np.linalg.norm(chroma(rgb)-color_ref, axis=-1) < .16
                    template = xyz[initial].copy()
                    assert len(template) > 20
                    template_center = np.median(template, axis=0)
                    height = float(template_center[2])
                    # Fixed view anchored by camera geometry, not target ground truth.
                    full = np.ones_like(mask)
                    anchor = on_plane(full, height, camera)
                    anchor[2] = height
                    support_threshold = max(12, int(.25*len(template)))
                    # Reference point is the initial observed-surface center, NOT object center.
                    reference_local = (template_center+camera['base']-state.xpos[body]) @ state.xmat[body].reshape(3, 3)
                    projected = project(template, camera)
                    yy, xx = np.nonzero(initial)
                    projection_error = float(np.max(np.abs(projected-np.stack((xx, yy), -1))))
                    assert projection_error < 1e-3
                valid = erode(mask) & (np.linalg.norm(chroma(rgb)-color_ref, axis=-1) < .16)
                measured = xyz[valid]
                if len(measured) >= support_threshold:
                    mode = 'MEASURED SURFACE'
                    estimate = np.median(measured, axis=0)
                    cloud = measured
                    tint = CYAN
                else:
                    estimate = on_plane(mask, height, camera)
                    if estimate is None:
                        estimate = template_center.copy() if not records else np.array(records[-1]['estimate'])
                        mode = 'HELD: NO MASK'
                    else:
                        mode = 'INFERRED: TABLE-PLANE PRIOR'
                    cloud = template-template_center+estimate
                    tint = ORANGE
                naive = xyz[mask]
                naive_center = np.median(naive, axis=0) if len(naive) else None
                truth = state.xpos[body]-camera['base'] + state.xmat[body].reshape(3, 3) @ reference_local
                error = float(np.linalg.norm(estimate-truth)*1000)
                record = dict(frame=int(raw), mode=mode, visible_support=len(measured), estimate=estimate.tolist(),
                              reference_gt=truth.tolist(), reference_error_mm=error, hidden_gt=bool(dense['hidden'][raw]),
                              mask_iou=float(ious[i]),
                              naive_reference_error_mm=float(np.linalg.norm(naive_center-truth)*1000) if naive_center is not None else None)
                records.append(record)
                clouds.append(cloud.astype('float32')); offsets.append(offsets[-1]+len(cloud))
                naive_clouds.append(naive.astype('float32')); naive_offsets.append(naive_offsets[-1]+len(naive))
                scene = xyz[::5, ::5].reshape(-1, 3)
                scene = scene[(np.abs(scene[:,:2]-anchor[:2]) < .38).all(-1) & (scene[:,2] < height+.3)]
                left = scatter_panel(scene, naive, naive_center, 'NAIVE: TCOW MASK x CURRENT DEPTH', PINK, anchor)
                right = scatter_panel(scene, cloud, estimate, mode, tint, anchor)
                next_raw = int(endpoints[i+1]) if i+1 < len(endpoints) else row['frames']
                for frame in range(int(raw), next_raw):
                    image = Image.new('RGB', (1440, 1000), '#ffffff'); draw = ImageDraw.Draw(image)
                    draw.text((24, 14), 'TCOW target tracking -> RGB-D surface cloud', font=font(28, True), fill='#192c43')
                    draw.text((24, 54), f'{case["label"]} | {row["target_object_id"]} | {row["path"]}', font=font(17), fill='#52647b')
                    image.paste(overlay(dense['rgb'][frame], mask, PINK).resize((640, 480)), (24, 111))
                    draw.text((24, 83), 'FRONT-OBLIQUE RGB + NEW TCOW MASK', font=font(18, True), fill=PINK)
                    image.paste(left, (690, 85)); image.paste(right, (690, 510))
                    draw.text((24, 618), f'{frame/25:.2f}s | {str(dense["phase"][frame])}', font=font(22, True), fill='#192c43')
                    draw.text((24, 660), mode, font=font(20, True), fill=tint)
                    draw.text((24, 697), f'Visible color-matched pixels: {len(measured)} / threshold {support_threshold}', font=font(17), fill='#52647b')
                    draw.text((24, 736), 'Cyan: measured visible target surface', font=font(18, True), fill=CYAN)
                    draw.text((24, 768), 'Orange: initial surface translated by mask + plane', font=font(17, True), fill=ORANGE)
                    draw.text((24, 813), 'Pink cloud may belong to cover / hand / table.', font=font(18), fill=PINK)
                    draw.text((24, 850), 'Orange is a hypothesis, not a depth measurement.', font=font(17), fill='#52647b')
                    draw.text((24, 885), f'Surface-reference error: {error:.1f} mm (GT: scoring only)', font=font(17), fill='#52647b')
                    draw.text((24, 941), 'New front RGB-D | top-trained TCOW, no fine-tuning | recorded expert actions | causal stride ~3', font=font(17), fill='#52647b')
                    draw.text((24, 970), 'No GT masks/poses used for reconstruction. Colored-object appearance gate; fixed-height prior fails under hidden lifting/rotation.', font=font(15), fill='#52647b')
                    writer.stdin.write(np.asarray(image).tobytes())
                    snapshot = 'initial' if frame == 0 else ('hidden' if dense['hidden'][frame] else ('reappeared' if frame >= row['decision_frames']['t_obj'] else None))
                    if snapshot and snapshot not in snapshots:
                        image.save(OUT/f'{name}_{snapshot}.png'); snapshots.add(snapshot)
            writer.stdin.close(); assert writer.wait() == 0
            with av.open(str(OUT/f'{name}.mp4')) as video:
                decoded = sum(1 for _ in video.decode(video=0))
            assert decoded == row['frames']
            np.savez_compressed(OUT/f'{name}.npz', frames=endpoints, clouds=np.concatenate(clouds),
                                offsets=offsets, naive_clouds=np.concatenate(naive_clouds), naive_offsets=naive_offsets,
                                template=template, camera_position=camera['position'], camera_rotation=camera['rotation'])
            (OUT/f'{name}.json').write_text(json.dumps(records, indent=2))
            metrics = {}
            for label, subset in [('all', records), ('hidden', [r for r in records if r['hidden_gt']]),
                                   ('visible_or_partial', [r for r in records if not r['hidden_gt']])]:
                metrics[label] = dict(frames=len(subset), median_reference_error_mm=float(np.median([r['reference_error_mm'] for r in subset])),
                    naive_median_reference_error_mm=float(np.median([r['naive_reference_error_mm'] for r in subset if r['naive_reference_error_mm'] is not None])))
            summary = dict(name=name, frames=decoded, duration_s=decoded/25, projection_roundtrip_max_pixel_error=projection_error,
                           initial_surface_points=len(template), metrics=metrics,
                           mask_iou_mean=float(ious.mean()), hidden_mask_iou_mean=float(ious[hidden].mean()) if hidden.any() else None)
            summaries.append(summary)
            (OUT/'summary.json').write_text(json.dumps(summaries, indent=2))
            print('VIDEO', json.dumps(summary), flush=True)
    config['status'] = 'complete'
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    with (OUT/'README.md').open('a') as stream:
        stream.write('\nCompleted: both videos decoded fully; projection roundtrip checked. See summary.json and per-frame JSON/NPZ.\n')


if __name__ == '__main__':
    main()
