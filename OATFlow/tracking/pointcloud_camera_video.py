"""Visualize decoded uint16 FFV1 depth and saved target cloud from the same camera."""
import json
from pathlib import Path
import shutil
import subprocess

import av
import mujoco
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES
from OATFlow.simulation.render import visual_model
from OATFlow.tracking.mask_video import DATA, digest, font, overlay
from OATFlow.tracking.point_data import project, top_camera
from OATFlow.tracking.pointcloud_front import configure
from OATFlow.tracking.pointcloud_video import backproject, CYAN, ORANGE, PINK


SOURCE = Path('/workspace/robotics_assets/experiments/e114_mask/test_pointcloud_3')
DEPTH_FILE = SOURCE.parent/'test_depth_compress/depth_uint16_0p1mm.mkv'
DEPTH_UNIT_M = .0001
OUT = DEPTH_FILE.parent/'view_uint16'
SCALE = 2.1
SIZE = (672, 504)
DEPTH_RANGE = (.5, 1.25)
DEPTH_LUT = (colormaps['turbo_r'](np.linspace(0, 1, 256))[:, :3]*255).astype('uint8')


def depth_image(depth):
    """Color every native depth pixel; preserve depth edges with nearest resizing."""
    assert np.isfinite(depth).all() and (depth >= 0).all()
    bins = np.rint(np.clip((depth-DEPTH_RANGE[0])/(DEPTH_RANGE[1]-DEPTH_RANGE[0]), 0, 1)*255).astype(int)
    rgb = DEPTH_LUT[bins].copy(); rgb[depth == 0] = [11, 20, 36]
    return Image.fromarray(rgb).resize(SIZE, Image.Resampling.NEAREST)


def target_overlay(image, cloud, camera, color):
    """Project only the saved target points; halo separates them from scene depth."""
    local = (cloud+camera['base']-camera['position']) @ camera['rotation']
    valid = local[:, 2] > 0
    if not valid.any():
        return image
    points, depth = cloud[valid], local[valid, 2]
    pixels = project(points, camera)*SCALE
    strength = .55+.45*(depth.max()-depth)/max(float(np.ptp(depth)), .001)
    tints = (np.asarray(color)[None]*strength[:, None]).astype('uint8')
    footprint = Image.new('L', SIZE)
    mask_draw = ImageDraw.Draw(footprint)
    for x, y in pixels:
        mask_draw.ellipse((x-1.4, y-1.4, x+1.4, y+1.4), fill=255)
    outer = (np.asarray(footprint.filter(ImageFilter.MaxFilter(3))) > 0) & (np.asarray(footprint) == 0)
    rgb = np.asarray(image).copy(); rgb[outer] = 255
    result = Image.fromarray(rgb); draw = ImageDraw.Draw(result)
    for i in np.argsort(depth)[::-1]:
        x, y = pixels[i]
        draw.ellipse((x-1.4, y-1.4, x+1.4, y+1.4), fill=tuple(tints[i]))
    return result


def main(*, depth_file=DEPTH_FILE, depth_unit_m=DEPTH_UNIT_M, out=OUT,
         readme=Path('docs/research/e114_uint16_view_README.md')):
    out.mkdir(exist_ok=False)
    shutil.copyfile(readme, out/'README.md')
    shutil.copyfile(__file__, out/'source.py')
    config = json.loads((SOURCE/'config.json').read_text())
    assert config['status'] == 'complete'
    compression = json.loads((depth_file.parent/'config.json').read_text())
    assert compression['status'] == 'complete'
    cases = [c for c in config['cases'] if c['row']['path'] == compression['case']['row']['path']]
    assert len(cases) == 1
    depth_hash = digest(depth_file)
    decoded = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '4',
        '-i', str(depth_file), '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'], capture_output=True, check=True).stdout
    codes = np.frombuffer(decoded, dtype='<u2').reshape(compression['frames'], 240, 320)
    depth_frames = codes.astype('float32')*depth_unit_m
    del codes, decoded
    output_config = dict(status='running', source=str(SOURCE), checkpoint=config['checkpoint'],
        epoch=config['epoch'], step=config['step'], source_camera=config['source_camera'],
        panels=['RGB + saved TCOW mask', 'decoded uint16 FFV1 depth + saved target point cloud'],
        depth_file=str(depth_file), depth_sha256=depth_hash, depth_unit_m=depth_unit_m,
        invalid_depth='uint16 code0 displayed dark; not a measured zero-distance surface',
        projection='same calibrated perspective camera, crop y40:280, scale2.1 in both panels',
        scene_colors=dict(quantity='axial camera depth in metres', colormap='turbo_r',
                          fixed_range_m=list(DEPTH_RANGE), clipped_outside_range=True, sampling_stride=1,
                          display_resize='nearest; full dense background'),
        target_overlay='saved cyan measured / orange inferred points; white silhouette halo; inferred surface shown through cover',
        source_sha256=digest(Path(__file__)), fps=25, resolution=[1440, 860],
        inference='none; reuse saved masks/clouds exactly', cases=cases)
    (out/'config.json').write_text(json.dumps(output_config, indent=2))
    model = visual_model(DATA/'scene.xml', object_bodies=NAMES)
    model.vis.global_.offwidth = model.vis.global_.offheight = 320
    state = mujoco.MjData(model)
    camera = configure(model, state, top_camera(DATA)['base'])
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    checks = []
    with mujoco.Renderer(model, height=320, width=320) as renderer:
        for case in cases:
            name, row = case['name'], case['row']
            sources = [SOURCE/f'{name}{suffix}' for suffix in ('.npz', '_masks.npz', '.json')]
            hashes = [digest(p) for p in sources]
            records = json.loads(sources[2].read_text())
            with np.load(sources[0]) as z:
                times, clouds, offsets = z['frames'], z['clouds'], z['offsets']
                np.testing.assert_allclose(camera['position'], z['camera_position'], atol=1e-12)
                np.testing.assert_allclose(camera['rotation'], z['camera_rotation'], atol=1e-12)
            with np.load(sources[1]) as z:
                masks = np.unpackbits(z['prediction_packed'], axis=-1, count=320, bitorder='little').astype(bool)
                hidden = z['hidden']
                np.testing.assert_array_equal(times, z['prediction_raw_index'])
            held = np.searchsorted(times, np.arange(row['frames']), side='right')-1
            qpos, phases = [], []
            for ref in (row['context_path'], row['path']):
                with np.load(DATA/ref/'sim_state.npz') as z: qpos.append(z['qpos'])
                with np.load(DATA/ref/'supervision.npz') as z: phases.append(z['phase'])
            qpos, phases = np.concatenate(qpos), np.concatenate(phases)
            writer = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', '1440x860', '-r', '25', '-i', '-', '-an',
                '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '19',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/f'{name}.mp4')], stdin=subprocess.PIPE)
            snapshots = set()
            projection_error = 0.
            for frame in tqdm(range(row['frames']), desc=f'{name}: matched-camera video', mininterval=5):
                k = int(held[frame])
                record = records[k]
                state.qpos[:] = qpos[frame]; mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera='cam_left', scene_option=option)
                rgb = renderer.render()[40:280].copy()
                depth = depth_frames[frame]
                if frame == 0:
                    xyz = backproject(depth, camera)
                    yy, xx = np.indices((240, 320))
                    valid = depth[::3, ::3] > 0
                    projection_error = float(np.max(np.abs(project(xyz[::3, ::3][valid], camera)-
                                                           np.stack((xx, yy), -1)[::3, ::3][valid])))
                    assert projection_error < 1e-3
                left = overlay(rgb, masks[k], PINK).resize(SIZE, Image.Resampling.BICUBIC)
                color = CYAN if record['mode'] == 'MEASURED SURFACE' else ORANGE
                cloud = clouds[offsets[k]:offsets[k+1]]
                right = target_overlay(depth_image(depth), cloud, camera, color)
                image = Image.new('RGB', (1440, 860), '#ffffff'); draw = ImageDraw.Draw(image)
                draw.text((24, 18), 'Uint16 depth | decoded FFV1 + target point cloud', font=font(27, True), fill='#192c43')
                draw.text((24, 61), f'{case["label"]} | {row["target_object_id"]} | {row["path"]}', font=font(18), fill='#52647b')
                draw.text((24, 103), 'RGB + TCOW MASK', font=font(19, True), fill=PINK)
                draw.text((744, 103), f'UINT16: {depth_unit_m*1000:g} mm/UNIT + TARGET CLOUD', font=font(19, True), fill='#192c43')
                image.paste(left, (24, 135)); image.paste(right, (744, 135))
                draw.text((24, 661), f'{frame/25:.2f}s | {str(phases[frame])}', font=font(22, True), fill='#192c43')
                draw.text((744, 661), record['mode'], font=font(19, True), fill=color)
                draw.text((24, 707), 'Pink: saved TCOW target mask', font=font(18, True), fill=PINK)
                draw.text((744, 707), 'Cyan: measured', font=font(17, True), fill=CYAN)
                draw.text((960, 707), 'Orange: inferred through cover', font=font(17, True), fill=ORANGE)
                draw.text((24, 750), 'Scene colors: camera depth (m), fixed scale', font=font(18), fill='#52647b')
                gradient = DEPTH_LUT[np.rint(np.linspace(0, 255, 500)).astype(int)][None].repeat(16, axis=0)
                image.paste(Image.fromarray(gradient), (850, 744))
                draw.text((744, 742), 'NEAR', font=font(15, True), fill='#52647b')
                draw.text((1364, 742), 'FAR', font=font(15, True), fill='#52647b')
                for value in (.5, .75, 1., 1.25):
                    x = 850+499*(value-DEPTH_RANGE[0])/(DEPTH_RANGE[1]-DEPTH_RANGE[0])
                    draw.text((x, 766), f'{value:g}', anchor='mt', font=font(14), fill='#52647b')
                draw.text((24, 802), f'Decoded FFV1: {depth_file.stat().st_size/1e6:.2f} MB | {depth_unit_m*1000:g} mm/unit | dark background: invalid | fixed color scale', font=font(16), fill='#52647b')
                draw.text((24, 833), 'Target cloud reused from original replay; orange is an inferred surface, not measured depth through cover.', font=font(16), fill='#52647b')
                writer.stdin.write(np.asarray(image).tobytes())
                snapshot = 'initial' if frame == 0 else ('hidden' if hidden[k] else
                            ('reappeared' if frame >= row['decision_frames']['t_obj'] else None))
                if snapshot and snapshot not in snapshots:
                    image.save(out/f'{name}_{snapshot}.png'); snapshots.add(snapshot)
            writer.stdin.close(); assert writer.wait() == 0
            with av.open(str(out/f'{name}.mp4')) as video:
                decoded = sum(1 for _ in video.decode(video=0))
            assert decoded == row['frames']
            assert hashes == [digest(p) for p in sources]
            check = dict(name=name, frames=decoded, duration_s=decoded/25,
                         projection_roundtrip_max_pixel_error=projection_error,
                         source_masks_clouds_metrics_unchanged=True)
            checks.append(check)
            (out/'summary.json').write_text(json.dumps(checks, indent=2))
            print('VIDEO', json.dumps(check), flush=True)
    output_config['status'] = 'complete'
    assert digest(depth_file) == depth_hash
    (out/'config.json').write_text(json.dumps(output_config, indent=2))


if __name__ == '__main__':
    main()
