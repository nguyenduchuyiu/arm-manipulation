"""Render front RGB-D and fisheye wrist RGB, masks and per-frame camera calibration."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import time

import av
import mujoco
import numpy as np
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES, BASE, write_json
from OATFlow.simulation.render import visual_model
from OATFlow.simulation.io import encoder
from OATFlow.tracking.mask_video import digest
from OATFlow.data_tools.wrist_camera import Fisheye, INTRINSICS, REFERENCE, DIAGONAL_FOV


VIEWS = ('front', 'wrist')
DEPTH_VIEWS = ('front',)
SCHEMA = 'front_rgbd_wrist_fisheye_rgb_v1'
CAMERAS = ('cam_front', 'wrist')
UNIT = .002
MAX_DEPTH = 6.5535
CLIP_NEAR = .001
CLIP_FAR = 10.


def clip_planes(model):
    # MuJoCo scales these by scene extent; parked covers must not set the camera's near plane.
    model.vis.map.znear = CLIP_NEAR/model.stat.extent
    model.vis.map.zfar = CLIP_FAR/model.stat.extent


def initialize(root):
    global ROOT, MODEL, STATE, RGB, SEG, OPTION, ONLY, GEOMS, LUT, WRIST
    ROOT = Path(root); MODEL = visual_model(ROOT/'scene.xml', object_bodies=NAMES)
    clip_planes(MODEL)
    MODEL.vis.global_.offwidth = 320; MODEL.vis.global_.offheight = 256
    STATE = mujoco.MjData(MODEL)
    RGB = mujoco.Renderer(MODEL, height=240, width=320)
    samples = MODEL.vis.quality.offsamples; MODEL.vis.quality.offsamples = 0
    SEG = mujoco.Renderer(MODEL, height=240, width=320)
    MODEL.vis.quality.offsamples = samples; SEG.enable_segmentation_rendering()
    WRIST = Fisheye(MODEL)
    OPTION = mujoco.MjvOption(); OPTION.geomgroup[3] = 0
    ONLY = mujoco.MjvOption(); ONLY.geomgroup[:] = 0; ONLY.geomgroup[5] = 1
    LUT = np.zeros(MODEL.ngeom, np.uint8); GEOMS = {}
    for index, name in enumerate(NAMES, 1):
        body = MODEL.body(name).id
        GEOMS[index] = np.flatnonzero((MODEL.geom_bodyid == body) & (MODEL.geom_rgba[:, 3] > 0))
        LUT[MODEL.geom_bodyid == body] = index
    for g in range(MODEL.ngeom):
        if MODEL.geom_bodyid[g] != 0 and MODEL.body(int(MODEL.geom_bodyid[g])).name not in NAMES:
            LUT[g] = 7
    LUT[MODEL.geom('table_top').id] = 8


def owners():
    result = np.zeros(4, np.uint8)
    for i, name in enumerate(NAMES[:4]):
        for j, cover in enumerate(NAMES[4:], 5):
            body = MODEL.body(cover).id
            p = STATE.xmat[body].reshape(3, 3).T @ (STATE.xpos[MODEL.body(name).id]-STATE.xpos[body])
            if abs(p[0]) < .074 and abs(p[1]) < .059 and -.005 < p[2] < .055:
                assert result[i] == 0
                result[i] = j
    return result


def depth_encoder(path):
    return subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'gray16le', '-s', '320x240', '-r', '25', '-i', '-', '-an', '-c:v', 'ffv1',
        '-level', '3', '-coder', '1', '-context', '1', '-g', '1', '-slices', '4', '-slicecrc', '1',
        '-threads', '1', str(path)], stdin=subprocess.PIPE)


def render(job):
    ref, target = job; directory = ROOT/ref
    if (directory/'render_complete.json').exists():
        result = json.loads((directory/'render_complete.json').read_text())
        assert result['camera_schema'] == SCHEMA, f'Stale render: {directory}'
        return result
    if shutil.disk_usage(ROOT).free < 3*2**30:
        raise RuntimeError('Less than3GiB free: stop RGB-D rendering; free space before resuming.')
    context = target is None
    meta = json.loads((directory/('context.json' if context else 'episode.json')).read_text())
    with np.load(directory/'sim_state.npz') as z: states = {k:z[k] for k in z.files}
    phases = np.load(directory/'supervision.npz')['phase']; n = len(phases)
    ids = list(range(1, 7)) if context else [target+1, 5, 6]
    targets = list(range(4)) if context else [target]
    instance = np.zeros((n, 2, 240, 320), np.uint8)
    amodal = np.zeros((n, 2, len(ids), 240, 40), np.uint8)
    owner = np.zeros((n, len(targets)), np.uint8)
    fraction = np.ones((n, 2, len(targets)), np.float32)
    fov = np.zeros((n, 2, len(targets)), bool)
    front = np.zeros((n, 2, len(targets)), np.uint8)
    semantic = np.zeros((n, len(targets)), np.uint8)
    tcow = None if context else np.zeros((n, 2, 3, 240, 40), np.uint8)
    relevance = None if context else np.zeros((n, 2, 240, 320), np.uint8)
    transforms = np.zeros((n, 2, 4, 4)); transforms[..., 3, 3] = 1
    intrinsics = np.zeros((2, 3, 3)); intrinsics[:, 2, 2] = 1
    for vi, camera in enumerate(CAMERAS):
        cid = MODEL.camera(camera).id; focal = 120/np.tan(np.deg2rad(MODEL.cam_fovy[cid])/2)
        intrinsics[vi] = [[focal, 0, 159.5], [0, focal, 119.5], [0, 0, 1]]
    intrinsics[1] = INTRINSICS
    rgb_writers = {v:encoder(directory/f'rgb_{v}.partial.mp4', 320, 240) for v in VIEWS}
    depth_writers = {v:depth_encoder(directory/f'depth_{v}.partial.mkv') for v in DEPTH_VIEWS}
    hashes = {v:hashlib.sha256() for v in DEPTH_VIEWS}
    errors = {v:dict(count=0, invalid=0, abs_sum_m=0., max_m=0.) for v in DEPTH_VIEWS}
    start = time.perf_counter()
    for frame in tqdm(range(n), desc=f'RGB-D {ref}', mininterval=10, leave=False):
        for key in ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat', 'eq_active'):
            getattr(STATE, key)[:] = states[key][frame]
        mujoco.mj_forward(MODEL, STATE)
        owner[frame] = owners()[targets]
        for vi, (view, camera) in enumerate(zip(VIEWS, CAMERAS)):
            cid = MODEL.camera(camera).id
            transforms[frame, vi, :3, :3] = STATE.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
            transforms[frame, vi, :3, 3] = STATE.cam_xpos[cid]
            if view == 'wrist':
                rgb = WRIST.render(STATE, OPTION)
            else:
                RGB.update_scene(STATE, camera=camera, scene_option=OPTION); rgb = RGB.render()
            rgb_writers[view].stdin.write(rgb.tobytes())
            if view in DEPTH_VIEWS:
                RGB.enable_depth_rendering(); depth = RGB.render(); RGB.disable_depth_rendering()
                valid = np.isfinite(depth) & (depth > 0) & (depth <= MAX_DEPTH)
                codes = np.zeros(depth.shape, '<u2'); codes[valid] = np.rint(depth[valid]/UNIT).astype('uint16')
                delta = np.abs(codes[valid].astype('float64')*UNIT-depth[valid])
                e = errors[view]; e['count'] += int(valid.sum()); e['invalid'] += int((~valid).sum())
                e['abs_sum_m'] += float(delta.sum()); e['max_m'] = max(e['max_m'], float(delta.max()))
                assert e['max_m'] <= .0011
                payload = codes.tobytes(); hashes[view].update(payload); depth_writers[view].stdin.write(payload)
            if view == 'wrist': seg = WRIST.render(STATE, OPTION, segmentation=True)
            else:
                SEG.update_scene(STATE, camera=camera, scene_option=OPTION); seg = SEG.render()
            visible = np.zeros((240, 320), np.uint8)
            valid_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            visible[valid_geom] = LUT[seg[..., 0][valid_geom]]; instance[frame, vi] = visible
            masks = {}
            for index in ids:
                geoms = GEOMS[index]; old = MODEL.geom_group[geoms].copy(); MODEL.geom_group[geoms] = 5
                if view == 'wrist': mask_render = WRIST.render(STATE, ONLY, segmentation=True)
                else:
                    SEG.update_scene(STATE, camera=camera, scene_option=ONLY); mask_render = SEG.render()
                masks[index] = mask_render[..., 0] >= 0; MODEL.geom_group[geoms] = old
                amodal[frame, vi, ids.index(index)] = np.packbits(masks[index], axis=-1, bitorder='little')
            for ti, t in enumerate(targets):
                mask = masks[t+1]; area = int(mask.sum()); fov[frame, vi, ti] = area > 0
                assert not np.any((visible == t+1) & ~mask)
                if area:
                    fraction[frame, vi, ti] = 1-np.count_nonzero(visible == t+1)/area
                    amounts = [np.count_nonzero(mask & (visible == c))/area for c in (5, 6)]
                    dominant = int(np.argmax(amounts))
                    if fraction[frame, vi, ti] >= .95 and amounts[dominant] >= .475: front[frame, vi, ti] = 5+dominant
                if not context:
                    tcow[frame, vi, 0] = amodal[frame, vi, 0]
                    for channel, index in ((1, front[frame, vi, ti]), (2, owner[frame, ti])):
                        if index: tcow[frame, vi, channel] = amodal[frame, vi, ids.index(int(index))]
        if context:
            semantic[frame] = 2 if phases[frame] == 'reveal' else 1
            if frame == 0: assert all(np.count_nonzero(instance[0, 0] == t+1) > 20 for t in targets)
            if phases[frame] in ('shuffle', 'shuffle_pause', 'hold'): assert (owner[frame] > 0).all()
        else:
            semantic[frame, 0] = 0 if phases[frame] == 'done' else (2 if str(phases[frame]).startswith(NAMES[target]+'_') else 1)
            label = target+1 if semantic[frame, 0] == 2 else NAMES.index(meta['selected_cover'])+1
            relevance[frame][instance[frame] == label] = semantic[frame, 0]
    checks = {}
    for view in VIEWS:
        writer = rgb_writers[view]; writer.stdin.close(); assert writer.wait() == 0
        with av.open(str(directory/f'rgb_{view}.partial.mp4')) as video:
            video.streams.video[0].thread_count = 1; assert sum(1 for _ in video.decode(video=0)) == n
        (directory/f'rgb_{view}.partial.mp4').replace(directory/f'rgb_{view}.mp4')
    for view in DEPTH_VIEWS:
        writer = depth_writers[view]; writer.stdin.close(); assert writer.wait() == 0
        decoded_hash = hashlib.sha256(); frames = 0
        with av.open(str(directory/f'depth_{view}.partial.mkv')) as video:
            video.streams.video[0].thread_count = 1
            for image in video.decode(video=0):
                decoded_hash.update(image.to_ndarray(format='gray16le').astype('<u2', copy=False).tobytes()); frames += 1
        assert frames == n and decoded_hash.hexdigest() == hashes[view].hexdigest()
        (directory/f'depth_{view}.partial.mkv').replace(directory/f'depth_{view}.mkv')
        checks[view] = dict(frames=n, codec_lossless=True, sha256_decoded=decoded_hash.hexdigest(),
            quantization_mae_mm=errors[view]['abs_sum_m']/errors[view]['count']*1000,
            quantization_max_mm=errors[view]['max_m']*1000, invalid_pixels=errors[view]['invalid'])
    camera_base = transforms.copy(); camera_base[..., :3, 3] -= BASE
    np.savez_compressed(directory/'calibration.npz', intrinsics=intrinsics,
        camera_to_world=transforms, camera_to_base=camera_base, views=np.array(VIEWS),
        depth_unit_m=UNIT, invalid_depth_code=0, depth_max_m=MAX_DEPTH,
        projection_model=np.array(['pinhole', 'fisheye_equidistant']), distortion_coefficients=np.zeros((2, 4)),
        wrist_reference=np.array(REFERENCE), wrist_diagonal_fov_deg=DIAGONAL_FOV,
        depth_views=np.array(DEPTH_VIEWS), visualization_rotation_deg=np.array([0, 180]),
        depth_model=np.array('ideal_geometric_optical_z; sensor range/focus not simulated'),
        render_clip_near_m=CLIP_NEAR, render_clip_far_m=CLIP_FAR,
        resolution=np.array([320, 240]), timestamp_s=states['timestamp_s'])
    masks = dict(instance=instance, frontmost_cover_ids=front, container_ids=owner,
        occlusion_fraction=fraction, target_in_fov=fov, semantic_entity=semantic,
        target_ids=np.array(targets), timestamp_s=states['timestamp_s'], views=np.array(VIEWS),
        resolution=np.array([320, 240]), bitorder=np.array('little'))
    if context: masks.update(amodal_packed=amodal, amodal_entity_ids=np.array(ids))
    else: masks.update(tcow_packed=tcow, relevance=relevance)
    np.savez_compressed(directory/'masks.npz', **{k:np.asfortranarray(v) if v.ndim >= 3 else v for k,v in masks.items()})
    if not context:
        write_json(directory/'input.json', dict(context_path='..', expert_path='.', views=list(VIEWS),
            prompt='prompt.npz', prompt_source='initial visible target mask from context frame0 front; no later GT in inputs',
            rgb_segments={v:[f'../rgb_{v}.mp4', f'rgb_{v}.mp4'] for v in VIEWS},
            depth_segments={v:[f'../depth_{v}.mkv', f'depth_{v}.mkv'] for v in DEPTH_VIEWS},
            calibration=['../calibration.npz', 'calibration.npz'], observations=['../observation.npz', 'observation.npz']))
    result = dict(path=ref, frames=n, kind='context' if context else 'expert', camera_schema=SCHEMA,
        elapsed_s=time.perf_counter()-start, bytes=sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
        depth_checks=checks, wrist_fov_fraction=float(fov[:, 1].mean()))
    write_json(directory/'render_complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(); root = args.data
    hold = root/'RENDER_HOLD.json'
    if hold.exists():
        print(f'Rendering is held: {hold.read_text().strip()}', flush=True)
        for _ in tqdm(iter(lambda: hold.exists(), False), desc='Waiting for camera correction', mininterval=30):
            time.sleep(5)
    config = json.loads((root/'config.json').read_text())
    assert config['status'] in ('physics_complete', 'rendering', 'render_complete')
    config.update(status='rendering', render_source_sha256=digest(Path(__file__)), depth_views=list(DEPTH_VIEWS),
        camera_schema=SCHEMA, projection_model=['pinhole', 'fisheye_equidistant'], wrist_reference=REFERENCE,
        wrist_diagonal_fov_deg=DIAGONAL_FOV, visualization_rotation_deg=[0, 180],
        wrist_camera_source_sha256=digest(Path(__file__).with_name('wrist_camera.py')))
    write_json(root/'config.json', config)
    rows = [json.loads(l) for l in (root/'manifest.jsonl').read_text().splitlines()]
    jobs = [(r, None) for r in sorted({r['context_path'] for r in rows})]+[(r['path'], r['target_id']) for r in rows]
    results = []; start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(str(root),)) as pool:
        futures = [pool.submit(render, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(jobs), desc='Front RGB-D + wrist fisheye RGB', mininterval=5):
            result = future.result(); results.append(result)
            write_json(root/'render_progress.json', dict(segments=len(results), total=len(jobs),
                frames=sum(r['frames'] for r in results), elapsed_s=time.perf_counter()-start,
                estimated_remaining_s=(time.perf_counter()-start)/len(results)*(len(jobs)-len(results))))
    config.update(status='render_complete', render_elapsed_s=time.perf_counter()-start)
    write_json(root/'config.json', config)
    write_json(root/'render_summary.json', dict(segments=results, elapsed_s=time.perf_counter()-start,
        total_bytes=sum(r['bytes'] for r in results), unique_frames=sum(r['frames'] for r in results)))


if __name__ == '__main__': main()
