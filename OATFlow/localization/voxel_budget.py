"""Count two-view voxel samples and measure real render/ViT/decoder costs."""
import argparse
from itertools import product
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES
from OATFlow.localization.model import WEIGHTS, state_hash
from OATFlow.localization.two_view import TwoViewXYZ
from OATFlow.simulation.io import write_json
from OATFlow.simulation.render import visual_model

DATA = Path('/workspace/robotics_assets/datasets/shuffle_pick_multiview')
GRID = Path('/workspace/robotics_assets/datasets/reach_voxel_grid/grid.npz')
VIEWS = ('top', 'left')


def setup():
    model = visual_model(DATA / 'scene.xml', object_bodies=NAMES)
    model.vis.global_.offwidth = model.vis.global_.offheight = 320
    state = mujoco.MjData(model)
    with np.load(DATA / 'g000/p00/sim_state.npz') as z:
        state.qpos[:] = z['qpos'][0]
    addresses = [int(model.joint(n+'_joint').qposadr[0]) for n in NAMES]
    for i, address in enumerate(addresses):
        state.qpos[address:address+3] = [-3-i, 0, .1]
        state.qpos[address+3:address+7] = [1, 0, 0, 0]
    state.qvel[:] = 0
    mujoco.mj_forward(model, state)
    return model, state, addresses


def cameras(model, state):
    result = []
    for name in VIEWS:
        index = model.camera('cam_'+name).id
        focal = float(160/np.tan(np.deg2rad(model.cam_fovy[index])/2))
        result.append(dict(name=name, size=[320, 320], focal=focal,
            K=[[focal, 0, 159.5], [0, focal, 159.5], [0, 0, 1]],
            world_position=state.cam_xpos[index].copy().tolist(),
            world_from_cv_rotation=(state.cam_xmat[index].reshape(3, 3) @ np.diag([1., -1., -1.])).tolist()))
    return result


def count_voxels(model, state, addresses, cams):
    with np.load(GRID) as z:
        accepted = np.flatnonzero(z['status'] == 2)
        xyz = z['centers_world_m'][accepted].copy()
        base = z['base_origin_world_m'].copy()
        total = len(z['status'])
    corners = xyz[:, None] + np.array(list(product((-.016, .016), repeat=3)))[None]
    frustum = np.ones(len(xyz), bool)
    per_camera = {}
    for cam in cams:
        points = (corners - cam['world_position']) @ np.array(cam['world_from_cv_rotation'])
        uv = cam['focal']*points[..., :2]/points[..., 2:] + 159.5
        valid = (points[..., 2] > 0).all(1) & ((uv >= 0) & (uv <= 319)).all((1, 2))
        per_camera[cam['name']] = int(valid.sum()); frustum &= valid
    good = []; body = model.body('RedCube').id; start = addresses[0]
    for index in tqdm(np.flatnonzero(frustum), desc='Filter object collisions at home', mininterval=5):
        state.qpos[start:start+3] = xyz[index]
        mujoco.mj_forward(model, state)
        bad = any(c.dist < -1e-6 and (model.geom_bodyid[c.geom1] == body or model.geom_bodyid[c.geom2] == body)
                  for c in state.contact)
        if not bad:
            good.append(index)
    state.qpos[start:start+3] = [-3, 0, .1]
    return xyz[good], base, dict(candidate_voxels=total, reach_verified_voxels=len(accepted),
        whole_cube_in_camera=per_camera, whole_cube_in_both=int(frustum.sum()),
        nonpenetrating_in_both=len(good), nominal_samples_one_object=len(accepted),
        filtered_samples_one_object=len(good),
        note='32mm cube, yaw0, fixed home; rendered robot-occlusion checks not included in full count')


def render_pilot(model, state, addresses, centers, base, output, count=256):
    rng = np.random.default_rng(113)
    rgb_renderer = mujoco.Renderer(model, 320, 320)
    samples = model.vis.quality.offsamples
    model.vis.quality.offsamples = 0
    mask_renderer = mujoco.Renderer(model, 320, 320)
    model.vis.quality.offsamples = samples
    mask_renderer.enable_segmentation_rendering()
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    rgb = []; masks = []; xyz = []; attempts = 0
    started = time.perf_counter()
    with tqdm(total=count, desc='Render paired RGB + masks', mininterval=5) as bar:
        while len(rgb) < count:
            attempts += 1
            assert attempts < count*10, 'Too many invisible pilot samples.'
            target = 0
            point = centers[rng.integers(len(centers))]
            state.qpos[addresses[0]:addresses[0]+3] = point
            yaw = 0.
            state.qpos[addresses[0]+3:addresses[0]+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            mujoco.mj_forward(model, state)
            body = model.body(NAMES[target]).id
            if any(c.dist < -1e-6 and (model.geom_bodyid[c.geom1] == body or model.geom_bodyid[c.geom2] == body)
                   for c in state.contact):
                continue
            images = []; binary = []
            for view in VIEWS:
                rgb_renderer.update_scene(state, camera='cam_'+view, scene_option=option)
                images.append(rgb_renderer.render().copy())
                mask_renderer.update_scene(state, camera='cam_'+view, scene_option=option)
                seg = mask_renderer.render()
                valid = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
                mask = np.zeros((320, 320), bool)
                mask[valid] = model.geom_bodyid[seg[..., 0][valid]] == body
                binary.append(mask)
            if min(int(m.sum()) for m in binary) < 16:
                continue
            rgb.append(np.array(images)); masks.append(np.array(binary)); xyz.append(point-base)
            bar.update(1)
            if len(rgb) <= 4:
                picture = Image.new('RGB', (640, 370), 'white')
                draw = ImageDraw.Draw(picture)
                for v in range(2):
                    pixels = images[v].copy(); pixels[binary[v]] = (.5*pixels[binary[v]]+[0, 127.5, 0]).astype(np.uint8)
                    picture.paste(Image.fromarray(pixels), (320*v, 50))
                draw.text((8, 8), f'{NAMES[target]} | XYZ base m: {np.round(xyz[-1], 3)} | green = query mask', fill='black')
                picture.save(output / f'sample_{len(rgb)}.png')
    rgb_renderer.close(); mask_renderer.close()
    return np.array(rgb), np.array(masks), np.array(xyz, np.float32), dict(samples=count,
        attempts=attempts, seconds=time.perf_counter()-started, renderer_workers=1,
        static_render=True, physics_steps=0, min_target_visible_pixels_per_view=16)


def train_steps(net, optimizer, features, masks, target, batch, steps, online_rgb=None):
    def step():
        ids = torch.randint(len(target), (batch,), device=target.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            selected_mask = masks[ids].cuda()
            pred = (net.from_features(features[ids].cuda(), selected_mask) if online_rgb is None else
                    net(online_rgb[ids], selected_mask))
            loss = F.mse_loss(pred, target[ids].cuda())
        loss.backward(); optimizer.step()
        return loss
    for _ in range(5):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []; losses = []
    mode = 'online' if online_rgb is not None else ('cached_GPU' if features.is_cuda else 'cached_RAM')
    for _ in tqdm(range(steps), desc=f'{mode} train B{batch}', mininterval=5):
        torch.cuda.synchronize(); start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize(); times.append(time.perf_counter()-start); losses.append(float(loss))
    assert np.isfinite(losses).all()
    return dict(batch=batch, measured_steps=steps, median_step_seconds=float(np.median(times)),
        mean_step_seconds=float(np.mean(times)), p90_step_seconds=float(np.percentile(times, 90)),
        samples_per_second=batch/np.mean(times), peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/2**30,
        last_loss=losses[-1], note='pilot sample repetition; timing only, not accuracy')


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copy2('docs/research/e113_README.md', out / 'README.md')
    torch.set_num_threads(4); torch.manual_seed(113)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model, state, addresses = setup(); cams = cameras(model, state)
    centers, base, counts = count_voxels(model, state, addresses, cams)
    write_json(out / 'counts.json', counts); write_json(out / 'cameras.json', dict(cameras=cams, base_origin_world_m=base.tolist()))
    print(json.dumps(counts), flush=True)
    rgb, masks, xyz, render = render_pilot(model, state, addresses, centers, base, out)
    write_json(out / 'render_timing.json', render)
    net = TwoViewXYZ().cuda().train()
    before = state_hash(net.encoder)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, fused=True)
    images = torch.from_numpy(rgb).cuda(); mask = torch.from_numpy(masks).cuda()
    target = torch.from_numpy((xyz-xyz.mean(0))/xyz.std(0)).cuda()
    features = []
    with torch.no_grad():
        for _ in range(3):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                net.encode(images[:16])
        torch.cuda.synchronize(); started = time.perf_counter()
        for start in tqdm(range(0, len(images), 16), desc='Frozen paired-view ViT cache', mininterval=5):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features.append(net.encode(images[start:start+16]).detach())
        torch.cuda.synchronize(); encoding_seconds = time.perf_counter()-started
    features = torch.cat(features)
    assert not features.requires_grad and torch.isfinite(features).all()
    training = []
    for batch in (16, 32, 64, 128):
        timing = train_steps(net, optimizer, features, mask, target, batch, 20)
        training.append(timing); print(json.dumps(dict(cached=timing)), flush=True)
    online = train_steps(net, optimizer, features, mask, target, 32, 20, images)
    ram = train_steps(net, optimizer, features.cpu(), mask.cpu(), target.cpu(), 128, 40)
    assert state_hash(net.encoder) == before
    assert all(not p.requires_grad and p.grad is None for p in net.encoder.parameters())
    best = max(training, key=lambda t: t['samples_per_second'])
    estimates = []
    for scope, count in [('current_camera_filtered', counts['filtered_samples_one_object']),
                         ('all_reach_voxels_nominal', counts['nominal_samples_one_object'])]:
        for variant in (1, 4, 10):
            n = count*variant
            n_train = int(n*.8)
            steps = (n_train+best['batch']-1)//best['batch']
            estimates.append(dict(scope=scope, variants_per_voxel=variant, samples=n, training_samples_80pct=n_train,
                raw_rgb_plus_bool_mask_gib=n*2*320*320*4/2**30,
                fp16_patch_cache_gib=n*2*400*768*2/2**30,
                mask_uint8_gib=n*2*320*320/2**30,
                render_seconds_one_worker=n/len(rgb)*render['seconds'],
                cache_seconds=n/len(rgb)*encoding_seconds,
                cached_train_seconds_per_epoch=steps*best['mean_step_seconds'],
                ram_cached_train_seconds_per_epoch=((n_train+127)//128)*ram['mean_step_seconds'],
                online_train_seconds_per_epoch=((n_train+31)//32)*online['mean_step_seconds']))
    result = dict(status='complete', counts=counts, render=render, encoder_samples=len(images),
        encoding_seconds=encoding_seconds, trainable_parameters=sum(p.numel() for p in params),
        encoder_frozen_unchanged=True, encoder_state_sha256=before, encoder_weights=str(WEIGHTS),
        cached_train=training, ram_cached_train=ram, online_train=online, extrapolation=estimates,
        limitations=['80/10/10 voxel-disjoint split assumed, not executed',
            'No full dataset, convergence or held-out accuracy test',
            'No disk loading/compression or validation cost in training estimate',
            'One red cube per scene; all other objects and covers parked away',
            'Full count excludes robot visibility rejection; yaw0, no jitter'])
    write_json(out / 'summary.json', result)
    print(json.dumps(result), flush=True)
    with (out / 'README.md').open('a') as f:
        f.write('\nBenchmark complete. Exact counts, timings and extrapolations in summary.json. Frozen encoder unchanged; features and repeated-batch weights were kept only in RAM and are discarded on exit. Full training not launched.\n')


if __name__ == '__main__':
    main()
