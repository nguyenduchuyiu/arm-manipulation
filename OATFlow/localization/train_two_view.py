"""Render/cache the voxel dataset, train or resume XYZ epochs, and test best."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F
from tqdm import tqdm

from OATFlow.localization.model import WEIGHTS, state_hash
from OATFlow.localization.two_view import TwoViewXYZ
from OATFlow.localization.voxel_budget import DATA, GRID, VIEWS, cameras, count_voxels, setup
from OATFlow.simulation.io import write_json

DATASET = Path('/workspace/robotics_assets/datasets/voxel_xyz_two_view')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def make_data(out):
    if DATASET.exists():
        config = json.loads((DATASET / 'config.json').read_text())
        assert config['status'] == 'complete' and config['version'] == 1
        assert config['grid_sha256'] == sha(GRID) and config['scene_sha256'] == sha(DATA / 'scene.xml')
        print(json.dumps(dict(status='reuse dataset', samples=config['samples'])), flush=True)
        return config
    DATASET.mkdir(parents=True, exist_ok=False)
    (DATASET / 'README.md').write_text('e113 single-object two-view voxel XYZ dataset. One fixed-yaw red cube, fixed top/left cameras and home robot. RGB uint8, masks little-endian bitpacked; sample count and accepted voxel IDs in config.json/labels.npz. Static renders, no physics rollout. Generation/training provenance: /workspace/arm-manipulation/docs/research/e113_train_README.md.\n')
    model, state, addresses = setup()
    cams = cameras(model, state)
    centers, base, counts = count_voxels(model, state, addresses, cams)
    with np.load(GRID) as z:
        lookup = {tuple(p): i for i, p in enumerate(z['centers_world_m'])}
    voxel_ids = np.array([lookup[tuple(p)] for p in centers], np.int32)
    camera_config = dict(cameras=cams, base_origin_world_m=base.tolist())
    write_json(DATASET / 'cameras.json', camera_config)
    write_json(out / 'cameras.json', camera_config)
    config = dict(status='rendering', version=1, counts=counts, grid_sha256=sha(GRID),
        scene_sha256=sha(DATA / 'scene.xml'), data_source=str(DATA), grid=str(GRID),
        object='RedCube', cube_side_m=.032, yaw_rad=0., views=list(VIEWS), resolution=[320, 320],
        base_origin_world_m=base.tolist(), minimum_visible_pixels=16, allocated_rows=len(centers),
        static_render=True, physics_steps=0)
    write_json(DATASET / 'config.json', config)
    rgb = open_memmap(DATASET / 'rgb.npy', mode='w+', dtype=np.uint8, shape=(len(centers), 2, 320, 320, 3))
    packed = open_memmap(DATASET / 'masks_packed.npy', mode='w+', dtype=np.uint8, shape=(len(centers), 2, 320, 40))
    renderer = mujoco.Renderer(model, 320, 320)
    samples = model.vis.quality.offsamples; model.vis.quality.offsamples = 0
    segmenter = mujoco.Renderer(model, 320, 320)
    model.vis.quality.offsamples = samples; segmenter.enable_segmentation_rendering()
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    accepted = []; rejected = []; truth = []; pixel_counts = []
    body = model.body('RedCube').id; started = time.perf_counter()
    for i in tqdm(range(len(centers)), desc='Render all voxel RGB/masks', mininterval=5):
        state.qpos[addresses[0]:addresses[0]+3] = centers[i]
        mujoco.mj_forward(model, state)
        np.testing.assert_allclose(state.xpos[body], centers[i], atol=1e-12, rtol=0)
        binary = []
        for view in VIEWS:
            segmenter.update_scene(state, camera='cam_'+view, scene_option=option)
            seg = segmenter.render()
            valid = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            mask = np.zeros((320, 320), bool)
            mask[valid] = model.geom_bodyid[seg[..., 0][valid]] == body
            binary.append(mask)
        pixels = np.array(binary).sum((1, 2))
        if pixels.min() < 16:
            rejected.append(int(voxel_ids[i]))
            continue
        row = len(accepted)
        for v, view in enumerate(VIEWS):
            renderer.update_scene(state, camera='cam_'+view, scene_option=option)
            rgb[row, v] = renderer.render()
        packed[row] = np.packbits(np.array(binary), axis=-1, bitorder='little')
        accepted.append(voxel_ids[i]); truth.append(state.xpos[body].copy()-base); pixel_counts.append(pixels)
        if i % 256 == 0:
            write_json(out / 'progress.json', dict(status='rendering', processed=i+1, total=len(centers),
                accepted=len(accepted), rejected=len(rejected), elapsed_s=time.perf_counter()-started))
    renderer.close(); segmenter.close(); rgb.flush(); packed.flush(); del rgb, packed
    np.savez(DATASET / 'labels.npz', voxel_id=np.array(accepted, np.int32), xyz_base_m=np.array(truth, np.float32),
        visible_pixels=np.array(pixel_counts, np.int32), rejected_voxel_id=np.array(rejected, np.int32))
    config.update(status='complete', samples=len(accepted), rejected_visibility=len(rejected),
        render_seconds=time.perf_counter()-started, labels_sha256=sha(DATASET / 'labels.npz'))
    write_json(DATASET / 'config.json', config)
    print(json.dumps(dict(dataset=config)), flush=True)
    return config


def metrics(prediction, truth):
    delta = (prediction-truth)*1000
    error = np.linalg.norm(delta, axis=-1)
    assert np.isfinite(error).all()
    return dict(samples=len(error), mean_mm=float(error.mean()), median_mm=float(np.median(error)),
        p90_mm=float(np.percentile(error, 90)), p95_mm=float(np.percentile(error, 95)), max_mm=float(error.max()),
        axis_mae_mm=np.abs(delta).mean(0).tolist(), within_1mm=float((error <= 1).mean()),
        within_5mm=float((error <= 5).mean()), within_10mm=float((error <= 10).mean()))


@torch.no_grad()
def evaluate(net, features, masks, labels, indices, mean, std, name):
    net.eval(); values = []
    for start in tqdm(range(0, len(indices), 128), desc=name, mininterval=5):
        ids = indices[start:start+128]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = net.from_features(features[ids].cuda(), masks[ids].cuda())
        values.append(prediction.cpu().numpy()*std+mean)
    prediction = np.concatenate(values)
    return metrics(prediction, labels[indices]), prediction


def trainable_state(net):
    return {k: v.detach().cpu() for k, v in net.state_dict().items() if not k.startswith('encoder.')}


def save_examples(out, rgb, masks, labels, indices, prediction, name):
    cams = json.loads((DATASET / 'cameras.json').read_text())
    error = np.linalg.norm(prediction-labels[indices], axis=-1)*1000
    chosen = np.argsort(error)[np.rint(np.linspace(0, len(error)-1, 4)).astype(int)]
    image = Image.new('RGB', (1280, 4*540), '#f5f7fa')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    for row, index in enumerate(chosen):
        sample = indices[index]; y = row*540
        draw.text((12, y+5), f'{name} | sample {sample} | error {error[index]:.2f} mm | GT green / prediction magenta', fill='#17283f', font=font)
        draw.text((12, y+28), f'GT XYZ: {labels[sample].round(4)} m   Pred: {prediction[index].round(4)} m', fill='#17283f', font=font)
        for view, cam in enumerate(cams['cameras']):
            pixels = np.array(rgb[sample, view], copy=True)
            region = masks[sample, view].numpy()
            pixels[region] = (.65*pixels[region]+[0, 89.25, 0]).astype(np.uint8)
            image.paste(Image.fromarray(pixels).resize((480, 480)), (view*640, y+55))
            for point, color in ((labels[sample], '#00e599'), (prediction[index], '#ff27b4')):
                p = (point + np.array(cams['base_origin_world_m']) - cam['world_position']) @ np.array(cam['world_from_cv_rotation'])
                uv = (cam['focal']*p[:2]/p[2]+159.5)*1.5 + [view*640, y+55]
                x, yy = uv
                draw.ellipse((x-7, yy-7, x+7, yy+7), outline=color, width=3)
    image.save(out / f'{name}_examples.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=10, help='Total epochs, including resumed epochs.')
    ap.add_argument('--resume', type=Path, help='Latest checkpoint with optimizer and RNG states.')
    args = ap.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=False)
    readme = 'e113_resume_README.md' if args.resume else 'e113_train_README.md'
    shutil.copy2(Path('docs/research') / readme, out / 'README.md')
    torch.set_num_threads(4); torch.manual_seed(113)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert resumed['epoch'] < args.epochs
        assert all(k in resumed for k in ('optimizer', 'rng_state', 'history', 'torch_rng', 'cuda_rng'))
    sources = [Path(__file__), Path('OATFlow/localization/two_view.py'),
               Path('OATFlow/localization/voxel_budget.py'), Path('OATFlow/localization/model.py'),
               Path('OATFlow/simulation/render.py')]
    config = dict(status='preparing', epochs=args.epochs, batch=128, lr=1e-4, weight_decay=.01, seed=113,
        optimizer='AdamW fused; constant lr', loss='MSE normalized XYZ, train-only statistics',
        encoder=str(WEIGHTS), encoder_file_sha256=sha(WEIGHTS), encoder_frozen=True,
        dataset=str(DATASET), source_sha256={str(p): sha(p) for p in sources},
        views=list(VIEWS), image_size=320, tokens_per_view=400, decoder_layers=4,
        decoder_width=384, heads=6, mlp_width=1536, query_tokens=1, cache='FP16 host RAM only',
        mask_input='visible binary mask per current view; mean16x16 occupancy embedded separately',
        dtype='BF16 autocast, FP32 weights and losses; TF32', gradient_checkpointing=False,
        joint_input=False, camera_parameters_input=False, action_control=False)
    if resumed:
        config.update(resume_from=str(args.resume.resolve()), resume_sha256=sha(args.resume),
            start_epoch=resumed['epoch'], start_step=resumed['step'])
        assert all(resumed['config'][k] == config[k] for k in ('batch', 'lr', 'weight_decay', 'seed',
            'encoder_file_sha256', 'tokens_per_view', 'decoder_layers', 'decoder_width', 'heads', 'mlp_width'))
    write_json(out / 'config.json', config)
    for p in sources:
        shutil.copy2(p, out / f'source_{p.parent.name}_{p.name}')
    dataset_config = make_data(out)
    config['dataset_config'] = dataset_config
    shutil.copy2(DATASET / 'cameras.json', out / 'cameras.json')
    with np.load(DATASET / 'labels.npz') as z:
        labels = z['xyz_base_m'].copy(); voxels = z['voxel_id'].copy()
    n = len(labels); assert len(np.unique(voxels)) == n
    rgb = np.load(DATASET / 'rgb.npy', mmap_mode='r')[:n]
    packed = np.load(DATASET / 'masks_packed.npy', mmap_mode='r')[:n]
    masks = torch.empty((n, 2, 320, 320), dtype=torch.bool)
    for start in tqdm(range(0, n, 256), desc='Unpack target masks to RAM', mininterval=5):
        masks[start:start+256] = torch.from_numpy(np.unpackbits(packed[start:start+256], axis=-1, count=320, bitorder='little').astype(bool))
    if resumed:
        assert sha(DATASET / 'labels.npz') == resumed['config']['dataset_config']['labels_sha256']
        with np.load(args.resume.parent / 'split.npz') as z:
            np.testing.assert_array_equal(z['voxel_id'], voxels)
            train, val, test = (z[k].copy() for k in ('train', 'val', 'test'))
        shutil.copy2(args.resume.parent / 'split.npz', out / 'split.npz')
    else:
        order = np.random.default_rng(113).permutation(n)
        train, val, test = np.split(order, [int(n*.8), int(n*.9)])
        np.savez(out / 'split.npz', train=train, val=val, test=test, voxel_id=voxels)
    assert not (set(voxels[train]) & set(voxels[val]) or set(voxels[train]) & set(voxels[test]) or set(voxels[val]) & set(voxels[test]))
    mean = labels[train].mean(0); std = labels[train].std(0); assert (std > 0).all()
    if resumed:
        np.testing.assert_array_equal(mean, resumed['mean'])
        np.testing.assert_array_equal(std, resumed['std'])
    normalized = torch.from_numpy((labels-mean)/std)
    net = TwoViewXYZ().cuda()
    if resumed:
        missing, unexpected = net.load_state_dict(resumed['model'], strict=False)
        assert not unexpected and all(k.startswith('encoder.') for k in missing)
        assert all(torch.equal(net.state_dict()[k].cpu(), v) for k, v in resumed['model'].items())
    before = state_hash(net.encoder)
    if resumed:
        assert before == resumed['encoder_state_sha256']
    features = torch.empty((n, 2, 400, 768), dtype=torch.float16)
    started = time.perf_counter(); net.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n, 16), desc='Cache frozen two-view ViT to RAM', mininterval=5):
            images = torch.from_numpy(np.array(rgb[start:start+16], copy=True)).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                feature = net.encode(images)
            assert torch.isfinite(feature).all()
            features[start:start+len(feature)] = feature.half().cpu()
            if start % 256 == 0:
                write_json(out / 'progress.json', dict(status='caching', samples_done=start+len(feature),
                    samples=n, elapsed_s=time.perf_counter()-started))
    cache_seconds = time.perf_counter()-started
    config.update(status='training', samples=n, split_counts=dict(train=len(train), val=len(val), test=len(test)),
        mean_base_m=mean.tolist(), std_base_m=std.tolist(), encoder_state_sha256=before,
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        cache_seconds=cache_seconds, cache_gib=features.numel()*features.element_size()/2**30,
        mask_ram_gib=masks.numel()*masks.element_size()/2**30)
    write_json(out / 'config.json', config); print(json.dumps(config), flush=True)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01, fused=True)
    rng = np.random.default_rng(114); best = float('inf'); history = []; step = 0
    start_epoch = 0; previous_elapsed = 0.
    if resumed:
        optimizer.load_state_dict(resumed['optimizer'])
        step = resumed['step']; start_epoch = resumed['epoch']; history = resumed['history']
        assert len(history) == start_epoch and history[-1]['step'] == step
        optimizer_steps = sorted({int(v['step']) for v in optimizer.state.values()})
        assert optimizer_steps == [step]
        assert optimizer.param_groups[0]['lr'] == config['lr']
        best = min(row['validation']['mean_mm'] for row in history)
        previous_elapsed = history[-1]['elapsed_s']
        prior_best = args.resume.parent / 'best.pt'
        saved_best = torch.load(prior_best, map_location='cpu', weights_only=False)
        assert saved_best['validation']['mean_mm'] == best
        shutil.copy2(prior_best, out / 'best.pt')
        baseline, _ = evaluate(net, features, masks, labels, val, mean, std, 'Verify resumed validation')
        assert abs(baseline['mean_mm']-resumed['validation']['mean_mm']) < 1e-5
        checks = dict(status='passed', epoch_restored=start_epoch, step_restored=step,
            model_exact=True, optimizer_steps=optimizer_steps, optimizer_lr=optimizer.param_groups[0]['lr'],
            original_split_sha256=sha(args.resume.parent / 'split.npz'), split_sha256=sha(out / 'split.npz'),
            normalization_exact=True, encoder_exact=True, baseline_validation=baseline,
            previous_best_validation_mm=best)
        write_json(out / 'resume_checks.json', checks); print(json.dumps(dict(resume_checks=checks)), flush=True)
        rng.bit_generator.state = resumed['rng_state']
        torch.set_rng_state(resumed['torch_rng']); torch.cuda.set_rng_state(resumed['cuda_rng'])
        del saved_best, resumed
    training_started = time.perf_counter()
    with (out / 'metrics.jsonl').open('w', buffering=1) as stream:
        for epoch in range(start_epoch+1, args.epochs+1):
            net.train(); chosen = rng.permutation(train); loss_sum = 0.; error_sum = 0.; seen = 0
            started = time.perf_counter()
            bar = tqdm(range(0, len(chosen), 128), desc=f'XYZ epoch{epoch}/{args.epochs}', mininterval=5)
            for start in bar:
                ids = chosen[start:start+128]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = net.from_features(features[ids].cuda(), masks[ids].cuda())
                    loss = F.mse_loss(prediction, normalized[ids].cuda())
                assert torch.isfinite(loss)
                loss.backward(); optimizer.step(); step += 1
                pred = prediction.detach().cpu().numpy()*std+mean
                error = np.linalg.norm(pred-labels[ids], axis=-1)*1000
                loss_value = float(loss.detach())
                seen += len(ids); loss_sum += loss_value*len(ids); error_sum += float(error.sum())
                bar.set_postfix(loss=f'{loss_value:.5f}', mm=f'{error.mean():.2f}', refresh=False)
                if step % 10 == 0:
                    record = dict(status='training', epoch=epoch, step=step, epoch_samples=seen,
                        loss=loss_value, xyz_mm=float(error.mean()), elapsed_s=previous_elapsed+time.perf_counter()-training_started)
                    stream.write(json.dumps(record)+'\n'); print(json.dumps(record), flush=True)
                    write_json(out / 'progress.json', record)
            train_seconds = time.perf_counter()-started
            val_metrics, _ = evaluate(net, features, masks, labels, val, mean, std, 'Validation XYZ')
            row = dict(epoch=epoch, step=step, training_mse=loss_sum/seen, training_mean_mm=error_sum/seen,
                train_seconds=train_seconds, validation=val_metrics, elapsed_s=previous_elapsed+time.perf_counter()-training_started)
            history.append(row); stream.write(json.dumps(row)+'\n'); print(json.dumps(row), flush=True)
            checkpoint = dict(model=trainable_state(net), mean=mean, std=std, config=config,
                epoch=epoch, step=step, validation=val_metrics, encoder_state_sha256=before)
            if val_metrics['mean_mm'] < best:
                best = val_metrics['mean_mm']; torch.save(checkpoint, out / 'best.pt')
            torch.save(dict(checkpoint, optimizer=optimizer.state_dict(), rng_state=rng.bit_generator.state,
                            history=history, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), out / 'latest.pt')
            write_json(out / 'history.json', history)
    best_checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    missing, unexpected = net.load_state_dict(best_checkpoint['model'], strict=False)
    assert not unexpected and all(k.startswith('encoder.') for k in missing)
    test_metrics, test_prediction = evaluate(net, features, masks, labels, test, mean, std, 'Held-out test best')
    train_metrics, train_prediction = evaluate(net, features, masks, labels, train, mean, std, 'Train best')
    val_metrics, _ = evaluate(net, features, masks, labels, val, mean, std, 'Validation best')
    np.savez(out / 'test_predictions.npz', indices=test, voxel_id=voxels[test],
        truth_base_m=labels[test], prediction_base_m=test_prediction)
    # Fresh RGB inference checks the complete path against the cached evaluation path.
    ids = test[:8]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        online = net(torch.from_numpy(np.array(rgb[ids], copy=True)).cuda(), masks[ids].cuda())
        cached = net.from_features(features[ids].cuda(), masks[ids].cuda())
    parity = np.linalg.norm((online.cpu().numpy()-cached.cpu().numpy())*std, axis=-1)*1000
    assert float(parity.max()) < 1., 'Online/cache XYZ discrepancy exceeds 1mm.'
    assert state_hash(net.encoder) == before
    assert all(not p.requires_grad and p.grad is None for p in net.encoder.parameters())
    assert sha(WEIGHTS) == config['encoder_file_sha256']
    save_examples(out, rgb, masks, labels, test, test_prediction, 'test')
    save_examples(out, rgb, masks, labels, train, train_prediction, 'train')
    result = dict(status='complete', epochs=args.epochs, epochs_this_run=args.epochs-start_epoch,
        start_epoch=start_epoch, steps=step, samples=n, split_counts=config['split_counts'],
        best_epoch=best_checkpoint['epoch'], best_step=best_checkpoint['step'],
        train=train_metrics, validation=val_metrics, test=test_metrics,
        training_and_evaluation_seconds=time.perf_counter()-training_started,
        cache_seconds=cache_seconds, render_seconds=dataset_config['render_seconds'],
        encoder_frozen_unchanged=True, online_cache_mean_difference_mm=float(parity.mean()),
        online_cache_max_difference_mm=float(parity.max()), cache_persisted=False,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        checkpoint_sha256=sha(out / 'best.pt'), action_control=False)
    write_json(out / 'summary.json', result); write_json(out / 'progress.json', result)
    config.update(status='complete'); write_json(out / 'config.json', config)
    with (out / 'README.md').open('a') as f:
        f.write(f'\nCompleted{args.epochs}total epochs/{step}updates on{n}samples; split{config["split_counts"]}. '
            f'Best epoch{result["best_epoch"]}: validation{val_metrics["mean_mm"]:.3f}mm, '
            f'test{test_metrics["mean_mm"]:.3f}mm. Encoder unchanged; '
            f'online/cache max difference{parity.max():.4f}mm. Exact metrics in summary.json. '
            'FP16 feature cache existed only in host RAM and is released on exit.\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
