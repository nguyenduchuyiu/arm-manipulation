"""Train shared ContextDecoder XYZ on frozen cached top/second-camera tokens."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from OATFlow.shuffle_bc.context_model import ContextPolicy
from OATFlow.top_pick.data import SOURCE
from OATFlow.tracking.linear_probe import fingerprint
from OATFlow.tracking.utils import write_json


def metrics(prediction, truth, records):
    errors = np.linalg.norm(prediction - truth, axis=-1) * 1000
    phases = np.array([r['phase'] for r in records])
    by_phase = {p: float(errors[phases == p].mean()) for p in sorted(set(phases))}
    value = dict(macro_phase_mm=float(np.mean(list(by_phase.values()))), mean_mm=float(errors.mean()),
        median_mm=float(np.median(errors)), p90_mm=float(np.quantile(errors, .9)), by_phase_mm=by_phase,
        mae_xyz_mm=(np.abs(prediction - truth).mean(0) * 1000).tolist(),
        within_mm={str(mm): float((errors <= mm).mean()) for mm in (5, 10, 20, 50)}, points=len(truth))
    details = [dict(**r, error_mm=float(error), prediction_base_m=p.tolist(), truth_base_m=t.tolist())
               for r, error, p, t in zip(records, errors, prediction, truth)]
    return value, details


def gpu_data(cache, name):
    top = Path(json.loads((cache / 'config.json').read_text())['top_source'])
    truth = np.load(top / f'{name}_xyz.npy')
    data = dict(top=torch.from_numpy(np.load(top / f'{name}_features.npy').reshape(-1, 300, 768)).cuda(),
        wrist=torch.from_numpy(np.load(cache / f'{name}_wrist.npy')).cuda(),
        joints=torch.from_numpy(np.load(cache / f'{name}_joints.npy')).cuda(),
        xyz=torch.from_numpy(truth).cuda(), truth=truth,
        records=json.loads((cache / f'{name}_points.json').read_text()))
    assert all(data[k].is_cuda and not data[k].requires_grad for k in ('top', 'wrist', 'joints', 'xyz'))
    assert len({len(data[k]) for k in ('top', 'wrist', 'joints', 'xyz', 'records')}) == 1
    return data


def batch(data, ids):
    return [data[k][ids] for k in ('top', 'wrist', 'joints')]


@torch.inference_mode()
def evaluate(model, data, batch_size=32, desc='Context XYZ validation'):
    model.eval(); predictions = []
    for start in tqdm(range(0, len(data['truth']), batch_size), desc=desc, mininterval=5, leave=False):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model.xyz_from_features(*batch(data, slice(start, start + batch_size)))
        predictions.append((prediction * model.xyz_std + model.xyz_mean).cpu().numpy())
    return metrics(np.concatenate(predictions), data['truth'], data['records'])


@torch.inference_mode()
def target_probe(model, data):
    groups = {}
    for i, point in enumerate(data['records']):
        if point['phase'] == 'post_shuffle':
            groups.setdefault(point['episode'].rsplit('/', 1)[0], []).append(i)
    original = []; changed = []; source = []; target = []; details = []
    for context, ids in sorted(groups.items()):
        ids = sorted(ids, key=lambda i: data['records'][i]['episode'])
        assert len(ids) == 4
        index = torch.tensor(ids, device='cuda'); swapped = index.roll(1)
        torch.testing.assert_close(data['wrist'][index], data['wrist'][swapped], rtol=0, atol=0)
        torch.testing.assert_close(data['joints'][index], data['joints'][swapped], rtol=0, atol=0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            a = model.xyz_from_features(*batch(data, index)) * model.xyz_std + model.xyz_mean
            b = model.xyz_from_features(data['top'][swapped], data['wrist'][index], data['joints'][index]) * model.xyz_std + model.xyz_mean
        gt = data['xyz'][index]; swapped_gt = data['xyz'][swapped]
        nearest = torch.cdist(a.float(), gt.float()).argmin(-1)
        nearest_swapped = torch.cdist(b.float(), gt.float()).argmin(-1)
        expected_swapped = torch.arange(4, device='cuda').roll(1)
        original.extend((nearest == torch.arange(4, device='cuda')).cpu().tolist())
        changed.extend((nearest_swapped == expected_swapped).cpu().tolist())
        source.extend(((a - gt).norm(dim=-1) * 1000).cpu().tolist())
        target.extend(((b - swapped_gt).norm(dim=-1) * 1000).cpu().tolist())
        details.append(dict(context=context, predictions=a.cpu().tolist(), swapped_predictions=b.cpu().tolist(),
            truth=gt.cpu().tolist(), swapped_truth=swapped_gt.cpu().tolist(),
            mean_prediction_change_mm=float((a - b).norm(dim=-1).mean() * 1000)))
    assert groups
    return dict(contexts=len(groups), queries=len(source), same_wrist_and_state_verified=True,
        original_nearest_target_accuracy=float(np.mean(original)), swapped_nearest_target_accuracy=float(np.mean(changed)),
        original_xyz_mean_mm=float(np.mean(source)), swapped_xyz_mean_mm=float(np.mean(target)), records=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(); assert args.batch > 0
    torch.set_num_threads(4); torch.manual_seed(108)
    torch.backends.cuda.matmul.allow_tf32 = True; rng = np.random.default_rng(108)
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e108_README.md', out / 'README.md')
    cache_config = json.loads((args.cache / 'config.json').read_text()); assert cache_config['status'] == 'complete'
    started = time.perf_counter(); write_json(out / 'progress.json', dict(status='loading GPU feature cache'))
    train = gpu_data(args.cache, 'train'); val = gpu_data(args.cache, 'val')
    model = ContextPolicy(); model.flow.requires_grad_(False)
    model.xyz_mean.copy_(torch.tensor(train['truth'].mean(0)))
    model.xyz_std.copy_(torch.tensor(train['truth'].std(0).clip(.01)))
    frozen_before = {name: fingerprint(getattr(model, name)) for name in ('flow', 'wrist')}
    frozen_before['tracker'] = fingerprint(model.tracker.backbone)
    assert frozen_before['tracker'] == cache_config['top_backbone_sha256']
    assert frozen_before['wrist'] == cache_config['vision_module_sha256']
    model.cuda(); model.tracker.cpu(); model.wrist.cpu(); torch.cuda.empty_cache()
    assert not model.context_decoder.gradient_checkpointing
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert all(not name.startswith(('flow.', 'tracker.', 'wrist.')) for name, _ in parameters)
    optimizer = torch.optim.AdamW([
        dict(params=[p for name, p in parameters if not name.startswith('xyz_head.')], lr=1e-4),
        dict(params=[p for name, p in parameters if name.startswith('xyz_head.')], lr=1e-5)],
        betas=(.9, .95), weight_decay=1e-4, fused=True)
    epochs = 30; epoch_steps = math.ceil(len(train['truth']) / args.batch)
    total_steps = 8 if args.preflight else epochs * epoch_steps
    config = dict(status='preflight' if args.preflight else 'training', seed=108, cache=str(args.cache),
        train_episodes=412, train_points=len(train['truth']), val_episodes=32, val_points=len(val['truth']),
        train_part=1, train_stride=25, validation_stride=10, surrogate_camera='left', actual_wrist_camera=False,
        architecture='frozen TCOW300tokens + frozen ViT64tokens + reach state token -> four-block ContextDecoder ->365x960context -> LN/flatten/LinearXYZ; full context also feeds reach FM',
        trainable='visual projections,view embeddings,4context blocks,XYZ head',
        frozen='original TCOW, ImageNet ViT-B/16, entire reach FM including state encoder',
        trainable_parameters=sum(p.numel() for _, p in parameters), context_tokens=365, context_width=960,
        top_feature_source=cache_config['top_source'], source_tokens_recomputed=False,
        FM_source=str(SOURCE), FM_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        FM_source_step=115000, action_representation='10D EE deltaXYZ/deltaRotation6D/absolute jaw',
        action_training=False, loss='current train-normalized XYZ MSE only, weight1',
        epochs=None if args.preflight else epochs, updates=total_steps, updates_per_epoch=epoch_steps, batch=args.batch,
        precision='BF16, FP32loss/parameters, TF32allowed', gradient_checkpointing=False, gradient_accumulation=False,
        optimizer='AdamW(.9,.95),wd1e-4,clip1', learning_rates=dict(context=1e-4, xyz_head=1e-5),
        warmup=50, schedule='cosine to10% LR', xyz_mean=model.xyz_mean.tolist(), xyz_std=model.xyz_std.tolist(),
        data_residency='complete top/wrist/joint/XYZ arrays on cuda; frozen vision modules offloaded to CPU',
        gpu_cache_allocated_gib=torch.cuda.memory_allocated() / 2**30,
        validation='same184fixed phase points; best macro of five phase means', test_status='held-out test not run yet',
        frozen_sha256=frozen_before,
        context_initial_sha256=fingerprint(model.context_decoder),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path('OATFlow/shuffle_bc/context_model.py'), Path('OATFlow/shuffle_bc/context_cache.py'),
             Path('OATFlow/policy/model.py'), Path('OATFlow/reach/model.py'), Path('OATFlow/reach/flow.py')]})
    write_json(out / 'config.json', config)
    for name in config['source_sha256']:
        p = Path(name).resolve(); dest = out / 'source' / p.relative_to(Path.cwd())
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    (out / 'source' / 'README.md').write_text('Exact source snapshot; hashes/train/test settings in ../config.json and ../README.md.\n')
    print(json.dumps(config), flush=True)
    initial, details = evaluate(model, val)
    write_json(out / 'initial_val.json', dict(metrics=initial, records=details))
    best = initial['macro_phase_mm']; best_epoch = 0; step = 0; samples_seen = 0; timings = []

    def save(name, epoch):
        payload = dict(model=model.context_state(), config=config, epoch=epoch, step=step,
            samples_seen=samples_seen, best_epoch=best_epoch, best_val_mm=best)
        temp = out / (name + '.tmp'); torch.save(payload, temp); temp.replace(out / (name + '.pt'))

    if not args.preflight: save('best', 0)
    training_started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, (1 if args.preflight else epochs) + 1):
        model.train(); order = torch.from_numpy(rng.permutation(len(train['truth']))).cuda()
        epoch_started = time.perf_counter(); loss_sum = 0.; epoch_seen = 0
        count = min(8 * args.batch, len(order)) if args.preflight else len(order)
        progress = tqdm(range(0, count, args.batch), desc=f'Context XYZ epoch {epoch}', mininterval=5)
        for start in progress:
            tick = time.perf_counter(); ids = order[start:start + args.batch]; step += 1
            for group, lr in zip(optimizer.param_groups, (1e-4, 1e-5)):
                group['lr'] = lr * min(step / 50, 1) * (.1 + .9 * (1 + math.cos(math.pi * (step - 1) / max(total_steps - 1, 1))) / 2)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model.xyz_from_features(*batch(train, ids))
            loss = F.mse_loss(prediction, model.normalize_xyz(train['xyz'][ids])); assert torch.isfinite(loss)
            loss.backward()
            if step == 1:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for _, p in parameters)
                assert all(p.grad is None for name in ('flow', 'tracker', 'wrist') for p in getattr(model, name).parameters())
                write_json(out / 'gradient_checks.json', dict(all_trainable_parameters_finite_nonzero=True,
                    frozen_gradients_none=True, XYZ_shapes_context_decoder=True, gradient_checkpointing=False))
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1., error_if_nonfinite=True)
            optimizer.step(); samples_seen += len(ids); epoch_seen += len(ids)
            mm = float(((prediction.detach() * model.xyz_std + model.xyz_mean - train['xyz'][ids]).norm(dim=-1).mean() * 1000))
            seconds = time.perf_counter() - tick; timings.append(seconds); loss_value = float(loss.detach())
            loss_sum += loss_value * len(ids)
            record = dict(status='training', epoch=epoch, step=step, total_steps=total_steps, batch=len(ids),
                loss=loss_value, current_xyz_mm=mm, grad_norm=float(norm), samples_seen=samples_seen,
                train_passes=samples_seen / len(train['truth']), batch_seconds=seconds,
                elapsed_s=time.perf_counter() - training_started, gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            with (out / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
            if step == 1 or step % 10 == 0:
                write_json(out / 'progress.json', record); print(json.dumps(record), flush=True)
            progress.set_postfix(loss=f'{loss_value:.4f}', mm=f'{mm:.1f}', refresh=False)
        if not args.preflight: assert epoch_seen == len(train['truth'])
        optimizer.zero_grad(set_to_none=True)
        metric, details = evaluate(model, val)
        if metric['macro_phase_mm'] < best:
            best = metric['macro_phase_mm']; best_epoch = epoch
            if not args.preflight: save('best', epoch)
        if not args.preflight: save('latest', epoch)
        record = dict(epoch=epoch, step=step, train_loss=loss_sum / epoch_seen,
            seconds=time.perf_counter() - epoch_started, validation=metric, best_epoch=best_epoch)
        with (out / 'validation.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
        write_json(out / 'last_validation.json', dict(**record, records=details)); print(json.dumps(record), flush=True)
    if not args.preflight:
        assert samples_seen == epochs * len(train['truth'])
        save('final', epochs)
        saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=False, mmap=True)
        model.load_context_state(saved['model'])
    model.eval(); optimizer.zero_grad(set_to_none=True)
    if args.preflight:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            context, kv = model.context(*batch(val, slice(0, 2)), with_kv=True)
            assert context.shape == (2, 365, 960) and len(kv) == 8 and all(k.shape[1] == 365 for k, _ in kv)
            actions = torch.zeros(2, 25, 10, device='cuda'); mask = torch.ones(2, 25, dtype=torch.bool, device='cuda')
            flow, _ = model.loss_parts_features(*batch(val, slice(0, 2)), actions, mask, val['xyz'][:2])
        gradient = torch.autograd.grad(flow, model.top_projection[1].weight)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        with torch.autocast('cuda', dtype=torch.bfloat16):
            action = model.sample_features(*batch(val, slice(0, 2)), model.flow.sample_noise(2, 'cuda'), steps=2)
        assert action.shape == (2, 25, 10) and torch.isfinite(action).all()
        write_json(out / 'FM_interface.json', dict(context_shape=[2, 365, 960], KV_sets=8,
            action_shape=list(action.shape), finite_actions=True, action_loss_gradient_reaches_context=True,
            action_quality_tested=False, flow_steps=2))
    frozen_after = {name: fingerprint(getattr(model, name)) for name in ('flow', 'wrist')}
    frozen_after['tracker'] = fingerprint(model.tracker.backbone); assert frozen_after == frozen_before
    write_json(out / 'frozen_checks.json', dict(unchanged=True, sha256=frozen_after))
    metric, details = evaluate(model, val)
    write_json(out / 'best_validation.json', dict(metrics=metric, records=details))
    probe = target_probe(model, val); write_json(out / 'target_probe.json', probe)
    fit = None
    if not args.preflight: fit, _ = evaluate(model, train, batch_size=args.batch, desc='Best context train fit')
    result = dict(status='preflight_passed' if args.preflight else 'complete', best_epoch=best_epoch,
        best_val_mm=best, validation=metric, train_fit=fit, steps=step, samples_seen=samples_seen,
        training_seconds=time.perf_counter() - training_started, elapsed_s=time.perf_counter() - started,
        median_batch_seconds=float(np.median(timings[2:])), peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        target_probe={k: v for k, v in probe.items() if k != 'records'})
    config.update(status=result['status'], best_epoch=best_epoch, best_val_mm=best)
    write_json(out / 'config.json', config); write_json(out / 'summary.json', result); write_json(out / 'progress.json', result)
    with (out / 'README.md').open('a') as stream:
        stream.write(f'\nActual {result["status"]}: batch{args.batch},{step}updates,{samples_seen}samples; '
            f'best epoch{best_epoch}, validation macro{best:.3f}mm. Exact metrics/probe in summary.json. '
            f'Median update{result["median_batch_seconds"]:.4f}s, peak GPU{result["peak_gpu_gib"]:.2f}GiB. '
            'FM/TCOW/ViT hashes unchanged. No action-control test.\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
