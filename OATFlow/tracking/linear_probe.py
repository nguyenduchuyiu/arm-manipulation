"""Frozen original TCOW, RAM-cached spatial tokens, and one affine XYZ readout."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from OATFlow.shuffle_bc.data import DATA, splits
from OATFlow.tracking.data import evaluation_plan
from OATFlow.tracking.point import PointXYZ, load_tcow
from OATFlow.tracking.point_data import load_points
from OATFlow.tracking.pretrain_xyz import endpoints, inputs, WEIGHTS
from OATFlow.tracking.utils import write_json


def fingerprint(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode()); digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def cache_features(model, episodes, plan, batch_size, out, name, statistics=False, persist=False):
    # Only the current frame is stored; its tokens already encode the causal video prefix.
    shape = (len(plan), 15 * 20 * 768)
    features = (np.lib.format.open_memmap(out / f'{name}_features.npy', mode='w+', dtype=np.float16, shape=shape)
                if persist else np.empty(shape, dtype=np.float16))
    truth = np.array([episodes[i]['xyz'][end] for i, end, _ in plan], dtype=np.float32)
    total = torch.zeros(features.shape[1], device='cuda', dtype=torch.float64)
    squares = torch.zeros_like(total)
    started = time.perf_counter(); timings = []
    for start in tqdm(range(0, len(plan), batch_size), desc=f'Cache {name}', mininterval=5):
        tick = time.perf_counter(); selected = plan[start:start + batch_size]
        rgb, query, _ = inputs(episodes, [(i, end) for i, end, _ in selected])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value = model.features(rgb, query)[:, -1].flatten(1).to(torch.float16)
        assert value.shape == (len(selected), features.shape[1]) and torch.isfinite(value).all()
        features[start:start + len(selected)] = value.cpu().numpy()
        if statistics:
            total += value.float().sum(0).double()
            squares += value.float().square().sum(0).double()
        elapsed = time.perf_counter() - started
        timings.append(time.perf_counter() - tick)
        record = dict(status='caching', split=name, completed=start + len(selected), total=len(plan),
            elapsed_s=elapsed, seconds_per_clip=elapsed / (start + len(selected)),
            eta_s=elapsed / (start + len(selected)) * (len(plan) - start - len(selected)),
            feature_cache_gib=features.nbytes / 2**30,
            gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(out / 'progress.json', record)
        if start == 0 or (start // batch_size + 1) % 50 == 0:
            print(json.dumps(record), flush=True)
    if persist:
        features.flush()
        np.save(out / f'{name}_xyz.npy', truth)
    stats = None
    if statistics:
        mean = total / len(plan)
        std = (squares / len(plan) - mean.square()).clamp_min(0).sqrt().clamp_min(.01)
        stats = (mean.float(), std.float())
    records = [dict(episode=episodes[i]['row']['path'], raw_frame=int(episodes[i]['raw_index'][end]), phase=phase)
               for i, end, phase in plan]
    write_json(out / f'{name}_points.json', records)
    timing = dict(points=len(plan), elapsed_s=time.perf_counter() - started,
        median_batch_s=float(np.median(timings)), bytes=features.nbytes)
    write_json(out / f'{name}_cache.json', timing)
    print(json.dumps(dict(cache_complete=name, **timing)), flush=True)
    return features, truth, records, stats


def normalized(features, mean, std):
    # Fixed train-only normalization improves conditioning. It folds into one raw-feature Linear.
    return (features.float() - mean) / std / math.sqrt(features.shape[-1])


def partition_train(rows):
    # Keep four target queries from a shared layout/permutation context together.
    layouts = {}
    for row in rows:
        layouts.setdefault(row['layout_id'], {}).setdefault(row['context_path'], []).append(row)
    parts = [[] for _ in range(5)]; rng = np.random.default_rng(107)
    for layout, contexts in sorted(layouts.items()):
        names = sorted(contexts); rng.shuffle(names)
        for rank, name in enumerate(names):
            queries = contexts[name]
            assert sorted(r['target_id'] for r in queries) == [0, 1, 2, 3]
            parts[(rank + layout) % 5].extend(queries)
    assert len({r['path'] for part in parts for r in part}) == len(rows)
    assert sum(map(len, parts)) == len(rows)
    return parts


@torch.inference_mode()
def evaluate(head, features, truth, records, feature_mean, feature_std, xyz_mean, xyz_std, batch=256):
    predictions = []
    for start in range(0, len(features), batch):
        values = features[start:start + batch]
        prediction = head(normalized(values, feature_mean, feature_std)) * xyz_std + xyz_mean
        predictions.append(prediction.cpu().numpy())
    prediction = np.concatenate(predictions)
    errors = np.linalg.norm(prediction - truth, axis=1) * 1000
    phases = np.array([r['phase'] for r in records])
    by_phase = {phase: float(errors[phases == phase].mean()) for phase in sorted(set(phases))}
    result = dict(macro_phase_mm=float(np.mean(list(by_phase.values()))), mean_mm=float(errors.mean()),
        median_mm=float(np.median(errors)), p90_mm=float(np.quantile(errors, .9)),
        mae_xyz_mm=(np.abs(prediction - truth).mean(0) * 1000).tolist(),
        within_mm={str(mm): float((errors <= mm).mean()) for mm in (5, 10, 20, 50)},
        by_phase_mm=by_phase, points=len(features))
    details = [dict(**r, error_mm=float(error), prediction_base_m=pred.tolist(), truth_base_m=gt.tolist())
               for r, error, pred, gt in zip(records, errors, prediction, truth)]
    return result, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--part', type=int, choices=range(1, 6))
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(107)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(107)
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e107_README.md', out / 'README.md')
    groups = splits()
    parts = partition_train(groups['train'])
    write_json(out / 'partitions.json', {str(i + 1): [r['path'] for r in rows] for i, rows in enumerate(parts)})
    train_rows = groups['train'] if args.part is None else parts[args.part - 1]
    if args.preflight: train_rows = train_rows[:4]
    val_rows = [r for r in groups['val'] if r['layout_id'] in (32, 33) and r['permutation'] in (0, 9, 16, 18)]
    if args.preflight: val_rows = val_rows[:4]
    write_json(out / 'progress.json', dict(status='loading RGB', step=0))
    train, segments = load_points(DATA, train_rows, stride=25)
    val, segments = load_points(DATA, val_rows, segments=segments, stride=10)
    points = endpoints(train)
    if args.preflight:
        points = [(i, int(end)) for i, ep in enumerate(train)
                  for end in np.unique(np.rint(np.linspace(0, len(ep['raw_index']) - 1, 16)).astype(int))]
    assert len(set(points)) == len(points)
    train_plan = [(i, end, str(train[i]['phase'][end])) for i, end in points]
    val_plan = evaluation_plan(val)
    config = dict(status='caching', preflight=args.preflight, seed=107, data=str(DATA),
        train_episodes=len(train), validation_episodes=len(val), train_points=len(points),
        train_part=args.part, total_parts=5, all_train_episodes=len(groups['train']),
        partition='context-disjoint, all4target queries together; shuffled contexts round-robin per layout, seed107',
        validation_points=len(val_plan), frames=30, context_stride=25, expert_stride=25, validation_stride=10,
        train_sampling='all cached endpoints once per shuffled epoch; segment terminal frames included' if not args.preflight else '16 evenly spaced endpoints per episode; GPU/code preflight only',
        crop='top 320-square rows40:280 -> 240x320',
        inputs='RGB and radius3 frame0 dot; raw-time causal prefixes; no ID/XYZ/joints/masks/calibration input',
        initialization='original official TCOW RGB/query weights, not e106 or e105',
        architecture='frozen TCOW -> current [15,20,768] -> flatten -> Linear(230400,3)',
        frozen='entire TCOW backbone, eval mode, no_grad; no pooling or segmentation head used',
        trainable_parameters=230400 * 3 + 3, head_initialization='zero weight and bias; mean-coordinate baseline',
        feature_dtype='float16 converted from BF16 backbone tokens',
        feature_cache='disk-backed FP16 NPY current-frame spatial tokens, persisted after cache completion' if args.part else 'host RAM only, current-frame spatial tokens; not saved to disk; lost at process exit',
        head_data_residency='entire selected-part train/validation feature caches and train targets on CUDA; CUDA shuffled indices',
        feature_cache_bytes=len(points) * 230400 * 2,
        feature_normalization='train-only per-feature mean/std (floor .01), then divide by sqrt(230400); fixed affine transform folded into exported raw Linear',
        xyz_normalization='train-only current target XYZ mean/std, std floor10mm',
        loss='MSE of normalized current XYZ only', epochs=3 if args.preflight else 30,
        batch=32 if args.preflight else 256, cache_batch=32,
        optimizer='AdamW(.9,.95), LR1e-3, weight_decay1e-2, grad clip1',
        schedule='linear warmup first100 updates, cosine to10% LR',
        validation='same184 fixed phase endpoints as e106; best macro of five phase mean Euclidean errors',
        test_status='not run; validation only during head training; no action policy',
        gradient_checkpointing=False, gradient_accumulation=False,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path('OATFlow/tracking/point.py'), Path('OATFlow/tracking/backbone.py'),
             Path('OATFlow/tracking/point_data.py'), Path('OATFlow/tracking/data.py'),
             Path('OATFlow/tracking/pretrain_xyz.py'), Path('OATFlow/shuffle_bc/data.py')]},
        weights_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (WEIGHTS / 'tracking_backbone.pt', WEIGHTS / 'tracking_query.pt')})
    write_json(out / 'config.json', config)
    for name in config['source_sha256']:
        source = Path(name).resolve()
        target = out / 'source' / source.relative_to(Path.cwd())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (out / 'source' / 'README.md').write_text('Exact source snapshot; hashes and actual train/test settings in ../config.json and ../README.md.\n')
    write_json(out / 'samples.json', dict(train=[r['path'] for r in train_rows], validation=[r['path'] for r in val_rows]))
    print(json.dumps(config), flush=True)
    if args.part:
        assert shutil.disk_usage(out).free > (len(train_plan) + len(val_plan)) * 230400 * 2 + 2**30
    model = PointXYZ(); load_tcow(model, WEIGHTS / 'tracking_backbone.pt', WEIGHTS / 'tracking_query.pt')
    model.requires_grad_(False); model.eval().cuda()
    before = fingerprint(model.backbone)
    train_features, truth, records, stats = cache_features(model, train, train_plan, config['cache_batch'], out, 'train', True, args.part is not None)
    val_features, val_truth, val_records, _ = cache_features(model, val, val_plan, config['cache_batch'], out, 'val', persist=args.part is not None)
    assert fingerprint(model.backbone) == before
    assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
    # Verify that changing the clicked object changes the cached representation on identical RGB.
    rgb, query, _ = inputs(train, [(0, len(train[0]['raw_index']) - 1)])
    changed = query.clone(); changed[:, :, 0] = query[:, :, 0].roll(64, dims=-1)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        a = model.features(rgb, query)[:, -1].flatten(1).half()
        b = model.features(rgb, changed)[:, -1].flatten(1).half()
    query_difference = float((a.float() - b.float()).abs().mean())
    assert query_difference > 0
    write_json(out / 'frozen_checks.json', dict(backbone_unchanged=True, all_backbone_gradients_none=True,
        backbone_sha256=before, changed_prompt_feature_mean_absolute_difference=query_difference))
    del model, rgb, query, changed, a, b, train, val, segments
    torch.cuda.empty_cache()
    feature_mean, feature_std = (s.clone() for s in stats)
    xyz_mean = torch.tensor(truth.mean(0), device='cuda')
    xyz_std = torch.tensor(truth.std(0).clip(.01), device='cuda')
    required = train_features.nbytes + val_features.nbytes + 2**30
    assert required < torch.cuda.mem_get_info()[0], 'Feature cache does not fit GPU; select one --part 1..5'
    transfer_started = time.perf_counter()
    train_features = torch.from_numpy(train_features).cuda()
    val_features = torch.from_numpy(val_features).cuda()
    train_target = (torch.from_numpy(truth).cuda() - xyz_mean) / xyz_std
    torch.cuda.synchronize()
    assert train_features.is_cuda and val_features.is_cuda and train_target.is_cuda
    residency = dict(train_device=str(train_features.device), val_device=str(val_features.device),
        train_features_gib=train_features.numel() * train_features.element_size() / 2**30,
        val_features_gib=val_features.numel() * val_features.element_size() / 2**30,
        gpu_allocated_gib=torch.cuda.memory_allocated() / 2**30,
        load_seconds=time.perf_counter() - transfer_started)
    write_json(out / 'gpu_cache.json', residency); print(json.dumps(dict(gpu_cache=residency)), flush=True)
    head = nn.Linear(230400, 3, device='cuda'); nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    assert sum(p.numel() for p in head.parameters()) == config['trainable_parameters']
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, betas=(.9, .95), weight_decay=1e-2, fused=True)
    steps_per_epoch = math.ceil(len(truth) / config['batch']); total_steps = config['epochs'] * steps_per_epoch
    config.update(status='training', backbone_sha256=before, xyz_mean=xyz_mean.tolist(), xyz_std=xyz_std.tolist(),
        steps=total_steps, steps_per_epoch=steps_per_epoch)
    write_json(out / 'config.json', config)
    with (out / 'README.md').open('a') as stream:
        stream.write(f'\nActual counts: {len(train_rows)}train episodes, {len(truth)}train endpoints, {len(val_rows)}validation episodes/{len(val_truth)}endpoints; '
            f'{config["epochs"]}epochs ×{steps_per_epoch}updates, batch{config["batch"]}; feature cache{config["feature_cache_bytes"]/2**30:.2f}GiB loaded on GPU for training.\n')
    best = math.inf; best_epoch = 0; step = 0; samples_seen = 0; started = time.perf_counter()

    def save(name, epoch):
        # Export a single Linear that consumes unnormalized raw spatial tokens and returns base metres.
        scale = feature_std * math.sqrt(230400)
        weight = xyz_std[:, None] * head.weight / scale[None]
        bias = xyz_mean + xyz_std * head.bias - weight @ feature_mean
        raw = nn.Linear(230400, 3, device='cuda')
        with torch.no_grad(): raw.weight.copy_(weight); raw.bias.copy_(bias)
        x = val_features[:4]
        expected = head(normalized(x, feature_mean, feature_std)) * xyz_std + xyz_mean
        # TF32 matmuls and cancellation in the folded bias can introduce sub-mm differences.
        torch.testing.assert_close(raw(x.float()), expected, atol=1e-4, rtol=1e-3)
        payload = dict(head={k: v.detach().cpu() for k, v in head.state_dict().items()},
            raw_xyz_head={k: v.detach().cpu() for k, v in raw.state_dict().items()},
            feature_mean=feature_mean.cpu(), feature_std=feature_std.cpu(),
            xyz_mean=xyz_mean.cpu(), xyz_std=xyz_std.cpu(), config=config,
            epoch=epoch, step=step, samples_seen=samples_seen, best_epoch=best_epoch, best_val_mm=best)
        temp = out / (name + '.tmp'); torch.save(payload, temp); temp.replace(out / (name + '.pt'))

    initial, details = evaluate(head, val_features, val_truth, val_records, feature_mean, feature_std, xyz_mean, xyz_std)
    write_json(out / 'initial_val.json', dict(metrics=initial, records=details))
    best = initial['macro_phase_mm']; save('best', 0)
    print(json.dumps(dict(epoch=0, validation=initial)), flush=True)
    try:
        for epoch in range(1, config['epochs'] + 1):
            order = torch.from_numpy(rng.permutation(len(truth))).cuda()
            epoch_loss = 0.; epoch_seen = 0; tick = time.perf_counter()
            progress = tqdm(range(0, len(order), config['batch']), desc=f'Linear epoch {epoch}/{config["epochs"]}', mininterval=5)
            for start in progress:
                ids = order[start:start + config['batch']]; step += 1
                x = train_features[ids]
                target = train_target[ids]
                fraction = (step - 1) / max(total_steps - 1, 1)
                lr = 1e-3 * min(step / min(100, total_steps), 1) * (.1 + .9 * (1 + math.cos(math.pi * fraction)) / 2)
                optimizer.param_groups[0]['lr'] = lr; optimizer.zero_grad(set_to_none=True)
                prediction = head(normalized(x, feature_mean, feature_std))
                loss = (prediction - target).square().mean(); assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); samples_seen += len(ids); epoch_seen += len(ids)
                loss_value = float(loss.detach()); epoch_loss += loss_value * len(ids)
                record = dict(status='training', epoch=epoch, step=step, total_steps=total_steps,
                    samples_seen=samples_seen, train_passes=samples_seen / len(truth), loss=loss_value, lr=lr,
                    batch=len(ids), elapsed_s=time.perf_counter() - started,
                    gpu_allocated_gib=torch.cuda.memory_allocated() / 2**30)
                with (out / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
                if start == 0 or step % 50 == 0:
                    write_json(out / 'progress.json', record); print(json.dumps(record), flush=True)
                progress.set_postfix(loss=f'{loss_value:.4f}', refresh=False)
            assert epoch_seen == len(truth)
            metric, details = evaluate(head, val_features, val_truth, val_records, feature_mean, feature_std, xyz_mean, xyz_std)
            if metric['macro_phase_mm'] < best:
                best = metric['macro_phase_mm']; best_epoch = epoch; save('best', epoch)
            save('latest', epoch)
            record = dict(epoch=epoch, step=step, train_loss=epoch_loss / epoch_seen,
                epoch_seconds=time.perf_counter() - tick, best_epoch=best_epoch, validation=metric)
            with (out / 'validation.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
            write_json(out / 'last_validation.json', dict(**record, records=details))
            print(json.dumps(record), flush=True)
    except KeyboardInterrupt:
        config.update(status='interrupted_by_user', last_logged_step=step)
        write_json(out / 'config.json', config)
        raise
    assert samples_seen == config['epochs'] * len(truth)
    save('final', config['epochs'])
    checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    head.load_state_dict(checkpoint['head'])
    validation, details = evaluate(head, val_features, val_truth, val_records, feature_mean, feature_std, xyz_mean, xyz_std)
    train_metric, _ = evaluate(head, train_features, truth, records, feature_mean, feature_std, xyz_mean, xyz_std)
    result = dict(status='preflight_passed' if args.preflight else 'complete', best_epoch=best_epoch,
        best_step=checkpoint['step'], training_seconds=time.perf_counter() - started,
        validation=validation, training=train_metric, samples_seen=samples_seen)
    write_json(out / 'summary.json', result)
    write_json(out / 'best_validation.json', dict(metrics=validation, records=details))
    config.update(status=result['status'], best_epoch=best_epoch, best_val_mm=best)
    write_json(out / 'config.json', config)
    with (out / 'README.md').open('a') as stream:
        stream.write(f'\nStatus: {result["status"]}; best epoch{best_epoch}, validation macro{best:.3f}mm. '
            'See summary.json and best_validation.json. '
            + ('Cached features persist as train_features.npy and val_features.npy.\n' if args.part else 'RAM cache released on process exit.\n'))
    write_json(out / 'progress.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
