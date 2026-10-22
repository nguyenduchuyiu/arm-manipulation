"""Full-finetune one-camera TCOW with a frame0 dot query and XYZ output."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.tracking.data import evaluation_plan, subset
from OATFlow.tracking.point import PointXYZ, load_tcow
from OATFlow.tracking.point_data import load_points, point_clip
from OATFlow.tracking.utils import save, summary, write_json


@torch.no_grad()
def evaluate(model, episodes, description, phases=None):
    model.eval()
    plan = evaluation_plan(episodes)
    if phases is not None:
        plan = [point for point in plan if point[2] in phases]
    prediction, truth, hidden, phase_names, records = [], [], [], [], []
    for start in tqdm(range(0, len(plan), 4), desc=description, mininterval=5):
        points = plan[start:start + 4]
        batches = [point_clip(episodes[i], end) for i, end, _ in points]
        rgb, query = (torch.cat([batch[j] for batch in batches]).cuda() for j in range(2))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            values = model.predict(rgb, query).cpu().numpy()
        prediction.extend(values)
        for i, end, phase in points:
            ep = episodes[i]
            truth.append(ep['xyz'][end]); hidden.append(ep['hidden'][end]); phase_names.append(phase)
            records.append(dict(path=ep['row']['path'], frame=int(ep['raw_index'][end]), phase=phase,
                                query_frame=0, query_uv=ep['query_uv'].tolist()))
    prediction, truth = np.array(prediction), np.array(truth)
    hidden, phase_names = np.array(hidden), np.array(phase_names)
    return summary(prediction, truth, hidden, phase_names), dict(prediction_base_m=prediction,
            truth_base_m=truth, hidden=hidden, phase=phase_names), records


def causality(model, episode):
    model.eval(); model.requires_grad_(False)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    end = next(end for _, end, phase in evaluation_plan([episode]) if phase == 'exposed')
    rgb, query, _ = point_clip(episode, end)
    rgb = (rgb.cuda().float() / 255).requires_grad_(); query = query.cuda()
    prediction = model(rgb, query)
    gradient, = torch.autograd.grad(prediction[:, 14].float().sum(), rgb)
    future_gradient = float(gradient[:, :, 15:].abs().max())
    changed = rgb.detach().clone(); changed[:, :, 15:] = 1 - changed[:, :, 15:]
    with torch.no_grad():
        other = model(changed, query)
        prefix = model(rgb.detach()[:, :, :15], query[:, :, :15])
    value_change = float((prediction[:, :15].detach() - other[:, :15]).abs().max())
    prefix_change = float((prediction[:, :15].detach() - prefix).abs().max())
    assert future_gradient == 0 and value_change == 0
    torch.testing.assert_close(prediction[:, :15].detach(), prefix, rtol=2e-5, atol=2e-5)
    torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return dict(future_rgb_gradient_max=future_gradient, future_rgb_value_change_max=value_change,
                prefix_extension_change_max=prefix_change, precision='FP32, TF32 disabled',
                phase='exposed', episode=episode['row']['path'])


@torch.no_grad()
def query_binding(model, episodes):
    model.eval()
    assert len({ep['row']['context_path'] for ep in episodes}) == 1
    results = {}
    for name, end in (('initial', 0), ('post_shuffle', len(episodes[0]['context']['time']) - 1)):
        clips = [point_clip(ep, end) for ep in episodes]
        assert all(torch.equal(clips[0][0], clip[0]) for clip in clips)
        rgb, query = (torch.cat([clip[j] for clip in clips]).cuda() for j in range(2))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model.predict(rgb, query)
        truth = torch.tensor(np.stack([ep['xyz'][end] for ep in episodes]), device='cuda')
        results[name] = dict(same_rgb=True, query_uv=[ep['query_uv'].tolist() for ep in episodes],
            prediction_base_m=prediction.cpu().tolist(), truth_base_m=truth.cpu().tolist(),
            mean_error_mm=float((prediction - truth).norm(dim=-1).mean() * 1000),
            prediction_pairwise_distance_mm=float(torch.pdist(prediction).mean() * 1000),
            truth_pairwise_distance_mm=float(torch.pdist(truth).mean() * 1000))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--local-query', action='store_true', help='match the frame0 3x3-patch appearance reference')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    root = Path('/workspace/robotics_assets/datasets/shuffle_pick_multiview')
    weights = Path('/workspace/robotics_assets/pretrained/vision')
    experiment = 'point_tracking_local' if args.local_query else 'point_tracking'
    readme = ('# Point-query XYZ training\n\nStatus: running.\n'
              'Full finetune, native causal TCOW, initial2Ddot, XYZ-only loss. '
              'Actual data, model, batch, LR, seed, normalization and splits: config.json/subset.json. '
              'Opening selects best; final evaluation covers hidden/lift and held-out groups.\n')
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    (out / 'README.md').write_text(readme)
    shutil.copyfile(weights / 'tracking_query.json', out / 'tcow_provenance.json')
    torch.set_num_threads(4); torch.manual_seed(100)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(100)
    splits = subset(root)
    write_json(out / 'subset.json', {k: [r['path'] for r in v] for k, v in splits.items()})
    model = PointXYZ(local_query=args.local_query)
    load_tcow(model, weights / 'tracking_backbone.pt', weights / 'tracking_query.pt')
    config = dict(seed=100, steps=args.steps, data=str(root), model_config=model.config,
        initialization='official TCOW pretrained RGB+query backbone; random XYZ head',
        weights=[str(weights / 'tracking_backbone.pt'), str(weights / 'tracking_query.pt')],
        inputs='top RGB only + radius3 binary dot at frame0; no object ID, GT XYZ or masks',
        query_annotation='project initial context-frame object center offline; only UV raster enters model',
        appearance_reference=('frame0 3x3 patches around the dot; shared LN+128D projection; '
                              '5*cosine similarity added to spatial scores' if args.local_query else 'none'),
        mask_use='top-view occlusion fractions for evaluation metrics only; no training mask files',
        frames=30, frame_stride=3, image_preprocessing='center crop source320x320 to240x320, rows40:280',
        batch=4, micro_batch=2,
        sampler='same context x4 target dots; common past endpoint in context, separate expert endpoints',
        phase_probabilities=[.15, .35, .2, .3], loss='normalized XYZ: 0.5 endpoint +0.5 prefix MSE',
        frozen_parameters=0, backbone_lr=1e-5, head_lr=3e-4, warmup=100,
        optimizer='AdamW', weight_decay=1e-4, gradient_clip=1., precision='BF16',
        schedule='cosine to 0.1 floor', validation_interval=250,
        selection='minimum validation opening error at t_obj, before grasp',
        episode_counts={k: len(v) for k, v in splits.items()},
        code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path('OATFlow/tracking').glob('*.py')})
    assert sum(p.numel() for p in model.parameters() if not p.requires_grad) == 0
    config['trainable_parameters'] = sum(p.numel() for p in model.parameters())
    write_json(out / 'config.json', config)
    write_json(out / 'progress.json', dict(status='loading top RGB', step=0, expected=args.steps))
    train, segments = load_points(root, splits['train'])
    val, segments = load_points(root, splits['val'], segments=segments, evaluation_labels=True)
    labels = np.concatenate([ep['xyz'] for ep in train])
    model.set_xyz_statistics(labels.mean(0), labels.std(0).clip(.01))
    write_json(out / 'normalization.json', dict(mean_base_m=model.xyz_mean.tolist(), std_m=model.xyz_std.tolist(),
                                              computed_on='train sampled frames only', std_floor_m=.01))
    write_json(out / 'cache.json', dict(storage='host RAM only', views=['top'], segments=len(segments),
                                       rgb_bytes=sum(segment['rgb'].numel() for segment in segments.values())))
    write_json(out / 'queries.json', {ep['row']['path']: ep['query_uv'].tolist() for ep in train + val})
    groups = {}
    for i, ep in enumerate(train):
        groups.setdefault(ep['row']['context_path'], []).append(i)
    groups = list(groups.values())
    assert all([train[i]['row']['target_id'] for i in group] == list(range(4)) for group in groups)
    model = model.cuda()
    optimizer = torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=1e-5),
        dict(params=[p for n, p in model.named_parameters() if not n.startswith('backbone.')], lr=3e-4)],
        weight_decay=1e-4)
    initial, _, _ = evaluate(model, val, 'Initial opening', phases=('exposed',))
    write_json(out / 'initial_val.json', initial)
    best = initial['all']['mean_mm']; save(out / 'best.pt', model, None, 0, config, best)
    start = time.monotonic(); losses = []
    for step in tqdm(range(1, args.steps + 1), desc=f'{experiment} XYZ updates', mininterval=5):
        scale = min(step / 100, 1) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        for group, lr in zip(optimizer.param_groups, (1e-5, 3e-4)):
            group['lr'] = lr * scale
        selected = groups[int(rng.integers(len(groups)))]
        category = int(rng.choice(4, p=[.15, .35, .2, .3]))
        common_end = int(rng.choice(train[selected[0]]['categories'][category])) if category < 3 else None
        batches = [point_clip(train[i], common_end if category < 3 else int(rng.choice(train[i]['categories'][3])))
                   for i in selected]
        model.train(); optimizer.zero_grad(set_to_none=True); update_loss = 0.
        for micro in (batches[:2], batches[2:]):
            rgb, query, xyz = (torch.cat([batch[j] for batch in micro]).cuda() for j in range(3))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(rgb, query)
            error = (prediction.float() - model.normalize_xyz(xyz)).square()
            loss = .5 * error[:, -1].mean() + .5 * error.mean()
            assert torch.isfinite(loss)
            (loss / 2).backward(); update_loss += float(loss.detach()) / 2
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); losses.append(update_loss)
        if step % 10 == 0:
            elapsed = time.monotonic() - start
            write_json(out / 'progress.json', dict(status='training', step=step, expected=args.steps,
                loss=float(np.mean(losses[-10:])), best_open_mm=best, elapsed_s=elapsed,
                eta_minutes=(args.steps - step) * elapsed / step / 60))
        if step % 250 == 0 or step == args.steps:
            metric, _, _ = evaluate(model, val, f'Opening validation {step}', phases=('exposed',))
            score = metric['all']['mean_mm']
            config['numpy_rng'] = rng.bit_generator.state
            if score < best:
                best = score; save(out / 'best.pt', model, None, step, config, best)
            save(out / 'latest.pt', model, None, step, config, best)
            with (out / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step, loss=float(np.mean(losses[-250:])), validation=metric)) + '\n')
            print(json.dumps(dict(step=step, val_open_mm=score, best_open_mm=best)), flush=True)
    # Keep one final optimizer state separately; do not duplicate it in every checkpoint.
    torch.save(dict(optimizer=optimizer.state_dict(), step=args.steps, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), numpy_rng=rng.bit_generator.state), out / 'optimizer.pt')
    del optimizer; model.zero_grad(set_to_none=True)
    checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(checkpoint['model']); best_step = checkpoint['step']; del checkpoint
    write_json(out / 'progress.json', dict(status='testing', step=args.steps, best_step=best_step))
    write_json(out / 'causality.json', causality(model, val[0]))
    write_json(out / 'query_binding.json', query_binding(model, val[:4]))
    results = {}
    metric, predictions, records = evaluate(model, val, 'Validation all phases')
    results['val'] = metric
    np.savez_compressed(out / 'val_predictions.npz', **predictions); write_json(out / 'val_samples.json', records)
    video_cases = [val[0]]
    del train, val, segments
    segments = {}
    for group in ('composition', 'spatial', 'combined'):
        episodes, segments = load_points(root, splits[group], segments=segments, evaluation_labels=True)
        metric, predictions, records = evaluate(model, episodes, 'Test ' + group)
        results[group] = metric
        np.savez_compressed(out / f'{group}_predictions.npz', **predictions)
        write_json(out / f'{group}_samples.json', records)
        if group == 'combined':
            video_cases.append(episodes[0])
        write_json(out / 'test.json', dict(best_step=best_step, metrics=results))
        print(json.dumps(dict(group=group, opened=metric['by_phase']['exposed'], hidden=metric['hidden'])), flush=True)
    from OATFlow.tracking.point_video import videos
    videos(model, video_cases, root, out, best_step)
    write_json(out / 'progress.json', dict(status='complete', step=args.steps, best_step=best_step, best_open_mm=best))
    (out / 'README.md').write_text(readme.replace('Status: running.', 'Status: complete.') +
        f'\nUpdates:{args.steps}; best:{best_step}; opening validation:{best:.3f}mm. '
        'Results:test.json; settings:config.json; videos:test_videos/.\n')



if __name__ == '__main__':
    main()
