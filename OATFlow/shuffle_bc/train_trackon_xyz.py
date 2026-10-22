"""Full-data stride-ten XYZ context training with frozen online Track-On-R and side ViT."""
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

from OATFlow.shuffle_bc.context_cache import read_left
from OATFlow.shuffle_bc.context_model import FrozenWristViT
from OATFlow.shuffle_bc.data import DATA, splits
from OATFlow.shuffle_bc.trackon import FrozenTrackOn, CHECKPOINT, REPOSITORY, COMMIT
from OATFlow.shuffle_bc.trackon_model import TrackOnContext
from OATFlow.shuffle_bc.train_context_xyz import metrics, target_probe
from OATFlow.top_pick.data import SOURCE, VISION
from OATFlow.tracking.data import evaluation_plan, read_segment, subset
from OATFlow.tracking.linear_probe import fingerprint
from OATFlow.tracking.point_data import top_camera, project
from OATFlow.tracking.utils import write_json

STRIDE = 10


def sampled_frames(count):
    return np.unique(np.r_[np.arange(0, count, STRIDE), count - 1])


def grouped(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['context_path'], []).append(row)
    return [sorted(values, key=lambda r: r['target_id']) for _, values in sorted(groups.items())]


def validation_points(rows):
    episodes = []
    for row in rows:
        phases = []; indices = []; nc = 0
        for ref in (row['context_path'], row['path']):
            with np.load(DATA / ref / 'supervision.npz') as labels:
                count = len(labels['phase'])
                selected = np.unique(np.r_[np.arange(0, count, 10), count - 1])
                phases.append(labels['phase'][selected]); indices.append(selected)
            if ref == row['context_path']: nc = len(selected)
        episodes.append(dict(phase=np.concatenate(phases), context=dict(time=np.empty(nc)), row=row,
                             raw_index=np.r_[indices[0], row['context_frames'] + indices[1]]))
    result = {row['path']: {} for row in rows}
    for i, end, phase in evaluation_plan(episodes):
        frame = int(episodes[i]['raw_index'][end])
        assert frame not in result[rows[i]['path']]
        result[rows[i]['path']][frame] = phase
    return result


def allocate(rows, validation=False):
    points = validation_points(rows) if validation else {
        r['path']: np.r_[sampled_frames(r['context_frames']),
                         r['context_frames'] + sampled_frames(r['expert_frames'])] for r in rows}
    requests = {}; locations = {}; records = []; cursor = 0
    for row in rows:
        indices = np.array(list(points[row['path']]), dtype=np.int64)
        locations[row['path']] = dict(zip(indices.tolist(), range(cursor, cursor + len(indices))))
        for ref, selected in ((row['context_path'], indices[indices < row['context_frames']]),
                              (row['path'], indices[indices >= row['context_frames']] - row['context_frames'])):
            requests.setdefault(ref, set()).update(selected.tolist())
        if validation:
            records.extend(dict(episode=row['path'], raw_frame=int(i), phase=points[row['path']][i]) for i in indices)
        cursor += len(indices)
    offsets = {}; unique = 0
    for ref, indices in sorted(requests.items()):
        offsets[ref] = dict(zip(sorted(indices), range(unique, unique + len(indices))))
        unique += len(indices)
    data = dict(top=torch.empty(cursor, 256, dtype=torch.float16),
                wrist=torch.empty(unique, 64, 768, dtype=torch.float16),
                joints=torch.empty(cursor, 6), xyz=torch.empty(cursor, 3),
                side_index=torch.empty(cursor, dtype=torch.int64), records=records,
                locations=locations, requests=requests, offsets=offsets, filled=0, side_filled=0)
    return data


def memory_bytes(data):
    return sum(v.numel() * v.element_size() for v in data.values() if isinstance(v, torch.Tensor))


@torch.no_grad()
def encode_side(encoder, data, ref):
    frames = sorted(data['requests'][ref])
    if not frames: return None
    segment = read_left(ref, frames)
    for start in range(0, len(frames), 64):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            feature = encoder(segment['rgb'][start:start + 64].cuda()).half().cpu()
        assert torch.isfinite(feature).all()
        ids = [data['offsets'][ref][raw] for raw in frames[start:start + 64]]
        data['wrist'][ids] = feature
    data['side_filled'] += len(frames)
    return segment


def store(data, row, raw, hidden, segment, segment_raw):
    location = data['locations'][row['path']].get(raw)
    if location is None: return
    index = segment['index'][segment_raw]
    ref = row['context_path'] if raw < row['context_frames'] else row['path']
    data['top'][location] = hidden.half().cpu()
    data['joints'][location] = torch.from_numpy(segment['joints'][index])
    data['xyz'][location] = torch.from_numpy(segment['xyz'][index, row['target_id']])
    data['side_index'][location] = data['offsets'][ref][segment_raw]
    data['filled'] += 1


@torch.no_grad()
def cache(tracker, encoder, data, rows, out, name):
    camera = top_camera(DATA); groups = grouped(rows); started = time.perf_counter()
    total_frames = sum(len(sampled_frames(r[0]['context_frames'])) +
                       sum(len(sampled_frames(x['expert_frames'])) for x in r) for r in groups)
    frames_done = 0; queries_done = 0
    progress = tqdm(total=total_frames, desc=f'Track-On + ViT cache {name}', mininterval=5)
    for group_index, group in enumerate(groups):
        first = group[0]; ref = first['context_path']
        top = read_segment(DATA, ref, STRIDE, views=('top',), center_crop=True)
        side = encode_side(encoder, data, ref)
        states = []; query_features = [[] for _ in group]
        for index, raw in enumerate(top['raw_index']):
            # Each prompt has independent query/support memory. Only image encoding is shared.
            with torch.autocast('cuda', dtype=torch.float16):
                features = tracker.features(top['rgb'][0, index:index + 1].cuda())
                if raw == 0:
                    states = [tracker.initialize(features, torch.tensor(
                        project(top['xyz'][0, row['target_id']], camera), device='cuda', dtype=torch.float32))
                              for row in group]
                for q, (row, state) in enumerate(zip(group, states)):
                    hidden, position, visible = tracker.step(features, state)
                    assert torch.isfinite(hidden).all() and torch.isfinite(position).all()
                    store(data, row, raw, hidden, side, raw)
                    query_features[q].append(hidden.half().cpu())
            frames_done += 1; queries_done += len(group); progress.update()
            if raw % 100 == 0:
                elapsed = time.perf_counter() - started
                write_json(out / 'progress.json', dict(status='caching', split=name, context=ref,
                    groups_done=group_index, groups_total=len(groups), image_frames=frames_done,
                    image_frames_total=total_frames, query_updates=queries_done, points_cached=data['filled'],
                    elapsed_s=elapsed, fps=frames_done / elapsed,
                    eta_s=elapsed / frames_done * (total_frames - frames_done)))
        # Each expert continuation starts at its own end-of-context state, not another target's rollout.
        del top, side
        for q, (row, state) in enumerate(zip(group, states)):
            top = read_segment(DATA, row['path'], STRIDE, views=('top',), center_crop=True)
            side = encode_side(encoder, data, row['path'])
            for index, raw in enumerate(top['raw_index']):
                with torch.autocast('cuda', dtype=torch.float16):
                    hidden, position, visible = tracker.step(
                        tracker.features(top['rgb'][0, index:index + 1].cuda()), state)
                assert torch.isfinite(hidden).all() and torch.isfinite(position).all()
                store(data, row, row['context_frames'] + raw, hidden, side, raw)
                query_features[q].append(hidden.half().cpu())
                frames_done += 1; queries_done += 1; progress.update()
            destination = out / 'query_cache' / row['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez(destination.with_suffix('.npz'), features=torch.stack(query_features[q]).numpy(),
                     raw_index=np.r_[sampled_frames(row['context_frames']),
                                     row['context_frames'] + sampled_frames(row['expert_frames'])])
            del top, side
        elapsed = time.perf_counter() - started
        record = dict(status='caching', split=name, groups_done=group_index + 1, groups_total=len(groups),
            image_frames=frames_done, image_frames_total=total_frames, query_updates=queries_done,
            points_cached=data['filled'], elapsed_s=elapsed, fps=frames_done / elapsed,
            eta_s=elapsed / frames_done * (total_frames - frames_done),
            gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(out / 'progress.json', record); print(json.dumps(record), flush=True)
    progress.close()
    assert data['filled'] == len(data['top']) and data['side_filled'] == len(data['wrist'])
    data['truth'] = data['xyz'].numpy()
    np.save(out / f'{name}_xyz.npy', data['truth'])
    write_json(out / f'{name}_episodes.json', rows)
    if data['records']: write_json(out / f'{name}_points.json', data['records'])
    for key in ('locations', 'requests', 'offsets'): del data[key]


@torch.no_grad()
def restore_cache(encoder, data, rows, source, out, name):
    assert rows == json.loads((source / f'{name}_episodes.json').read_text())
    groups = grouped(rows); started = time.perf_counter()
    progress = tqdm(groups, desc=f'Reuse Track-On / encode side {name}', mininterval=5)
    for done, group in enumerate(progress, 1):
        context = encode_side(encoder, data, group[0]['context_path'])
        for row in group:
            expert = encode_side(encoder, data, row['path'])
            expected = np.r_[sampled_frames(row['context_frames']),
                             row['context_frames'] + sampled_frames(row['expert_frames'])]
            with np.load((source / 'query_cache' / row['path']).with_suffix('.npz')) as saved:
                np.testing.assert_array_equal(saved['raw_index'], expected)
                feature = saved['features']
                assert feature.shape == (len(expected), 256) and feature.dtype == np.float16
                assert np.isfinite(feature).all()
                for raw, hidden in zip(expected, feature):
                    before_expert = raw < row['context_frames']
                    segment = context if before_expert else expert
                    local = int(raw if before_expert else raw - row['context_frames'])
                    store(data, row, int(raw), torch.from_numpy(hidden), segment, local)
            del expert
        del context
        elapsed = time.perf_counter() - started
        record = dict(status='rebuilding side RAM cache', split=name, groups_done=done, groups_total=len(groups),
            points_cached=data['filled'], elapsed_s=elapsed, eta_s=elapsed / done * (len(groups) - done),
            tracking_recomputed=False, gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(out / 'progress.json', record)
        if done == 1 or done % 20 == 0: print(json.dumps(record), flush=True)
    assert data['filled'] == len(data['top']) and data['side_filled'] == len(data['wrist'])
    data['truth'] = data['xyz'].numpy()
    np.testing.assert_array_equal(data['truth'], np.load(source / f'{name}_xyz.npy'))
    for key in ('locations', 'requests', 'offsets'): del data[key]


def batch(data, ids):
    return [data['top'][ids].cuda(), data['wrist'][data['side_index'][ids]].cuda(), data['joints'][ids].cuda()]


@torch.inference_mode()
def evaluate(model, data, size=512):
    model.eval(); predictions = []
    for start in range(0, len(data['top']), size):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model.xyz_from_features(*batch(data, slice(start, start + size)))
        predictions.append((prediction * model.xyz_std + model.xyz_mean).cpu().numpy())
    return metrics(np.concatenate(predictions), data['truth'], data['records'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=1280)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--epochs', type=int, default=10, help='total epoch target, including completed epochs')
    args = parser.parse_args(); assert args.batch > 0 and args.epochs > 0
    assert not (args.resume and args.preflight)
    resumed = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True) if args.resume else None
    if resumed:
        assert 0 < resumed['epoch'] < args.epochs
        assert resumed['config']['batch'] == args.batch and resumed['config']['stride'] == STRIDE
    torch.set_num_threads(4); torch.manual_seed(110)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e110_resume_README.md' if resumed else 'docs/research/e110_README.md', out / 'README.md')
    train_rows = splits()['train']; val_rows = subset(DATA)['val']
    if args.preflight:
        train_rows = grouped(train_rows)[0]; val_rows = grouped(val_rows)[0]
    config = dict(status='initializing', seed=110, data=str(DATA), train_episodes=len(train_rows),
        val_episodes=len(val_rows), stride=STRIDE, epochs=args.epochs if not args.preflight else 1,
        preflight=args.preflight, batch=args.batch, tracker='Track-On-R, official latest', tracker_repo=str(REPOSITORY),
        tracker_commit=COMMIT, tracker_checkpoint=str(CHECKPOINT),
        tracker_checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        dinov3='facebook/dinov3-vits16plus-pretrain-lvd1689m', support_grid=20, memory_size=72,
        prompts='one frame0 target click + fixed support grid; independent memory for each prompt',
        frozen='entire Track-On-R including query decoder/adapter, ImageNet ViT-B/16, reach FM/state encoder',
        trainable='visual projections, view embeddings, four context blocks, XYZ head',
        context_tokens=66, context_width=960, actual_wrist_camera=False, surrogate_camera='left',
        loss='train-normalized current-frame XYZ MSE, weight1', action_training=False,
        FM_source=str(SOURCE), FM_source_step=115000, FM_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        vision_source=str(VISION), vision_source_sha256=hashlib.sha256(VISION.read_bytes()).hexdigest(), gradient_checkpointing=False,
        tracker_precision='FP16 autocast, FP32 pretrained weights', context_precision='BF16 autocast, FP32 weights/loss',
        cache='all unique side features CPU FP16; query features CPU FP16 and persisted per episode; stream raw RGB per segment',
        optimizer='AdamW(.9,.95),weight_decay1e-4,clip1', learning_rates=dict(context=1e-4, xyz_head=1e-5),
        validation='fixed184phase endpoints of32episodes sampled at stride10; tracker updates on stride10 frames plus segment terminals',
        test_status='not run', rollout_status='not run', resume_checkpoint=str(args.resume) if resumed else None,
        resume_epoch=resumed['epoch'] if resumed else 0,
        resume_step=resumed['step'] if resumed else 0,
        resume_lr_schedule='constant at checkpoint LR; original cosine already reached its floor' if resumed else None,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path('OATFlow/shuffle_bc/trackon.py'), Path('OATFlow/shuffle_bc/trackon_model.py'),
            Path('OATFlow/policy/model.py'), Path('OATFlow/shuffle_bc/context_model.py'))})
    for name in config['source_sha256']:
        source = Path(name).resolve(); destination = out / 'source' / source.relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
    shutil.copy2(REPOSITORY / 'config/test.yaml', out / 'source' / 'trackon_test.yaml')
    write_json(out / 'config.json', config)
    write_json(out / 'progress.json', dict(status='loading frozen features / ViT'))
    encoder = FrozenWristViT().cuda()
    if resumed:
        previous = resumed['config']
        for key in ('tracker_checkpoint_sha256', 'vision_source_sha256', 'FM_source_sha256'):
            assert config[key] == previous[key], key
        for name in ('OATFlow/shuffle_bc/trackon.py', 'OATFlow/shuffle_bc/trackon_model.py',
                     'OATFlow/policy/model.py', 'OATFlow/shuffle_bc/context_model.py'):
            assert config['source_sha256'][name] == previous['source_sha256'][name], name
        before = previous['frozen_sha256']
        assert fingerprint(encoder) == before['wrist']
        dinov3_revision = previous['dinov3_revision']
    else:
        tracker = FrozenTrackOn().cuda()
        before = dict(tracker=fingerprint(tracker), wrist=fingerprint(encoder))
        dinov3_revision = tracker.model.backbone.vit_encoder.dinov3.config._commit_hash
    train = allocate(train_rows); val = allocate(val_rows, validation=True)
    if not args.preflight:
        assert len(train['top']) == 134726 and len(val['top']) == 184
    config.update(status='caching', train_points=len(train['top']), val_points=len(val['top']),
        unique_train_side_frames=len(train['wrist']), train_cache_gib=memory_bytes(train) / 2**30,
        dinov3_revision=dinov3_revision,
        frozen_sha256=before)
    write_json(out / 'config.json', config); print(json.dumps(config), flush=True)
    if resumed:
        source = args.resume.parent
        restore_cache(encoder, train, train_rows, source, out, 'train')
        restore_cache(encoder, val, val_rows, source, out, 'val')
        assert fingerprint(encoder) == before['wrist']
        # Reference the original compact queries/labels so a later resume can reuse them too.
        for name in ('query_cache', 'train_xyz.npy', 'val_xyz.npy', 'train_episodes.json',
                     'val_episodes.json', 'val_points.json'):
            (out / name).symlink_to((source / name).resolve())
    else:
        cache(tracker, encoder, train, train_rows, out, 'train')
        cache(tracker, encoder, val, val_rows, out, 'val')
        assert before == dict(tracker=fingerprint(tracker), wrist=fingerprint(encoder))
        del tracker
    del encoder
    torch.cuda.empty_cache()
    model = TrackOnContext().cuda()
    model.xyz_mean.copy_(train['xyz'].mean(0).cuda())
    model.xyz_std.copy_(train['xyz'].std(0, correction=0).clamp_min(.01).cuda())
    if resumed:
        torch.testing.assert_close(model.xyz_mean.cpu(), resumed['model']['xyz_mean'], rtol=0, atol=0)
        torch.testing.assert_close(model.xyz_std.cpu(), resumed['model']['xyz_std'], rtol=0, atol=0)
        model.load_context_state(resumed['model'])
    frozen_flow = fingerprint(model.flow)
    if resumed: assert frozen_flow == resumed['config']['frozen_flow_sha256']
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert all(not name.startswith('flow.') for name, _ in parameters)
    optimizer = torch.optim.AdamW([
        dict(params=[p for name, p in parameters if not name.startswith('xyz_head.')], lr=1e-4),
        dict(params=[p for name, p in parameters if name.startswith('xyz_head.')], lr=1e-5)],
        betas=(.9, .95), weight_decay=1e-4, fused=True)
    if resumed:
        optimizer.load_state_dict(resumed['optimizer'])
        assert len(optimizer.state) == len(parameters)
        assert all(int(state['step']) == resumed['step'] for state in optimizer.state.values())
        resume_lrs = [group['lr'] for group in optimizer.param_groups]
    epoch_steps = math.ceil(len(train['top']) / args.batch)
    total_steps = 8 if args.preflight else args.epochs * epoch_steps
    config.update(status='training', updates_per_epoch=epoch_steps, total_steps=total_steps,
        trainable_parameters=sum(p.numel() for _, p in parameters), xyz_mean=model.xyz_mean.tolist(),
        xyz_std=model.xyz_std.tolist(), frozen_flow_sha256=frozen_flow,
        resumed_learning_rates=resume_lrs if resumed else None)
    write_json(out / 'config.json', config)
    initial, details = evaluate(model, val)
    write_json(out / 'initial_val.json', dict(metrics=initial, records=details))
    best = initial['macro_phase_mm']; best_epoch = 0; step = 0; samples_seen = 0
    rng = np.random.default_rng(110); started = time.perf_counter()
    start_epoch = 0; start_step = 0
    if resumed:
        start_epoch = resumed['epoch']; step = start_step = resumed['step']
        samples_seen = resumed['samples_seen']; best = resumed['best_val_mm']; best_epoch = resumed['best_epoch']
        assert step == start_epoch * epoch_steps and samples_seen == start_epoch * len(train['top'])
        reference = json.loads((args.resume.parent / 'last_validation.json').read_text())['validation']
        assert abs(initial['mean_mm'] - reference['mean_mm']) < .05
        rng.bit_generator.state = resumed['numpy_rng']
        torch.set_rng_state(resumed['torch_rng']); torch.cuda.set_rng_state(resumed['cuda_rng'])
        write_json(out / 'resume_checks.json', dict(epoch=start_epoch, step=step, samples_seen=samples_seen,
            optimizer_states=len(optimizer.state), optimizer_steps_restored=True,
            numpy_torch_cuda_rng_restored=True, frozen_hashes_match=True, xyz_normalization_exact=True,
            validation_mean_before_mm=reference['mean_mm'], validation_mean_restored_mm=initial['mean_mm'],
            tracking_recomputed=False, learning_rates=resume_lrs))
        del resumed
        resumed = True

    def save(name, epoch):
        payload = dict(model=model.context_state(), optimizer=optimizer.state_dict(), config=config,
            epoch=epoch, step=step, samples_seen=samples_seen, best_val_mm=best, best_epoch=best_epoch,
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        temporary = out / (name + '.tmp'); torch.save(payload, temporary); temporary.replace(out / (name + '.pt'))

    save('best', start_epoch)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch + 1, config['epochs'] + 1):
        model.train(); order = rng.permutation(len(train['top'])); loss_sum = 0.; seen = 0
        if args.preflight: order = np.resize(order, 8 * args.batch)
        progress = tqdm(range(0, len(order), args.batch), desc=f'Track-On context epoch {epoch}', mininterval=5)
        epoch_started = time.perf_counter()
        for start in progress:
            tick = time.perf_counter(); ids = torch.from_numpy(order[start:start + args.batch]); step += 1
            if not resumed:
                for group, base in zip(optimizer.param_groups, (1e-4, 1e-5)):
                    group['lr'] = base * min(step / 50, 1) * (.1 + .9 * (1 + math.cos(math.pi * (step - 1) / max(total_steps - 1, 1))) / 2)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model.xyz_from_features(*batch(train, ids))
            truth = train['xyz'][ids].cuda()
            loss = F.mse_loss(prediction, model.normalize_xyz(truth)); assert torch.isfinite(loss)
            loss.backward()
            if step == start_step + 1:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for _, p in parameters)
                assert all(p.grad is None for p in model.flow.parameters())
                write_json(out / 'gradient_checks.json', dict(all_trainable_finite_nonzero=True,
                    frozen_flow_gradients_none=True, xyz_shapes_shared_context=True, checkpointing=False))
            grad = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1., error_if_nonfinite=True)
            optimizer.step(); samples_seen += len(ids); seen += len(ids)
            mm = float(((prediction.detach() * model.xyz_std + model.xyz_mean - truth).norm(dim=-1).mean() * 1000))
            loss_value = float(loss.detach()); loss_sum += loss_value * len(ids)
            elapsed = time.perf_counter() - started
            record = dict(status='training', epoch=epoch, step=step, total_steps=total_steps, batch=len(ids),
                loss=loss_value, xyz_mm=mm, grad_norm=float(grad), samples_seen=samples_seen,
                train_passes=samples_seen / len(train['top']), batch_seconds=time.perf_counter() - tick,
                elapsed_s=elapsed, eta_s=elapsed / (step - start_step) * (total_steps - step),
                learning_rates=[group['lr'] for group in optimizer.param_groups],
                gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            with (out / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
            if step == start_step + 1 or step % 10 == 0:
                write_json(out / 'progress.json', record); print(json.dumps(record), flush=True)
            progress.set_postfix(loss=f'{loss_value:.4f}', mm=f'{mm:.1f}', refresh=False)
        assert args.preflight or seen == len(train['top'])
        metric, details = evaluate(model, val)
        if metric['macro_phase_mm'] < best:
            best = metric['macro_phase_mm']; best_epoch = epoch; save('best', epoch)
        save('latest', epoch)
        record = dict(epoch=epoch, step=step, train_loss=loss_sum / seen,
            seconds=time.perf_counter() - epoch_started, validation=metric, best_epoch=best_epoch)
        with (out / 'validation.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
        write_json(out / 'last_validation.json', dict(**record, records=details)); print(json.dumps(record), flush=True)
    assert fingerprint(model.flow) == frozen_flow
    if not args.preflight: assert samples_seen == config['epochs'] * len(train['top'])
    save('final', config['epochs'])
    saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=False, mmap=True)
    model.load_context_state(saved['model']); model.eval()
    metric, details = evaluate(model, val)
    write_json(out / 'best_validation.json', dict(metrics=metric, records=details))
    # The validation set is small: materialize aligned inputs only for the prompt-swap probe.
    probe_data = dict(top=val['top'].cuda(), wrist=val['wrist'][val['side_index']].cuda(),
                      joints=val['joints'].cuda(), xyz=val['xyz'].cuda(), records=val['records'])
    write_json(out / 'target_probe.json', target_probe(model, probe_data))
    result = dict(status='preflight_passed' if args.preflight else 'complete', steps=step, samples_seen=samples_seen,
        best_epoch=best_epoch, validation=metric, training_seconds=time.perf_counter() - started,
        frozen_weights_unchanged=True, gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        start_epoch=start_epoch, additional_epochs=config['epochs'] - start_epoch,
        additional_updates=step - start_step)
    config.update(status=result['status']); write_json(out / 'config.json', config)
    write_json(out / 'summary.json', result); write_json(out / 'progress.json', result)
    with (out / 'README.md').open('a') as stream: stream.write('\nActual result: ' + json.dumps(result) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
