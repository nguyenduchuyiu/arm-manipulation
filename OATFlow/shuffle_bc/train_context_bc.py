"""Full-data frozen-vision shared-context/reach-FM behavior cloning."""
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
from OATFlow.shuffle_bc.context_model import ContextPolicy
from OATFlow.shuffle_bc.data import DATA, splits, load, clip
from OATFlow.shuffle_bc.train_context_xyz import metrics
from OATFlow.simulation.pose import rotation_6d
from OATFlow.top_pick.data import SOURCE
from OATFlow.tracking.data import evaluation_plan
from OATFlow.tracking.linear_probe import fingerprint
from OATFlow.tracking.utils import write_json

CONTEXT = Path('/workspace/robotics_assets/experiments/e108_context/train/best.pt')


def groups(rows):
    result = {}
    for row in rows:
        result.setdefault(row['context_path'], []).append(row)
    return list(result.values())


def counts(rows):
    context = sum(len(np.unique(np.r_[np.arange(0, r['context_frames'], 10), r['context_frames']-1])) for r in rows)
    expert = sum(len(range(0, r['expert_frames']-1, 3)) for r in rows)
    return context, expert


def raw_batch(episodes, points):
    clips = [clip(episodes[i], end) for i, end, _ in points]
    rgb, query, xyz = (torch.cat([c[j] for c in clips]).cuda() for j in range(3))
    actions = np.zeros((len(points), 25, 10), np.float32)
    masks = np.zeros((len(points), 25), bool)
    for b, (i, end, active) in enumerate(points):
        if active:
            ep = episodes[i]; raw = int(ep['raw_index'][end]-ep['row']['context_frames'])
            last = len(ep['ee'])-1; assert 0 <= raw < last
            indices = raw+np.arange(25); future = np.minimum(indices+1, last)
            actions[b, :, :3] = ep['ee'][future]-ep['ee'][raw]
            actions[b, :, 3:9] = rotation_6d(ep['rotation'][future]@ep['rotation'][raw].T)
            actions[b, :, 9] = ep['jaw'][np.minimum(indices, last-1)]
            masks[b] = indices < last
    return rgb, query, torch.from_numpy(actions).cuda(), torch.from_numpy(masks).cuda(), xyz


@torch.no_grad()
def cache_rows(model, rows, validation=False, microbatch=8):
    episodes, _, points = load(rows)
    records = None
    if validation:
        plan = evaluation_plan(episodes)
        expert = [p for p in points if p[2]]
        rng = np.random.default_rng(105000)
        chosen = [expert[j] for j in np.sort(rng.choice(len(expert), 64, replace=False))]
        points = [(i, end, False) for i, end, _ in plan] + chosen
        records = [dict(episode=rows[i]['path'], raw_frame=int(episodes[i]['raw_index'][end]), phase=phase)
                   for i, end, phase in plan]
    else:
        assert (sum(not p[2] for p in points), sum(p[2] for p in points)) == counts(rows)
    requests = {}; keys = []
    for i, end, _ in points:
        row = rows[i]; raw = int(episodes[i]['raw_index'][end])
        ref, frame = ((row['context_path'], raw) if raw < row['context_frames']
                      else (row['path'], raw-row['context_frames']))
        keys.append((ref, frame)); requests.setdefault(ref, set()).add(frame)
    side = {ref: read_left(ref, frames) for ref, frames in requests.items()}
    data = {key: [] for key in ('top', 'wrist', 'joints', 'actions', 'mask', 'xyz')}
    model.eval()
    for start in tqdm(range(0, len(points), microbatch), desc='Encode frozen group features', mininterval=10, leave=False):
        selected = points[start:start+microbatch]
        rgb, query, actions, mask, xyz = raw_batch(episodes, selected)
        current = keys[start:start+len(selected)]
        left = torch.stack([side[ref]['rgb'][side[ref]['index'][raw]] for ref, raw in current]).cuda()
        joints = torch.tensor(np.array([side[ref]['joints'][side[ref]['index'][raw]] for ref, raw in current]), device='cuda')
        truth = np.array([side[ref]['xyz'][side[ref]['index'][raw], rows[i]['target_id']]
                          for (ref, raw), (i, _, _) in zip(current, selected)])
        np.testing.assert_allclose(xyz[:, -1].cpu().numpy(), truth, rtol=0, atol=1e-7)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            top, wrist = model.encode(rgb, query, left)
        for key, value in zip(data, (top.half(), wrist.half(), joints, actions, mask, xyz[:, -1])):
            assert torch.isfinite(value).all(); data[key].append(value)
    data = {key: torch.cat(value) for key, value in data.items()}
    data['points'] = points; data['records'] = records
    if validation:
        data['xyz_count'] = len(plan)
    return data


def losses(model, data, ids, active):
    model.flow.requires_grad_(active)
    context, kv = model.context(*(data[k][ids] for k in ('top', 'wrist', 'joints')), with_kv=active)
    prediction = model.xyz_head(context).float()
    auxiliary = F.mse_loss(prediction, model.normalize_xyz(data['xyz'][ids]))
    flow = auxiliary.new_zeros(())
    if active:
        actions = data['actions'][ids]; mask = data['mask'][ids]
        values = torch.cat((actions[..., :9], -1-2*actions[..., 9:]/.0255), -1)
        noisy, times, target = model.flow.training_path(values)
        velocity = model.flow.velocity(noisy, times, kv)
        flow = ((velocity.float()-target.float()).square()*mask[..., None]).sum()/(mask.sum()*10)
    error = (prediction.detach()*model.xyz_std+model.xyz_mean-data['xyz'][ids]).norm(dim=-1).mean()*1000
    return flow, auxiliary, error


def phase_batches(data, size):
    result = []
    for active in (False, True):
        ids = sorted((i for i, p in enumerate(data['points']) if p[2] == active),
                     key=lambda i: (data['points'][i][1], data['points'][i][0]))
        result.extend((active, ids[start:start+size]) for start in range(0, len(ids), size))
    return result


@torch.no_grad()
def evaluate(model, data, size=32):
    model.eval(); n = data['xyz_count']; prediction = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, n, size):
            xyz = model.xyz_from_features(*(data[k][start:min(start+size, n)] for k in ('top', 'wrist', 'joints')))
            prediction.append((xyz*model.xyz_std+model.xyz_mean).cpu().numpy())
        value, _ = metrics(np.concatenate(prediction), data['xyz'][:n].cpu().numpy(), data['records'])
        numerator = 0.; denominator = 0
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(109000)
            for start in range(n, len(data['points']), size):
                ids = slice(start, start+size)
                flow, _, _ = losses(model, data, ids, True)
                valid = int(data['mask'][ids].sum()); numerator += float(flow)*valid; denominator += valid
    return dict(flow=numerator/denominator, xyz=value)


def state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()
            if not key.startswith(('tracker.', 'wrist.'))}


def load_state(model, values):
    assert values.keys() == state(model).keys()
    missing, extra = model.load_state_dict(values, strict=False)
    assert not extra and all(key.startswith(('tracker.', 'wrist.')) for key in missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(); assert args.batch > 0 and args.epochs > 0
    torch.set_num_threads(4); torch.manual_seed(109)
    torch.backends.cuda.matmul.allow_tf32 = True; rng = np.random.default_rng(109)
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e109_README.md', out/'README.md')
    write_json(out/'progress.json', dict(status='loading frozen vision / e108 context / reach FM'))
    split = splits(); train_rows = split['train'][:4] if args.preflight else split['train']
    val_rows = [r for r in split['val'] if r['layout_id'] in (32, 33) and r['permutation'] in (0, 9, 16, 18)]
    train_groups = groups(train_rows); nc, ne = counts(train_rows)
    epoch_updates = sum(math.ceil(c/args.batch)+math.ceil(e/args.batch) for c, e in map(counts, train_groups))
    model = ContextPolicy()
    saved = torch.load(CONTEXT, map_location='cpu', weights_only=False, mmap=True)
    assert saved['epoch'] == 27 and saved['step'] == 2457
    model.load_context_state(saved['model']); del saved
    reach = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
    assert reach['step'] == 115000
    assert all(torch.equal(v, reach['model']['flow.'+k]) for k, v in model.flow.state_dict().items()); del reach
    frozen = dict(tracker=fingerprint(model.tracker.backbone), wrist=fingerprint(model.wrist))
    initial_flow = fingerprint(model.flow)
    model.cuda(); model.flow.requires_grad_(True)
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        dict(params=[p for n, p in params if not n.startswith(('flow.', 'xyz_head.'))], lr=1e-4),
        dict(params=[p for n, p in params if n.startswith('xyz_head.')], lr=1e-5),
        dict(params=[p for n, p in params if n.startswith('flow.')], lr=2e-5)],
        betas=(.9, .95), weight_decay=1e-4, fused=True)
    config = dict(status='preflight' if args.preflight else 'training', seed=109, data=str(DATA),
        context_checkpoint=str(CONTEXT), context_checkpoint_sha256=hashlib.sha256(CONTEXT.read_bytes()).hexdigest(),
        FM_source=str(SOURCE), FM_source_step=115000, initial_FM_exact=True,
        frozen_sha256=frozen, initial_flow_sha256=initial_flow,
        train_episodes=len(train_rows), train_contexts=len(train_groups), context_points=nc, expert_points=ne,
        train_points=nc+ne, full_train=not args.preflight, batch=args.batch, epochs=args.epochs,
        epoch_updates=epoch_updates, total_updates=epoch_updates*args.epochs,
        context_stride=10, expert_stride=3, microbatch_encode=8, frames=30, val_episodes=len(val_rows),
        loss='current normalized XYZ MSE weight1 + expert-only masked FM velocity MSE weight1',
        learning_rates=dict(context=1e-4, xyz_head=1e-5, FM=2e-5), warmup=100, schedule='constant after warmup',
        gradient_checkpointing=False, gradient_accumulation=False, precision='BF16/TF32',
        optimizer='AdamW(.9,.95),wd1e-4,clip1', actual_wrist_camera=False, surrogate_camera='left',
        frozen='original TCOW / ImageNet ViT-B16; FM additionally frozen on context-only updates',
        cache='current shared context/four query demos only; GPU FP16 frozen tokens; discarded after group',
        action_representation='H25 EE deltaXYZ3/relativeRotation6D/absolute jaw1 from fixed current EE anchor',
        xyz_normalization='preserve e108 train-part-only statistics for warm-started XYZ head',
        validation='32episodes/184XYZpoints+64expert chunks; FM seed109000; best by FM loss',
        test_status='not run', source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path('OATFlow/shuffle_bc/context_model.py'), Path('OATFlow/shuffle_bc/data.py'),
             Path('OATFlow/shuffle_bc/context_cache.py'), Path('OATFlow/reach/flow.py'), Path('OATFlow/policy/model.py')]})
    write_json(out/'config.json', config)
    for name in config['source_sha256']:
        source = Path(name).resolve(); dest = out/'source'/source.relative_to(Path.cwd())
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
    (out/'source'/'README.md').write_text('Exact training sources; hashes and settings in ../config.json.\n')
    write_json(out/'samples.json', dict(train=[r['path'] for r in train_rows], validation=[r['path'] for r in val_rows]))
    print(json.dumps(config), flush=True)
    val = cache_rows(model, val_rows, validation=True)
    initial = evaluate(model, val); write_json(out/'initial_validation.json', initial)
    best = initial['flow']; step = 0; samples_seen = 0; start_epoch = 1; next_group = 0; order = None
    if args.resume:
        restored = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
        assert restored['config']['batch'] == args.batch and restored['config']['epochs'] == args.epochs
        assert restored['config']['train_points'] == nc+ne
        load_state(model, restored['model']); optimizer.load_state_dict(restored['optimizer'])
        step = restored['step']; samples_seen = restored['samples_seen']; best = restored['best_flow']
        start_epoch = restored['epoch']; next_group = restored['next_group']; order = restored['order']
        rng.bit_generator.state = restored['numpy_rng']; torch.set_rng_state(restored['torch_rng'])
        torch.cuda.set_rng_state(restored['cuda_rng']); shutil.copy2(args.resume.parent/'best.pt', out/'best.pt')
        if next_group == len(train_groups):
            start_epoch += 1; next_group = 0; order = None
        config['resume'] = str(args.resume); del restored
    started = time.perf_counter(); timings = {False: [], True: []}; encode_seconds = []; checks = {}; last_active = None

    def save(name, epoch, cursor, resumable=False):
        payload = dict(model=state(model), config=config, step=step, epoch=epoch, samples_seen=samples_seen, best_flow=best)
        if resumable:
            payload.update(optimizer=optimizer.state_dict(), next_group=cursor, order=order,
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        temp = out/(name+'.tmp'); torch.save(payload, temp); temp.replace(out/(name+'.pt'))

    if not args.preflight and not args.resume: save('best', 0, 0)
    for epoch in range(start_epoch, args.epochs+1):
        if order is None or next_group == len(train_groups):
            order = rng.permutation(len(train_groups)); next_group = 0
        progress = tqdm(range(next_group, len(order)), desc=f'e109 epoch{epoch}/{args.epochs} contexts', mininterval=10)
        for cursor in progress:
            rows = train_groups[order[cursor]]
            write_json(out/'progress.json', dict(status='encoding current group', epoch=epoch, group=cursor+1,
                groups=len(order), context=rows[0]['context_path'], step=step, elapsed_s=time.perf_counter()-started))
            tick = time.perf_counter(); data = cache_rows(model, rows); torch.cuda.synchronize()
            encoding = time.perf_counter()-tick; encode_seconds.append(encoding)
            schedule = phase_batches(data, args.batch)
            if args.preflight:
                a = [s for s in schedule if not s[0]]; b = [s for s in schedule if s[0]]
                schedule = [a[0], a[-1], b[0], b[-1], b[0], b[-1], a[0]]
            group_samples = 0
            for active, ids in tqdm(schedule, desc='Context XYZ / expert FM+XYZ', mininterval=10, leave=False):
                tick = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True); step += 1
                for pg, lr in zip(optimizer.param_groups, (1e-4, 1e-5, 2e-5)):
                    pg['lr'] = lr*min(step/100, 1.)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    flow, auxiliary, error = losses(model, data, ids, active); loss = flow+auxiliary
                assert torch.isfinite(loss)
                if active and not checks.get('action_gradient_reaches_context'):
                    probes = torch.autograd.grad(flow, (model.top_projection[1].weight,
                        model.context_decoder.layers[0].self_attn.q_proj.weight, model.xyz_head[-1].weight),
                        retain_graph=True, allow_unused=True)
                    assert all(p is not None and torch.isfinite(p).all() and p.abs().sum() > 0 for p in probes[:2])
                    assert probes[2] is None; checks['action_gradient_reaches_context'] = True
                    checks['XYZ_head_outside_action_path'] = True
                loss.backward()
                assert all(p.grad is None for module in (model.tracker, model.wrist) for p in module.parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in params if p.requires_grad)
                before = None
                if not active:
                    assert not flow.requires_grad and float(flow) == 0
                    assert all(p.grad is None for p in model.flow.parameters())
                    if args.preflight: before = fingerprint(model.flow)
                    checks['context_FM_grad_none'] = True
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1., error_if_nonfinite=True)
                optimizer.step()
                if before is not None:
                    assert fingerprint(model.flow) == before
                    checks['context_FM_unchanged_even_with_momentum'] = bool(timings[True])
                torch.cuda.synchronize(); seconds = time.perf_counter()-tick; timings[active].append(seconds)
                samples_seen += len(ids); group_samples += len(ids)
                record = dict(status='training', epoch=epoch, group=cursor+1, groups=len(order), step=step,
                    phase='expert_joint' if active else 'context_xyz', batch=len(ids), flow=float(flow.detach()),
                    xyz=float(auxiliary.detach()), current_xyz_mm=float(error), grad_norm=float(norm),
                    samples_seen=samples_seen, full_data_passes=samples_seen/(nc+ne), batch_seconds=seconds,
                    elapsed_s=time.perf_counter()-started, gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
                with (out/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
                if active != last_active or step % 10 == 0:
                    write_json(out/'progress.json', record); print(json.dumps(record), flush=True)
                last_active = active
            if not args.preflight: assert group_samples == sum(counts(rows))
            del data; optimizer.zero_grad(set_to_none=True)
            record = dict(epoch=epoch, completed_groups=cursor+1, step=step, encoding_seconds=encoding,
                median_context_update_s=float(np.median(timings[False])), median_expert_update_s=float(np.median(timings[True])),
                median_group_encoding_s=float(np.median(encode_seconds)),
                estimated_epoch_s=float(np.median(encode_seconds))*len(train_groups)+
                    sum(math.ceil(c/args.batch)*np.median(timings[False])+math.ceil(e/args.batch)*np.median(timings[True])
                        for c, e in map(counts, train_groups)))
            write_json(out/'timing.json', record); print(json.dumps(record), flush=True)
            if args.preflight or cursor == 0 or (cursor+1) % 10 == 0 or cursor+1 == len(order):
                metric = evaluate(model, val); print(json.dumps(dict(step=step, validation=metric)), flush=True)
                if metric['flow'] < best:
                    best = metric['flow']
                    if not args.preflight: save('best', epoch, cursor+1)
                if not args.preflight: save('latest', epoch, cursor+1, True)
                with (out/'validation.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=step, epoch=epoch, group=cursor+1, metrics=metric))+'\n')
                write_json(out/'gradient_checks.json', checks)
            if args.preflight: break
        next_group = len(order); order = None
        if args.preflight: break
    assert dict(tracker=fingerprint(model.tracker.backbone), wrist=fingerprint(model.wrist)) == frozen
    assert fingerprint(model.flow) != initial_flow
    if args.preflight:
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            actions = model.sample_features(*(val[k][:1] for k in ('top', 'wrist', 'joints')),
                model.flow.sample_noise(1, 'cuda'), steps=2)
        assert actions.shape == (1, 25, 10) and torch.isfinite(actions).all()
        checks.update(frozen_vision_unchanged=True, FM_updated_on_expert=True, finite_H25_action=True)
        write_json(out/'gradient_checks.json', checks)
    else:
        assert samples_seen == args.epochs*(nc+ne)
        save('final', args.epochs, len(train_groups))
    result = dict(status='preflight_passed' if args.preflight else 'complete', steps=step,
        samples_seen=samples_seen, elapsed_s=time.perf_counter()-started, batch=args.batch,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30, best_flow=best, timing=record)
    config['status'] = result['status']; write_json(out/'config.json', config)
    write_json(out/'summary.json', result); write_json(out/'progress.json', result)
    with (out/'README.md').open('a') as stream:
        stream.write(f'\nActual stage: {result["status"]}; batch{args.batch}; {step}updates; '
            f'{samples_seen}samples; peak{result["peak_gpu_gib"]:.2f}GiB. Exact settings/results: config.json,summary.json.\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
