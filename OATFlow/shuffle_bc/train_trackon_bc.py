"""Joint context-decoder / reach-FM BC with frozen Track-On-R and side-camera ViT."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.shuffle_bc.context_model import FrozenWristViT
from OATFlow.shuffle_bc.data import DATA, splits
from OATFlow.shuffle_bc.trackon import FrozenTrackOn, CHECKPOINT, COMMIT
from OATFlow.shuffle_bc.trackon_model import TrackOnContext
from OATFlow.shuffle_bc.trackon_bc_data import allocate, batch, cache, grouped, schedule
from OATFlow.shuffle_bc.train_context_bc import losses
from OATFlow.shuffle_bc.train_context_xyz import metrics
from OATFlow.shuffle_bc.train_trackon_xyz import memory_bytes
from OATFlow.top_pick.data import SOURCE, VISION
from OATFlow.tracking.data import subset
from OATFlow.tracking.linear_probe import fingerprint
from OATFlow.tracking.utils import write_json

CONTEXT = Path('/workspace/robotics_assets/experiments/e110_trackon/train_2/best.pt')


@torch.no_grad()
def evaluate(model, data, size=64):
    model.eval(); n = data['xyz_count']; prediction = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, n, size):
            values = batch(data, slice(start, min(start + size, n)))
            xyz = model.xyz_from_features(*(values[k] for k in ('top', 'wrist', 'joints')))
            prediction.append((xyz * model.xyz_std + model.xyz_mean).cpu().numpy())
        metric, _ = metrics(np.concatenate(prediction), data['truth'][:n], data['records'])
        numerator = 0.; denominator = 0
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(109000)
            for start in range(n, len(data['top']), size):
                values = batch(data, slice(start, start + size))
                flow, _, _ = losses(model, values, slice(None), True)
                valid = int(values['mask'].sum()); numerator += float(flow) * valid; denominator += valid
    return dict(flow=numerator / denominator, xyz=metric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(); assert args.batch > 0 and args.epochs > 0
    assert not (args.preflight and args.resume)
    torch.set_num_threads(4); torch.manual_seed(111)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(111)
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e111_README.md', out / 'README.md')
    write_json(out / 'progress.json', dict(status='loading e110 context / reach FM'))
    train_rows = splits()['train']; val_rows = subset(DATA)['val']
    if args.preflight:
        train_rows = grouped(train_rows)[0]; val_rows = grouped(val_rows)[0]
    model = TrackOnContext()
    source = torch.load(CONTEXT, map_location='cpu', weights_only=False, mmap=True)
    assert source['epoch'] == 79 and source['step'] == 8374
    model.load_context_state(source['model'])
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in source['model'].items())
    source_frozen = source['config']['frozen_sha256']; del source
    reach = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
    assert reach['step'] == 115000
    assert all(torch.equal(v, reach['model']['flow.' + k]) for k, v in model.flow.state_dict().items())
    del reach
    initial_flow = fingerprint(model.flow)
    restored = torch.load(args.resume, map_location='cpu', mmap=True, weights_only=False) if args.resume else None
    if restored: assert restored['next_epoch'] <= args.epochs, 'Checkpoint already meets requested epoch budget'
    config = dict(status='initializing', seed=111, full_data=not args.preflight,
        data=str(DATA), train_episodes=len(train_rows), train_contexts=len(grouped(train_rows)),
        val_episodes=len(val_rows), context_checkpoint=str(CONTEXT), context_epoch=79, context_step=8374,
        context_checkpoint_sha256=hashlib.sha256(CONTEXT.read_bytes()).hexdigest(), context_initial_exact=True,
        FM_source=str(SOURCE), FM_source_step=115000, FM_initial_exact=True,
        FM_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), initial_FM_sha256=initial_flow,
        tracker='official Track-On-R', tracker_commit=COMMIT, tracker_checkpoint=str(CHECKPOINT),
        tracker_checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        vision_source=str(VISION), vision_source_sha256=hashlib.sha256(VISION.read_bytes()).hexdigest(),
        frozen_sha256=source_frozen, frozen='entire Track-On/DINOv3 and side ViT-B16',
        trainable='top/side projections, view embeddings, context decoder, XYZ head, FM including state encoder',
        context_stride=10, expert_stride=3, tracker_updates='context stride10+terminal; expert stride3+terminal',
        validation_tracker_updates='expert stride3 plus exact fixed XYZ evaluation endpoints; no future features',
        context_tokens=66, context_width=960, actual_wrist_camera=False, surrogate_camera='left',
        support_grid=20, tracker_memory=72, gradient_checkpointing=False, gradient_accumulation=False,
        loss='normalized current XYZ MSE weight1 + expert-only masked FM velocity MSE weight1',
        XYZ_normalization='preserve e110 full-training XYZ statistics',
        action_representation='H25/25Hz fixed-current-anchor EE deltaXYZ3, relative rotation6D, absolute physical jaw1',
        action_normalization='preserve reach buffers; jaw=-1-2*physical/.0255',
        context_FM='all flow/state parameters frozen with grad=None and no AdamW update on masked context',
        optimizer='AdamW(.9,.95),weight_decay1e-4,clip1',
        learning_rates=dict(context=1e-4, xyz_head=1e-5, FM=2e-5), warmup_updates=100,
        schedule='constant after warmup', precision='BF16 train / FP16 native tracker', batch=args.batch,
        epochs=1 if args.preflight else args.epochs, cache='deduplicated side FP16 in RAM only; compact native queries on disk',
        validation='fixed XYZ phase endpoints +64expert chunks(seed105000); FM noise/time seed109000; best by FM loss',
        rollout_status='not run', test_status='not run', resume_checkpoint=str(args.resume) if args.resume else None,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path('OATFlow/shuffle_bc/trackon_bc_data.py'), Path('OATFlow/shuffle_bc/trackon_model.py'),
            Path('OATFlow/shuffle_bc/trackon.py'), Path('OATFlow/shuffle_bc/train_context_bc.py'),
            Path('OATFlow/shuffle_bc/train_trackon_xyz.py'), Path('OATFlow/shuffle_bc/context_cache.py'),
            Path('OATFlow/shuffle_bc/context_model.py'), Path('OATFlow/reach/flow.py'), Path('OATFlow/policy/model.py'))})
    for name in config['source_sha256']:
        source_path = Path(name).resolve()
        dest = out / 'source' / source_path.relative_to(Path.cwd())
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source_path, dest)
    write_json(out / 'config.json', config)
    train = allocate(train_rows); val = allocate(val_rows, validation=True)
    nc = sum(not p[2] for p in train['points']); ne = len(train['points']) - nc
    if not args.preflight: assert (nc, ne) == (61688, 234434) and val['xyz_count'] == 184
    config.update(status='caching', context_points=nc, expert_points=ne, points_per_epoch=nc + ne,
                  val_XYZ_points=val['xyz_count'], val_BC_chunks=64,
                  train_cache_gib=memory_bytes(train)/2**30, val_cache_gib=memory_bytes(val)/2**30)
    write_json(out / 'config.json', config); print(json.dumps(config), flush=True)
    encoder = FrozenWristViT().cuda(); assert fingerprint(encoder) == source_frozen['wrist']
    tracker = None if restored else FrozenTrackOn().cuda()
    if tracker is not None: assert fingerprint(tracker) == source_frozen['tracker']
    cache_source = args.resume.parent if restored else None
    cache_seconds = cache(tracker, encoder, train, train_rows, out, 'train', source=cache_source)
    cache_seconds += cache(tracker, encoder, val, val_rows, out, 'val', source=cache_source)
    assert fingerprint(encoder) == source_frozen['wrist']
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    if tracker is not None:
        assert fingerprint(tracker) == source_frozen['tracker']
        assert all(not p.requires_grad and p.grad is None for p in tracker.parameters())
    if restored: (out / 'query_cache').symlink_to((args.resume.parent / 'query_cache').resolve())
    del tracker, encoder; torch.cuda.empty_cache()
    model.cuda(); model.flow.requires_grad_(True)
    params = list(model.named_parameters())
    optimizer = torch.optim.AdamW([
        dict(params=[p for n, p in params if not n.startswith(('flow.', 'xyz_head.'))], lr=1e-4),
        dict(params=[p for n, p in params if n.startswith('xyz_head.')], lr=1e-5),
        dict(params=[p for n, p in params if n.startswith('flow.')], lr=2e-5)],
        betas=(.9, .95), weight_decay=1e-4, fused=True)
    step = 0; samples_seen = 0; start_epoch = 1; next_batch = 0; order = None
    best = float('inf'); best_epoch = 0
    if restored:
        previous = restored['config']
        for key in ('batch', 'points_per_epoch', 'source_sha256', 'frozen_sha256', 'context_checkpoint_sha256'):
            assert previous[key] == config[key], key
        model.load_state_dict(restored['model'], strict=True); optimizer.load_state_dict(restored['optimizer'])
        step = restored['step']; samples_seen = restored['samples_seen']; start_epoch = restored['epoch']
        start_epoch = restored['next_epoch']; next_batch = restored['next_batch']
        order = restored['order']; best = restored['best_flow']
        best_epoch = restored['best_epoch']; rng.bit_generator.state = restored['numpy_rng']
        torch.set_rng_state(restored['torch_rng']); torch.cuda.set_rng_state(restored['cuda_rng'])
        shutil.copy2(args.resume.parent / 'best.pt', out / 'best.pt'); del restored
    epoch_updates = len(schedule(train, train_rows, range(len(grouped(train_rows))), args.batch))
    config.update(status='training', updates_per_epoch=epoch_updates, total_steps=epoch_updates*config['epochs'],
                  trainable_parameters=sum(p.numel() for _, p in params), cache_seconds=cache_seconds,
                  xyz_mean=model.xyz_mean.tolist(), xyz_std=model.xyz_std.tolist())
    write_json(out / 'config.json', config)
    initial = evaluate(model, val); write_json(out / 'initial_validation.json', initial)
    if not args.resume: best = initial['flow']
    print(json.dumps(dict(initial_validation=initial)), flush=True)
    checks = dict(frozen_vision_unchanged=True, context_initial_exact=True, reach_initial_exact=True)
    timings = {False: [], True: []}; started = time.perf_counter(); last_active = None

    def save(name, epoch, cursor, resumable=False):
        payload = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, config=config,
                       epoch=epoch, step=step, samples_seen=samples_seen, best_flow=best, best_epoch=best_epoch)
        if resumable:
            payload.update(optimizer=optimizer.state_dict(), next_epoch=epoch + int(order is None),
                next_batch=0 if order is None else cursor, order=order,
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        temporary = out / (name + '.tmp'); torch.save(payload, temporary); temporary.replace(out / (name + '.pt'))

    if not args.resume and not args.preflight: save('best', 0, 0)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, config['epochs'] + 1):
        if order is None: order = rng.permutation(len(grouped(train_rows)))
        batches = schedule(train, train_rows, order, args.batch)
        if args.preflight:
            a = [b for b in batches if not b[1]]; b = [b for b in batches if b[1]]
            full_a = (a[0][0], False, np.resize(a[0][2], args.batch).tolist())
            full_b = (b[0][0], True, np.resize(b[0][2], args.batch).tolist())
            batches = [full_a, a[-1], full_b, b[-1], full_b, b[-1], full_a]
        phase_updates = {active: sum(a == active for _, a, _ in batches) for active in (False, True)}
        epoch_started = time.perf_counter(); seen = 0
        progress = tqdm(range(next_batch, len(batches)), desc=f'e111 epoch{epoch}/{config["epochs"]}', mininterval=5)
        for cursor in progress:
            group, active, ids = batches[cursor]
            tick = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True); step += 1
            for pg, lr in zip(optimizer.param_groups, (1e-4, 1e-5, 2e-5)): pg['lr'] = lr*min(step/100, 1.)
            values = batch(train, ids)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                flow, auxiliary, error = losses(model, values, slice(None), active); loss = flow + auxiliary
            assert torch.isfinite(loss)
            if active and not checks.get('action_gradient_reaches_context'):
                probes = torch.autograd.grad(flow, (model.top_projection[1].weight,
                    model.wrist_projection[1].weight, model.flow.proprio[0].weight,
                    model.context_decoder.layers[0].self_attn.q_proj.weight, model.xyz_head[-1].weight),
                    retain_graph=True, allow_unused=True)
                assert all(p is not None and torch.isfinite(p).all() and p.abs().sum() > 0 for p in probes[:-1])
                assert probes[-1] is None
                checks.update(action_gradient_reaches_context=True, action_gradient_reaches_both_views_and_state=True,
                              XYZ_head_outside_action_path=True)
            loss.backward()
            if args.preflight or step == 1:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in params if p.requires_grad)
            before = None
            if not active:
                assert not flow.requires_grad and float(flow) == 0
                assert all(p.grad is None for p in model.flow.parameters())
                checks['masked_context_FM_grad_none'] = True
                if args.preflight: before = fingerprint(model.flow)
            grad = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1., error_if_nonfinite=True)
            optimizer.step()
            if before is not None:
                assert fingerprint(model.flow) == before
                if timings[True]: checks['masked_context_FM_unchanged_after_momentum'] = True
            torch.cuda.synchronize(); seconds = time.perf_counter() - tick; timings[active].append(seconds)
            seen += len(ids); samples_seen += len(ids)
            record = dict(status='training', epoch=epoch, step=step, epoch_batch=cursor+1, epoch_batches=len(batches),
                group=group, phase='expert_FM_XYZ' if active else 'context_XYZ', batch=len(ids),
                loss=float(loss.detach()), flow=float(flow.detach()), xyz=float(auxiliary.detach()),
                xyz_mm=float(error), grad_norm=float(grad), batch_seconds=seconds, samples_seen=samples_seen,
                train_passes=samples_seen/len(train['top']), elapsed_s=time.perf_counter()-started,
                gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
            with (out/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
            if active != last_active or step % 10 == 0:
                write_json(out/'progress.json', record); print(json.dumps(record), flush=True)
            last_active = active
            progress.set_postfix(flow=f'{float(flow):.4f}', xyz=f'{float(auxiliary):.4f}', refresh=False)
            if not args.preflight and step == 1:
                save('latest', epoch, cursor+1, True)
            if not args.preflight and step % 100 == 0:
                metric = evaluate(model, val)
                if metric['flow'] < best:
                    best = metric['flow']; best_epoch = epoch; save('best', epoch, cursor+1)
                write_json(out/'last_validation.json', dict(epoch=epoch, step=step, metrics=metric))
                with (out/'validation.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(epoch=epoch, step=step, metrics=metric))+'\n')
                print(json.dumps(dict(step=step, validation=metric)), flush=True)
                save('latest', epoch, cursor+1, True)
            if timings[False] and timings[True]:
                medians = {active: float(np.median(timings[active][-100:])) for active in (False, True)}
                estimated = sum(phase_updates[a]*medians[a] for a in (False, True))
                write_json(out/'timing.json', dict(epoch=epoch, step=step, context_update_s=medians[False],
                    expert_update_s=medians[True], optimization_epoch_s=estimated,
                    cache_seconds=cache_seconds, excludes_validation_and_checkpoint_IO=True))
        next_batch = 0; order = None
        metric = evaluate(model, val)
        if metric['flow'] < best:
            best = metric['flow']; best_epoch = epoch
            if not args.preflight: save('best', epoch, len(batches))
        if not args.preflight: save('latest', epoch, len(batches), True)
        result = dict(epoch=epoch, step=step, seconds=time.perf_counter()-epoch_started,
                      metrics=metric, best_flow=best, best_epoch=best_epoch)
        write_json(out/'last_validation.json', result)
        with (out/'validation.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
        print(json.dumps(result), flush=True)
        if args.preflight:
            model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                values = batch(val, slice(0, 1))
                action = model.sample_features(*(values[k] for k in ('top', 'wrist', 'joints')),
                    model.flow.sample_noise(1, 'cuda'), steps=2)
            assert action.shape == (1, 25, 10) and torch.isfinite(action).all()
            assert fingerprint(model.flow) != initial_flow
            checks.update(finite_H25_action=True, FM_updated_on_expert=True)
            save('latest', epoch, len(batches), True)
            saved = torch.load(out/'latest.pt', map_location='cpu', mmap=True, weights_only=False)
            assert saved['model'].keys() == model.state_dict().keys()
            assert all(torch.equal(v.cpu(), saved['model'][k]) for k, v in model.state_dict().items())
            assert len(saved['optimizer']['state']) == len(params)
            assert saved['next_epoch'] == epoch + 1 and saved['next_batch'] == 0 and saved['order'] is None
            checks.update(checkpoint_weights_exact=True, optimizer_states_saved=len(params),
                          epoch_boundary_resume_cursor_correct=True)
            del saved
        elif epoch == start_epoch and not args.resume: assert seen == len(train['top'])
        write_json(out/'gradient_checks.json', checks)
    if not args.preflight:
        assert samples_seen == args.epochs*len(train['top'])
        save('final', args.epochs, len(batches))
    result.update(status='preflight_passed' if args.preflight else 'complete', batch=args.batch,
                  gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30, samples_seen=samples_seen,
                  training_seconds=time.perf_counter()-started)
    config['status'] = result['status']; write_json(out/'config.json', config)
    write_json(out/'summary.json', result); write_json(out/'progress.json', result)
    with (out/'README.md').open('a') as stream: stream.write('\nActual result: '+json.dumps(result)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
