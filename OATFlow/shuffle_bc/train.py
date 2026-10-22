"""Joint TCOW/reach-FM behavior cloning, with XYZ auxiliary supervision only."""
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

from OATFlow.shuffle_bc.data import DATA, ROOT, splits, load, batch, training_batches
from OATFlow.shuffle_bc.model import ShufflePolicy, WEIGHTS
from OATFlow.top_pick.data import SOURCE


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def evaluate(model, episodes, points):
    model.eval()
    sums = np.zeros(3)
    valid = sum(min(25, len(episodes[i]['ee']) - 1 - int(episodes[i]['raw_index'][end] - episodes[i]['row']['context_frames'])) for i, end, active in points)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(105000)
        for point in tqdm(points, desc='Fixed BC validation', mininterval=10, leave=False):
            inputs = batch(episodes, [point])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                flow, aux, mm = model.loss_parts(*inputs)
            sums += [float(flow) * int(inputs[4].sum()) / valid,
                     float(aux) / len(points), float(mm) / len(points)]
    return dict(val_flow=float(sums[0]), val_xyz=float(sums[1]), val_current_xyz_mm=float(sums[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--steps', type=int)
    budget.add_argument('--epochs', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if (args.steps is not None and args.steps < 1) or (args.epochs is not None and args.epochs < 1):
        parser.error('steps/epochs must be positive')
    torch.set_num_threads(4); torch.manual_seed(105)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(105)
    ROOT.mkdir(exist_ok=True)
    out = args.output or ROOT / ('preflight_6' if args.preflight else 'train_2')
    out.mkdir(exist_ok=False)
    shutil.copyfile(Path('docs/research/e105_README.md'), out / 'README.md')
    groups = splits()
    # Fixed validation subset; train uses every original train episode.
    val_rows = [r for r in groups['val'] if r['layout_id'] in (32, 33) and r['permutation'] in (0, 9, 16, 18)]
    train_rows = groups['train'][:4] if args.preflight else groups['train']
    train, segments, points = load(train_rows)
    val, segments, val_points = load(val_rows[:4] if args.preflight else val_rows, segments)
    val_points = [point for point in val_points if point[2]]
    val_rng = np.random.default_rng(105000)
    validation = [val_points[i] for i in np.sort(val_rng.choice(len(val_points), min(64, len(val_points)), replace=False))]
    model = ShufflePolicy()
    reach = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
    assert reach['step'] == 115000
    assert all(torch.equal(value, reach['model']['flow.' + name]) for name, value in model.flow.state_dict().items())
    tcow = torch.load(WEIGHTS / 'tracking_backbone.pt', map_location='cpu', weights_only=True, mmap=True)['backbone']
    query_weight = torch.load(WEIGHTS / 'tracking_query.pt', map_location='cpu', weights_only=True)['query_weight']
    assert all(torch.equal(value, torch.cat((tcow[name], query_weight), 1) if name == 'patch_embed.proj.weight' else tcow[name])
               for name, value in model.tracker.backbone.state_dict().items())
    del reach, tcow, query_weight
    labels = np.concatenate([ep['xyz'] for ep in train])
    model.tracker.set_xyz_statistics(labels.astype(np.float64).mean(0), labels.astype(np.float64).std(0).clip(.01))
    order = training_batches(train, points, rng)
    epoch_steps = len(order)
    if args.preflight:
        n = args.steps or 20
        context = [ids for ids in order if not points[ids[0]][2]]
        expert = [ids for ids in order if points[ids[0]][2]]
        order = [context[i % len(context)] for i in range(n//2)] + [expert[i % len(expert)] for i in range(n-n//2)]
        if n >= 4:
            # Also verify masking after the FM has nonzero AdamW momentum.
            order[-1] = context[0]
    config = dict(status='preflight' if args.preflight else 'training', seed=105, data=str(DATA),
        steps=len(order) if args.preflight else (epoch_steps*args.epochs if args.epochs else args.steps or epoch_steps),
        epochs=args.epochs, batch=4, micro_batch=4, gradient_accumulation=False,
        train_episodes=len(train), val_episodes=len(val), train_points=len(points), val_points=len(validation),
        context_points=sum(not p[2] for p in points), expert_points=sum(p[2] for p in points),
        epoch_updates=epoch_steps, context_updates=sum(not points[ids[0]][2] for ids in order),
        expert_updates=sum(points[ids[0]][2] for ids in order),
        sampling='shuffle shared contexts per epoch; within each context chronological stride10 TCOW-only then chronological stride3 joint expert, paired target queries',
        frames=30, context_endpoint_stride=10, expert_endpoint_stride=3,
        rgb_cache='context union(stride3,stride10,terminal); expert stride3+terminal',
        crop='320square rows40:280, native240x320',
        prefix='uniform raw-time frame0 through current, nearest available past RGB; no future RGB; prompt only frame0',
        inputs='top RGB history, initial radius3 UV dot, measured five arm joints and jaw',
        actions='H25 at25Hz: future measured fixed-anchor EE deltaXYZ3, relative rotation6D, absolute commanded jaw1',
        action_loss='masked normalized flow velocity MSE; context-only batches skip FM forward and all FM/goal optimizer updates',
        auxiliary='MSE normalized XYZ across all30 past/current frames, weight1',
        architecture='TCOW pooled770 -> LN/Linear/GELU hidden256 -> LN/Linear960 goal token; measured state token -> FM',
        xyz_head='Linear256to3 from shared tracker hidden, loss only; not fed to FM',
        mask_head=False, mask_files_loaded=False, target_id_input=False, ground_truth_xyz_input=False,
        pretrained=dict(tcow_rgb=str(WEIGHTS / 'tracking_backbone.pt'), tcow_query=str(WEIGHTS / 'tracking_query.pt'),
                        reach=str(SOURCE), reach_step=115000),
        initialization_checks=dict(all_TCOW_backbone_tensors_exact=True, all_reach_flow_tensors_exact=True),
        frozen='none', lrs=dict(backbone=1e-5, fm=2e-5, new_heads=1e-4), warmup=100,
        schedule='constant after warmup', optimizer='AdamW(.9,.95),wd1e-4,clip1',
        precision='BF16', gradient_checkpointing=False,
        normalization='reach state/action buffers preserved; XYZ from train cached frames only with10mm std floor',
        xyz_mean=model.tracker.xyz_mean.tolist(), xyz_std=model.tracker.xyz_std.tolist(),
        validation='fixed64 chunks, noise/time seed105000; select best by FM loss',
        test_status='planned full-policy K10 rollout after training; no expert cover opening',
        parameters=sum(p.numel() for p in model.parameters()),
        rgb_cache_bytes=sum(s['rgb'].numel() for s in segments.values()),
        source_sha256={str(p): sha(p) for p in list(Path('OATFlow/shuffle_bc').glob('*.py')) +
                       [Path('OATFlow/tracking/point.py'), Path('OATFlow/tracking/backbone.py')]},
        weights_sha256={str(p): sha(p) for p in (SOURCE, WEIGHTS / 'tracking_backbone.pt', WEIGHTS / 'tracking_query.pt')})
    write(out / 'config.json', config)
    write(out / 'samples.json', dict(train=[r['path'] for r in train_rows], val=[r['path'] for r in val_rows[:len(val)]],
                                    validation=[dict(path=val[i]['row']['path'], raw_frame=int(val[i]['raw_index'][end])) for i, end, active in validation]))
    model = model.cuda()
    optimizer = torch.optim.AdamW([
        dict(params=model.tracker.backbone.parameters(), lr=1e-5),
        dict(params=model.flow.parameters(), lr=2e-5),
        dict(params=[p for name, p in model.named_parameters() if not name.startswith(('tracker.backbone.', 'flow.'))], lr=1e-4)],
        betas=(.9, .95), weight_decay=1e-4, fused=True)
    best = math.inf; best_step = 0; samples_seen = 0
    cursor = 0
    started = time.perf_counter()

    def save(name, step, resumable=False):
        payload = dict(model=model.state_dict(), step=step, config=config, samples_seen=samples_seen,
                       best_val=best, best_step=best_step)
        if resumable:
            payload.update(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(), numpy_rng=rng.bit_generator.state, order=order, cursor=cursor)
        torch.save(payload, out / (name + '.pt'))

    progress = tqdm(range(1, config['steps'] + 1), desc='e105 TCOW + FM BC', mininterval=5)
    for step in progress:
        if cursor == len(order):
            order = training_batches(train, points, rng); cursor = 0
        ids = order[cursor]; cursor += 1; samples_seen += len(ids)
        selected = [points[i] for i in ids]
        active = selected[0][2]
        assert all(p[2] == active for p in selected)
        batch_started = time.perf_counter()
        x = batch(train, selected)
        assert bool(x[4].any()) == active
        model.train(); optimizer.zero_grad(set_to_none=True)
        for group, lr in zip(optimizer.param_groups, (1e-5, 2e-5, 1e-4)):
            group['lr'] = min(step / 100, 1.) * lr
        with torch.autocast('cuda', dtype=torch.bfloat16):
            flow, aux, mm = model.loss_parts(*x)
            loss = flow + aux
        assert torch.isfinite(loss)
        if active and 'gradient_checks' not in config:
            gradients = torch.autograd.grad(flow,
                (model.tracker.backbone.patch_embed.proj.weight, model.goal[1].weight,
                 model.tracker.xyz[3].weight), retain_graph=True, allow_unused=True)
            assert gradients[0][:, 3].abs().sum() > 0 and gradients[1].abs().sum() > 0
            assert gradients[2] is None
            config['gradient_checks'] = dict(action_loss_reaches_TCOW_query_channel=True,
                action_loss_reaches_goal_projection=True, XYZ_output_head_not_in_action_path=True)
            write(out / 'config.json', config)
        loss.backward()
        if not active:
            assert float(flow) == 0 and not flow.requires_grad
            assert all(p.grad is None for p in model.flow.parameters())
            assert all(p.grad is None for p in model.goal.parameters())
            config['context_mask_checks'] = dict(flow_loss_zero=True, FM_grad_None=True,
                                                goal_projection_grad_None=True, no_AdamW_update_for_FM_or_goal=True)
        assert model.tracker.backbone.patch_embed.proj.weight.grad[:, 3].abs().sum() > 0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        reference = None
        if args.preflight and not active and step in (1, config['steps']):
            reference = {name: p.detach().clone() for name, p in model.named_parameters()
                         if name.startswith(('flow.', 'goal.'))}
        optimizer.step()
        if reference is not None:
            assert all(torch.equal(p, reference[name]) for name, p in model.named_parameters() if name in reference)
            config['context_mask_checks']['FM_and_goal_weights_bit_exact_after_optimizer_step'] = True
            if step == config['steps']:
                config['context_mask_checks']['unchanged_even_after_expert_optimizer_momentum'] = True
            del reference
        record = dict(step=step, samples_seen=samples_seen, train_passes=samples_seen/len(points),
                      phase='expert_joint' if active else 'context_tcow', batch_size=len(ids),
                      flow=float(flow.detach()), xyz=float(aux.detach()), current_xyz_mm=float(mm.detach()),
                      grad_norm=float(norm), elapsed_s=time.perf_counter()-started,
                      batch_seconds=time.perf_counter()-batch_started,
                      gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
        with (out / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        write(out / 'progress.json', record)
        progress.set_postfix(phase=record['phase'], loss=f"{record['flow'] + record['xyz']:.4f}",
                             FM=f"{record['flow']:.4f}", XYZ=f"{record['xyz']:.4f}",
                             mm=f"{record['current_xyz_mm']:.1f}", refresh=False)
        if step == 1 or step % 250 == 0 or step == config['steps']:
            record.update(evaluate(model, val, validation[:4] if args.preflight else validation))
            if record['val_flow'] < best:
                best = record['val_flow']; best_step = step
                if not args.preflight: save('best', step)
            if not args.preflight: save('latest', step, True)
            with (out / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
    config.update(status='complete', elapsed_s=time.perf_counter()-started, samples_seen=samples_seen,
                  train_passes=samples_seen/len(points), best_val=best, best_step=best_step,
                  gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
    if args.epochs is not None and not args.preflight:
        assert samples_seen == args.epochs * len(points) and cursor == len(order)
    if not args.preflight:
        save('final', step)
    write(out / 'config.json', config)
    readme = out / 'README.md'
    readme.write_text(readme.read_text() + f'\nTraining status: complete; steps {step}, samples {samples_seen}, passes {samples_seen/len(points):.4f}. Test not yet run.\n')


if __name__ == '__main__':
    main()
