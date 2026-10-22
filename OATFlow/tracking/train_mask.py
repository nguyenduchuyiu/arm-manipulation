"""Fine-tune pretrained TCOW for amodal target masks, one top camera only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.tracking.mask import TargetMaskTCOW, frame_iou, load_original, mask_loss
from OATFlow.tracking.mask_data import batch, clip, load, splits, train_plan
from OATFlow.tracking.data import evaluation_plan
from OATFlow.tracking.utils import write_json


DATA = Path('/workspace/robotics_assets/datasets/shuffle_pick_multiview')
WEIGHTS = Path('/workspace/robotics_assets/pretrained/vision')


@torch.no_grad()
def evaluate(model, episodes, batch_size, description):
    model.eval()
    records = []
    plan = evaluation_plan(episodes)
    for start in tqdm(range(0, len(plan), batch_size), desc=description, mininterval=5):
        selected = plan[start:start + batch_size]
        rgb, query, truth, _, _ = batch(episodes, [(i, end) for i, end, _ in selected])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(rgb, query)
        values = frame_iou(logits[:, -1], truth[:, -1]).cpu().tolist()
        for (i, end, phase), value in zip(selected, values):
            ep = episodes[i]
            records.append(dict(path=ep['row']['path'], raw_frame=int(ep['raw_index'][end]),
                                phase=phase, hidden=bool(ep['hidden'][end]), iou=value))
    def score(rows):
        return dict(count=len(rows), mean_iou=float(np.mean([r['iou'] for r in rows])) if rows else None)
    result = dict(all=score(records), hidden=score([r for r in records if r['hidden']]),
                  visible=score([r for r in records if not r['hidden']]),
                  by_phase={phase: score([r for r in records if r['phase'] == phase])
                            for phase in sorted({r['phase'] for r in records})})
    return result, records


def save(path, model, optimizer, epoch, step, config, best, rng):
    state = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                 model_config=model.config, epoch=epoch, step=step, config=config, best_val_iou=best)
    if optimizer is not None:
        state.update(optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def examples(model, episodes, out, name):
    from PIL import Image, ImageDraw
    from OATFlow.tracking.mask_data import clip
    rows = []
    model.eval()
    points = [point for point in evaluation_plan(episodes) if point[2] in ('post_shuffle', 'exposed', 'lift')][:9]
    for i, end, phase in points:
        values = clip(episodes[i], end)
        rgb, query, truth = (value[None].cuda() for value in values[:3])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(rgb, query)
        predicted = (logits[0, -1] > 0).cpu().numpy()
        target = truth[0, -1].cpu().numpy()
        pixels = rgb[0, :, -1].permute(1, 2, 0).cpu().numpy()
        panels = []
        for region, color in ((target, [40, 240, 70]), (predicted, [240, 50, 190])):
            overlay = pixels.astype(float).copy()
            overlay[region] = .5 * overlay[region] + .5 * np.array(color)
            panels.append(overlay.astype(np.uint8))
        image = Image.fromarray(np.concatenate(panels, axis=1))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 639, 20), fill='black')
        iou = float(frame_iou(logits[:, -1], truth[:, -1])[0])
        draw.text((6, 4), f'{episodes[i]["row"]["path"]} | {phase} | GT green / prediction pink | IoU {iou:.3f}', fill='white')
        rows.append(image)
    canvas = Image.new('RGB', (640 * 3, 240 * 3))
    for n, image in enumerate(rows):
        canvas.paste(image, ((n % 3) * 640, (n // 3) * 240))
    canvas.save(out / f'{name}_examples.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error('epochs and batch must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(114)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(114)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    readme = Path('docs/research/e114_mask_README.md')
    shutil.copyfile(readme, out / 'README.md')
    config = dict(experiment='e114_mask', status='preparing', seed=114,
        data=str(DATA), epochs=args.epochs, batch=args.batch, preflight=args.preflight,
        inputs='top RGB and initial visible target mask at frame0 only; no later GT mask inputs',
        target='amodal target mask, including behind covers and the robot',
        initialization='official TCOW RGB+query backbone AND native pretrained target-mask output rows',
        removed='occluder/container output rows and unused flag head; no XYZ/pooling/context decoder/FM',
        frozen_parameters=0, trainable='all TCOW backbone and target-mask readout parameters',
        clip_frames=30, image_size=[240, 320], rgb_cache_stride=3,
        preprocessing='crop rows40:280 from native320x320; RGB(.45,.225)',
        sampler='full train episodes once per epoch; random causal prefix endpoint; uniformly sample30past/current frames, frame0 retained',
        phase_probabilities=dict(reveal=.15, shuffle=.35, hold=.2, expert=.3),
        loss='original TCOW target branch: 0.2 weighted BCE +0.4 bootstrapped BCE +0.4 soft Jaccard',
        loss_weights=dict(occluded_frame=5, occluded_pixel=2, hard_negative=3, query_frame=.2),
        loss_implementation='class balance floor.05; bootstrap top fraction max(.15,1-8.5*progress); query frame downweight fixed for every batch member',
        optimizer='AdamW', backbone_lr=1e-5, head_lr=1e-4, weight_decay=1e-4,
        warmup_steps=50, schedule='constant after linear warmup', gradient_clip=.3,
        precision='BF16 autocast with FP32 parameters/optimizer', gradient_checkpointing=False,
        numpy_madvise_hugepage=bool(np._core.multiarray._get_madvise_hugepage()),
        cache='host RAM uint8 RGB and packed masks only; no feature cache (backbone trainable)',
        selection='maximum mean validation endpoint IoU over visible/shuffle/post_shuffle/exposed/lift probes',
        validation='fixed32 episodes, layouts32/33 and permutations0/9/16/18; not full256 val',
        action_representation=None, rollout='not run; segmentation fine-tuning only',
        future_FM='project current TCOW300 spatial hidden tokens + auxiliary camera tokens + measured joint embedding to FM width; direct cross-attention, no context decoder; not implemented in this run')
    chosen = splits(DATA)
    if args.preflight:
        chosen = dict(train=chosen['train'][:8], val=chosen['val'][:4])
    config['episode_counts'] = {k: len(v) for k, v in chosen.items()}
    write_json(out / 'config.json', config)
    write_json(out / 'splits.json', {k: [r['path'] for r in v] for k, v in chosen.items()})
    write_json(out / 'progress.json', dict(status='loading RAM cache', epoch=0, step=0))
    source = out / 'source'
    source.mkdir()
    for name in ('mask.py', 'mask_data.py', 'train_mask.py', 'backbone.py', 'point.py'):
        shutil.copyfile(Path('OATFlow/tracking') / name, source / name)
    config['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
    start = time.perf_counter()
    def cache_progress(group):
        def report(done, total):
            status = dict(status='loading RAM cache', group=group, completed_segments=done,
                          total_segments=total, elapsed_seconds=time.perf_counter()-start, epoch=0, step=0)
            memory = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
            status['process_rss_gib'] = int(memory['VmRSS'].split()[0]) / 2**20
            write_json(out / 'progress.json', status)
        return report
    train, segments = load(DATA, chosen['train'], progress=cache_progress('train'))
    val, segments = load(DATA, chosen['val'], segments, progress=cache_progress('val'))
    cache_seconds = time.perf_counter() - start
    cache = dict(segments=len(segments), rgb_gib=sum(s['rgb'].numel() for s in segments.values()) / 2**30,
                 mask_gib=sum(s['packed'].nbytes+s['visible'].nbytes for s in segments.values()) / 2**30,
                 seconds=cache_seconds, persistent_features=False)
    write_json(out / 'cache.json', cache)
    print('RAM CACHE', json.dumps(cache), flush=True)
    model = TargetMaskTCOW()
    load_original(model, WEIGHTS)
    config['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config['status'] = 'preflight' if args.preflight else 'running'
    assert config['trainable_parameters'] == sum(p.numel() for p in model.parameters())
    write_json(out / 'config.json', config)
    model.cuda()
    optimizer = torch.optim.AdamW([dict(params=model.backbone.parameters(), lr=1e-5),
        dict(params=model.tracker_post_linear.parameters(), lr=1e-4)], weight_decay=1e-4, fused=True)
    initial, records = evaluate(model, val, args.batch, 'Original TCOW validation')
    write_json(out / 'baseline.json', dict(metrics=initial, records=records))
    print('BASELINE', json.dumps(initial), flush=True)
    total_steps = (16 if args.preflight else args.epochs * ((len(train) + args.batch - 1) // args.batch))
    history = []
    timings = []
    best = -1.
    step = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    epochs = 1 if args.preflight else args.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        plan = train_plan(train, rng)
        if args.preflight:
            plan = (plan * ((16 * args.batch + len(plan) - 1) // len(plan)))[:16 * args.batch]
        bar = tqdm(range(0, len(plan), args.batch), desc=f'e114 epoch {epoch}/{epochs}', mininterval=5)
        for offset in bar:
            tick = time.perf_counter()
            rgb, query, truth, visible, occlusion = batch(train, plan[offset:offset+args.batch])
            step += 1
            factor = min(1., step / 50)
            for group, lr in zip(optimizer.param_groups, (1e-5, 1e-4)):
                group['lr'] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(rgb, query)
                loss, weighted_bce, bootstrap, jaccard = mask_loss(logits, truth, visible, occlusion, step / total_steps)
            loss.backward()
            if step == 1:
                assert model.backbone.patch_embed.proj.weight.grad[:, 3].abs().sum() > 0
                assert model.tracker_post_linear.weight.grad.abs().sum() > 0
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .3, error_if_nonfinite=True)
            optimizer.step()
            iou = float(frame_iou(logits.detach()[:, -1], truth[:, -1]).mean())
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - tick)
            record = dict(epoch=epoch, step=step, loss=float(loss), weighted_bce=float(weighted_bce),
                          bootstrap_bce=float(bootstrap), soft_jaccard=float(jaccard),
                          endpoint_iou=iou, grad_norm=float(norm), seconds=timings[-1])
            with (out / 'metrics.jsonl').open('a') as file:
                file.write(json.dumps(record) + '\n')
            bar.set_postfix(loss=f'{float(loss):.4f}', IoU=f'{iou:.3f}', BCE=f'{float(weighted_bce):.4f}', J=f'{float(jaccard):.3f}', refresh=False)
            if step % 10 == 0 or step == 1:
                print('TRAIN', json.dumps(record), flush=True)
                speed = float(np.mean(timings[-50:]))
                write_json(out / 'progress.json', dict(status='training', epoch=epoch, step=step,
                    total_steps=total_steps, latest=record, seconds_per_update=speed,
                    estimated_train_epoch_seconds=speed*((len(train)+args.batch-1)//args.batch),
                    remaining_train_seconds=(total_steps-step)*speed,
                    peak_vram_gib=torch.cuda.max_memory_allocated()/2**30))
            if step % 256 == 0 and not args.preflight:
                save(out / 'latest.pt', model, optimizer, epoch-1, step, config, best, rng)
        result, records = evaluate(model, val, args.batch, f'Validation epoch {epoch}')
        score = result['all']['mean_iou']
        history.append(dict(epoch=epoch, step=step, validation=result))
        write_json(out / 'history.json', history)
        print('VALIDATION', json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            save(out / 'best.pt', model, None, epoch, step, config, best, rng)
            write_json(out / 'best_validation.json', dict(epoch=epoch, step=step, metrics=result, records=records))
        save(out / 'latest.pt', model, optimizer, epoch, step, config, best, rng)
    summary = dict(status='complete', epochs=epochs, steps=step, best_val_iou=best,
        baseline_val_iou=initial['all']['mean_iou'], cache=cache,
        training_validation_seconds=time.perf_counter()-started,
        steady_update_seconds=float(np.mean(timings[3:])), peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    weights = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(weights['model'], strict=True)
    examples(model, val[:4], out, 'val')
    if not args.preflight:
        for group in ('composition', 'spatial', 'combined'):
            episodes, segments = load(DATA, chosen[group], segments)
            result, records = evaluate(model, episodes, args.batch, f'Test {group}')
            write_json(out / f'test_{group}.json', dict(metrics=result, records=records))
            examples(model, episodes[:4], out, group)
            summary[group] = result
    write_json(out / 'summary.json', summary)
    write_json(out / 'progress.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
