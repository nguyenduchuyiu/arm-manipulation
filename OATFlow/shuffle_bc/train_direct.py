"""Expert-only direct visual FM BC; frozen TCOW/ViT caches, no context decoder."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.shuffle_bc.context_model import FrozenWristViT
from OATFlow.shuffle_bc.data import DATA, splits
from OATFlow.shuffle_bc.direct_data import allocate, batch, cache_group, grouped, memory_gib
from OATFlow.shuffle_bc.direct_model import DirectPolicy
from OATFlow.top_pick.data import SOURCE, VISION
from OATFlow.tracking.data import subset
from OATFlow.tracking.linear_probe import fingerprint
from OATFlow.tracking.mask import TargetMaskTCOW
from OATFlow.tracking.mask_video import digest
from OATFlow.tracking.utils import write_json


TRACKER = Path('/workspace/robotics_assets/experiments/e114_mask/train_3/best.pt')
def frozen_models():
    saved = torch.load(TRACKER, map_location='cpu', weights_only=True, mmap=True)
    assert (saved['epoch'], saved['step']) == (9, 4608)
    tracker = TargetMaskTCOW()
    tracker.load_state_dict(saved['model'], strict=True)
    return tracker.eval().requires_grad_(False).cuda(), FrozenWristViT().eval().requires_grad_(False).cuda()


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9, .95), weight_decay=1e-4, fused=True)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    numerator, denominator = 0., 0
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(115000)
        for start in tqdm(range(0, len(data['top']), 32), desc='Fixed expert FM validation', mininterval=5, leave=False):
            values = batch(data, slice(start, start+32))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model.loss(**values)
            n = int(values['mask'].sum())
            numerator += float(loss)*n
            denominator += n
    return numerator/denominator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--encoding-batch', type=int, default=32)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    assert min(args.epochs, args.batch, args.encoding_batch) > 0
    torch.set_num_threads(4)
    torch.manual_seed(115)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(115)
    size, encoding_size = args.batch, args.encoding_batch
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e115_README.md', out / 'README.md')
    write_json(out / 'progress.json', dict(status='initializing frozen perception and FM', step=0))
    config = dict(status='initializing', seed=115, data=str(DATA), train_episodes=2048,
        epochs=args.epochs, batch=size, encoding_batch=encoding_size,
        TCOW_checkpoint=str(TRACKER), TCOW_epoch=9, TCOW_step=4608, TCOW_sha256=digest(TRACKER),
        FM_checkpoint=str(SOURCE), FM_source_step=115000, FM_source_sha256=digest(SOURCE),
        auxiliary_camera='left, static surrogate wrist', actual_wrist_camera=False,
        vision_source=str(VISION), vision_sha256=digest(VISION),
        frozen='entire e114 TCOW including unused mask head; auxiliary ImageNet ViT-B16',
        trainable='only flow.*: FM including existing joint proprio MLP/state projection/context KV projections',
        architecture='300 TCOW current spatial hidden +64auxiliary ViT tokens +1joint token -> FM cross-attention',
        context_decoder=False, XYZ_head=False, XYZ_loss=False, trainable_visual_projection=False,
        visual_width_adapter='fixed per-token RMS normalization, zero-pad768to960, fixed view codes in padding',
        TCOW_history='same as e114:30uniform past/current cached stride3frames, frame0 retained, context concatenated before expert',
        prompt='visible target mask at context frame0; zero query at all later frames',
        observations='top RGB history, current left RGB, measured current joints only; no GT object XYZ, targetID or later masks',
        expert_stride=3, context_training_samples=0, expert_points=234434,
        actions='H25/25Hz measured EE deltaXYZ3 and relative rotation6D anchored at current pose; absolute jaw1',
        loss='expert-only masked normalized flow velocity MSE', normalization='reach buffers unchanged; jaw=-1-2physical/.0255',
        cache='RAM only:300TCOW tokens int8 with FP16 scale per16features;64auxiliary tokens FP16; same quantize/dequantize at online inference',
        training_order='epoch1 shuffled context groups:encode each group once then train its expert chunks; epochs2+ global shuffled RAM indices',
        optimizer='AdamW(.9,.95),weight_decay1e-4,clip1', learning_rate=2e-5, warmup=100,
        precision='BF16 autocast; FP32 parameters and optimizer', gradient_checkpointing=False,
        validation='32fixed held-out episodes,8uniform expert stride3starts per episode; fixed32batch/noise-time seed115000',
        selection='minimum fixed validation FM loss', rollout_status='not run',
        resume_checkpoint=str(args.resume) if args.resume else None,
        source_sha256={str(p): digest(p) for p in [Path(__file__), Path('OATFlow/shuffle_bc/direct_model.py'),
            Path('OATFlow/shuffle_bc/direct_data.py'), Path('OATFlow/shuffle_bc/context_model.py'),
            Path('OATFlow/tracking/mask.py'), Path('OATFlow/tracking/backbone.py'), Path('OATFlow/reach/flow.py')]})
    for name in config['source_sha256']:
        path = Path(name).resolve()
        destination = out / 'source' / path.relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    write_json(out / 'config.json', config)
    train_rows, val_rows = splits()['train'], subset(DATA)['val']
    write_json(out / 'episodes.json', dict(train=train_rows, val=val_rows))
    model = DirectPolicy().cuda()
    optimizer = optimizer_for(model)
    tracker, camera = frozen_models()
    frozen = dict(TCOW=fingerprint(tracker), auxiliary_ViT=fingerprint(camera))
    write_json(out / 'frozen_hashes.json', frozen)
    train, val = allocate(train_rows), allocate(val_rows, validation=True)
    assert len(train['top']) == 234434 and len(val['top']) == 256
    config.update(cache_gib=memory_gib(train)+memory_gib(val), train_cache_gib=memory_gib(train),
                  val_cache_gib=memory_gib(val), trainable_parameters=sum(p.numel() for p in model.parameters()))
    write_json(out / 'config.json', config)
    print('CONFIG', json.dumps(config), flush=True)
    np.savez_compressed(out / 'train_indices.npz', episode=np.concatenate([np.full(len(a), i, np.int32) for i, a in enumerate(train['starts'])]),
                        expert_raw_frame=np.concatenate(train['starts']))
    for group in grouped(val_rows):
        cache_group(tracker, camera, val, group, encoding_size, name='val')
    assert val['filled'].all()
    source_weights = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)['model']
    assert all(torch.equal(value.cpu(), source_weights['flow.'+name]) for name, value in model.flow.state_dict().items())
    del source_weights
    initial = evaluate(model, val)
    write_json(out / 'initial_validation.json', dict(flow=initial))
    cache_groups = grouped(train_rows)
    cache_order = rng.permutation(len(cache_groups)).tolist()
    next_group, start_epoch, next_batch, order = 0, 2, 0, None
    step, samples_seen, best = 0, 0, initial
    resumed_phase = 'cache_epoch1'
    if args.resume:
        previous = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
        for key in ('TCOW_sha256', 'FM_source_sha256', 'batch', 'source_sha256'):
            assert previous['config'][key] == config[key], key
        model.load_state_dict(previous['model'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        step, samples_seen, best = previous['step'], previous['samples_seen'], previous['best_flow']
        cache_order = previous['cache_order']
        resumed_phase, next_group = previous['phase'], previous['next_group']
        start_epoch, next_batch, order = previous['next_epoch'], previous['next_batch'], previous['order']
        rng.bit_generator.state = previous['numpy_rng']
        torch.set_rng_state(previous['torch_rng'])
        torch.cuda.set_rng_state(previous['cuda_rng'])
        shutil.copyfile(args.resume.parent / 'best.pt', out / 'best.pt')
        del previous
    else:
        payload = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, epoch=0, step=0,
                       samples_seen=0, best_flow=initial, config=config)
        torch.save(payload, out / 'best.pt')
    started = time.perf_counter()
    timings = []

    def save(name, phase, epoch, cursor=0, epoch_order=None, group_cursor=0):
        actual_epoch = (1 if phase == 'cache_epoch1' else
                        epoch-1 if name == 'latest' and cursor == 0 and epoch_order is None else epoch)
        payload = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, config=config,
            epoch=actual_epoch, step=step, samples_seen=samples_seen, best_flow=best)
        if name == 'latest':
            payload.update(optimizer=optimizer.state_dict(), phase=phase, next_group=group_cursor,
                cache_order=cache_order, next_epoch=epoch, next_batch=cursor, order=epoch_order,
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        temporary = out / f'{name}.tmp'
        torch.save(payload, temporary)
        temporary.replace(out / f'{name}.pt')

    def update(ids, epoch, phase):
        nonlocal step, samples_seen
        tick = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = batch(train, ids)
        for group in optimizer.param_groups:
            group['lr'] = 2e-5*min((step+1)/100, 1.)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model.loss(**values)
        loss.backward()
        assert all(p.grad is None and not p.requires_grad for vision in (tracker, camera) if vision is not None for p in vision.parameters())
        if step == 0:
            assert model.flow.state_proj.weight.grad.abs().sum() > 0
            assert model.flow.proprio[0].weight.grad.abs().sum() > 0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        samples_seen += len(values['top'])
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-tick)
        record = dict(status='training', phase=phase, epoch=epoch, step=step, samples_seen=samples_seen,
            batch=len(values['top']), flow=float(loss), grad_norm=float(norm), seconds=timings[-1],
            cache_filled=int(train['filled'].sum()), cache_total=len(train['top']),
            elapsed_seconds=time.perf_counter()-started, peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
        with (out / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        if step == 1 or step%10 == 0:
            print('TRAIN', json.dumps(record), flush=True)
            write_json(out / 'progress.json', record)

    def validate(epoch, group_cursor=0):
        nonlocal best
        loss = evaluate(model, val)
        record = dict(epoch=epoch, step=step, group_cursor=group_cursor, flow=loss, samples_seen=samples_seen)
        with (out / 'validation.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print('VALIDATION', json.dumps(record), flush=True)
        if loss < best:
            best = loss
            save('best', 'epoch', epoch)
        return loss

    model.eval()
    for position, group_id in enumerate(tqdm(cache_order, desc='Epoch1: frozen cache + expert FM', mininterval=5)):
        group = cache_groups[group_id]
        cache_group(tracker, camera, train, group, encoding_size, out, timer=started)
        if resumed_phase != 'cache_epoch1' or position < next_group:
            continue
        ids = np.concatenate([np.arange(*train['slices'][i]) for i in group])
        rng.shuffle(ids)
        for offset in range(0, len(ids), size):
            update(ids[offset:offset+size], 1, 'cache_epoch1')
        if position == 0:
            assert dict(TCOW=fingerprint(tracker), auxiliary_ViT=fingerprint(camera)) == frozen
        if position == 0 or (position+1)%16 == 0 or position+1 == len(cache_order):
            validate(1, position+1)
            save('latest', 'cache_epoch1', 2, group_cursor=position+1)
    assert train['filled'].all()
    assert dict(TCOW=fingerprint(tracker), auxiliary_ViT=fingerprint(camera)) == frozen
    assert all(not p.requires_grad and p.grad is None for v in (tracker, camera) for p in v.parameters())
    if resumed_phase == 'cache_epoch1':
        assert samples_seen == len(train['top'])
        save('latest', 'epoch', 2)
    del tracker, camera
    tracker = camera = None
    torch.cuda.empty_cache()
    config.update(status='training from complete RAM cache', cache_completed_seconds=time.perf_counter()-started)
    write_json(out / 'config.json', config)
    for epoch in range(start_epoch, args.epochs+1):
        if order is None:
            order = rng.permutation(len(train['top']))
            next_batch = 0
        for offset in tqdm(range(next_batch, len(order), size), desc=f'FM epoch {epoch}/{args.epochs}', mininterval=5):
            update(order[offset:offset+size], epoch, 'RAM_cache')
            if step%256 == 0:
                save('latest', 'epoch', epoch, offset+size, order)
        validate(epoch)
        order = None
        next_batch = 0
        save('latest', 'epoch', epoch+1)
    assert samples_seen == args.epochs*len(train['top'])
    result = dict(status='complete', epochs=args.epochs, step=step, samples_seen=samples_seen, best_flow=best,
        initial_flow=initial, cache_gib=config['cache_gib'], frozen_vision_unchanged=True,
        elapsed_seconds=time.perf_counter()-started, median_update_seconds=float(np.median(timings)),
        peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(out / 'summary.json', result)
    write_json(out / 'progress.json', result)
    config['status']='complete'
    write_json(out / 'config.json', config)
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
