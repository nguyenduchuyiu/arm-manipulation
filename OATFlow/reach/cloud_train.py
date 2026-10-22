"""E118: scratch single-camera target-cloud reach with the full E88 budget."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.reach.cloud_data import DATA, REFERENCE, ROOT, sha, write
from OATFlow.reach.cloud_eval import chunks, rollouts
from OATFlow.reach.pointcloud import PointCloudReach


def arrays():
    return {key: np.load(DATA/name, mmap_mode='r') for key, name in dict(
        actions='actions.npy', mask='action_mask.npy', joints='joint_states/state_joint_position.npy',
        episode='chunk_trajectory_index.npy').items()}


def fresh():
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model = PointCloudReach().cuda()
    statistics = json.loads((REFERENCE/'normalization.json').read_text())
    for name in ('state_mean', 'state_std', 'action_mean', 'action_std'):
        getattr(model.flow, name).copy_(torch.tensor(statistics[name], device='cuda'))
    for name in ('xyz_mean', 'xyz_std'):
        getattr(model, name).copy_(torch.tensor(statistics[name], device='cuda'))
    return model, statistics


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .95), weight_decay=1e-4, fused=True)


def lr_at(step):
    if step <= 100:
        return 2e-4*step/100
    if step >= 60000:
        return 2e-5
    fraction = (step-100)/(60000-100)
    return 2e-5 + .5*(2e-4-2e-5)*(1+math.cos(math.pi*fraction))


def model_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def batch(model, opt, data, clouds, ids, step):
    goal = torch.tensor(np.array(clouds[data['episode'][ids]]), device='cuda')
    state = torch.tensor(np.array(data['joints'][ids]), device='cuda')
    target = torch.tensor(np.array(data['actions'][ids]), device='cuda')
    mask = torch.tensor(np.array(data['mask'][ids]), device='cuda')
    for group in opt.param_groups:
        group['lr'] = lr_at(step)
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model.loss(goal, state, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return float(loss.detach()), float(norm)


def save(path, value):
    if shutil.disk_usage(path.parent).free < 2 << 30:
        raise RuntimeError('Less than2GiB free before checkpoint write')
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary); temporary.replace(path)


def preflight(output, clouds):
    output.mkdir(parents=True, exist_ok=False)
    (output/'README.md').write_text('# E118 preflight\n\nDisposable scratch model, full B256 BF16 training, '
        'cloud invariance/gradient checks and two-goal oracle execution. No retained weights.\n')
    model, _ = fresh(); opt = optimizer_for(model); data = arrays()
    with np.load(REFERENCE/'split.npz') as z:
        ids = np.random.default_rng(0).permutation(z['train_chunks'])[:256]
    model.eval()
    cloud = torch.tensor(np.array(clouds[:2]), device='cuda')
    joints = torch.tensor(np.array(data['joints'][:2]), device='cuda')
    kv = model.conditioning(cloud, joints)
    swapped = model.conditioning(cloud.flip(1), joints)
    error = max(float((a-b).abs().max()) for pair, other in zip(kv, swapped) for a, b in zip(pair, other))
    assert error == 0, 'Point order changed conditioning'
    translated = model.conditioning(cloud+torch.tensor([.02, 0, 0], device='cuda'), joints)
    difference = max(float((a-b).abs().max()) for pair, other in zip(kv, translated) for a, b in zip(pair, other))
    assert difference > 1e-5, 'Encoder lost absolute location'
    model.train(); initial = model_hash(model); losses = []
    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for step in tqdm(range(1, 21), desc='B256 scratch preflight', mininterval=5):
        losses.append(batch(model, opt, data, clouds, ids, step)[0])
        if step == 1:
            missing = [name for name, p in model.named_parameters() if p.grad is None]
            assert not missing, missing
            assert model.point_encoder[0].weight.grad.abs().sum() > 0
            assert model.flow.layers[-1].mlp.down_proj.weight.grad.abs().sum() > 0
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start
    assert initial != model_hash(model)
    rows = [json.loads(line) for line in (DATA/'trajectories.jsonl').read_text().splitlines()]
    with np.load(REFERENCE/'evaluation_indices.npz') as z:
        oracle, _ = rollouts(model.eval(), clouds, z['val_episodes'][:2], rows, oracle=True)
    assert oracle['summary']['success_1mm'] == 1, oracle['summary']
    result = dict(status='passed', point_order_error=error, translation_conditioning_difference=difference,
        batch=256, steps=20, elapsed_s=elapsed, seconds_per_update=elapsed/20,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        losses=losses, all_parameters_receive_gradients=True, initialization='random; no learned weights loaded',
        oracle=oracle, estimated_training_hours=120000*elapsed/20/3600,
        estimate_excludes_validation_and_checkpoints=True)
    write(output/'summary.json', result); print(json.dumps(result), flush=True)


def train(output, clouds, resume=None):
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    cloud_config = json.loads((ROOT/'clouds/config.json').read_text())
    assert cloud_config['status'] == 'complete' and len(clouds) == 32017
    assert cloud_config['clouds_sha256'] == sha(ROOT/'clouds/clouds.npy')
    model, statistics = fresh(); opt = optimizer_for(model); data = arrays()
    with np.load(REFERENCE/'split.npz') as z:
        train_ids = z['train_chunks'].copy()
    with np.load(REFERENCE/'evaluation_indices.npz') as z:
        evaluation = {k: z[k].copy() for k in z.files}
    rng = np.random.default_rng(0); order = rng.permutation(train_ids)
    cursor = seen = first_step = 0; best = [-1., -1e30]; history = []
    if resume:
        saved = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True); opt.load_state_dict(saved['optimizer'])
        assert saved['config']['clouds_sha256'] == cloud_config['clouds_sha256']
        order = saved['sampler']['order']; cursor = saved['sampler']['cursor']; seen = saved['sampler']['samples_seen']
        first_step = saved['step']; best = saved['best']; history = saved['validation_history']
        rng.bit_generator.state = saved['rng']['numpy_sampler']
        torch.set_rng_state(saved['rng']['torch_cpu']); torch.cuda.set_rng_state(saved['rng']['torch_cuda'])
        config = saved['config']; del saved
    else:
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(ROOT/'README.md', output/'README.md')
        for name in ('normalization.json', 'split.npz', 'evaluation_indices.npz'):
            shutil.copyfile(REFERENCE/name, output/name)
        config = dict(status='running', experiment='e118_cloud', initialization='random_from_scratch', seed=0,
            source_weights=None, initial_parameters_sha256=model_hash(model), dataset=str(DATA),
            cloud_dataset=str(ROOT/'clouds'), clouds_sha256=cloud_config['clouds_sha256'],
            cloud_input='128 base-frame points from calibrated front depth of isolated2cm voxel;2mm depth units',
            goal_XYZ_policy_input=False, frozen='none', parameters=sum(p.numel() for p in model.parameters()),
            batch_size=256, max_updates=120000, horizon=25, sample_hz=25, chunk_stride=2,
            schedule='warmup100 to2e-4; cosine to2e-5 at60000; constant2e-5 through120000',
            optimizer='fusedAdamW beta0.9/0.95 wd1e-4 clip1', precision='BF16 autocast/FP32 labels',
            input_state='measured five arm joints rad + measured jaw m',
            action='deltaXYZ + anchor-relative rotation6D + absolute jaw; original E88 labels/normalization',
            architecture='Point MLP3/64/128/256 + symmetric max + Linear256/960; same2context-token16layer720wide FM',
            auxiliary_loss=None, evaluation_every=5000, validation_chunks=2048, validation_rollouts=32,
            execute_chunk=15, flow_steps=10, max_policy_frames=300,
            model_selection='stable5mm validation success then lower median final error',
            validation_definition='reconstructed evaluator; explicit last25frame stability; not claimed byte-identical to removed historical evaluator',
            reference=str(REFERENCE), split_sha256=sha(REFERENCE/'split.npz'),
            evaluation_indices_sha256=sha(REFERENCE/'evaluation_indices.npz'),
            normalization_sha256=sha(REFERENCE/'normalization.json'),
            code_hashes={n: sha(Path(__file__).parent/n) for n in ('pointcloud.py', 'cloud_data.py', 'cloud_train.py', 'cloud_eval.py', 'flow.py')})
    np.testing.assert_array_equal(np.sort(order), np.sort(train_ids))
    rows = [json.loads(line) for line in (DATA/'trajectories.jsonl').read_text().splitlines()]
    write(output/'config.json', config)
    started = time.perf_counter(); losses = []; norms = []; block = time.perf_counter()
    model.train()
    bar = tqdm(range(first_step+1, 120001), desc='E118 scratch cloud reach', mininterval=5)
    for step in bar:
        if cursor+256 > len(order):
            order = rng.permutation(train_ids); cursor = 0
        ids = order[cursor:cursor+256]; cursor += 256; seen += 256
        loss, norm = batch(model, opt, data, clouds, ids, step)
        losses.append(loss); norms.append(norm)
        if step % 100 == 0:
            metric = dict(step=step, loss=float(np.mean(losses)), grad_norm=float(np.mean(norms)),
                lr=lr_at(step), samples_seen=seen, effective_train_passes=seen/len(train_ids),
                seconds_per_update=(time.perf_counter()-block)/len(losses),
                elapsed_this_launch_s=time.perf_counter()-started,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
            with (output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(metric)+'\n')
            write(output/'progress.json', dict(status='running', **metric))
            tqdm.write(json.dumps(metric)); bar.set_postfix(loss=f'{metric["loss"]:.5f}', lr=f'{lr_at(step):.2g}')
            losses.clear(); norms.clear(); block = time.perf_counter()
        if step % 5000 == 0:
            model.eval()
            validation = dict(step=step, chunks=chunks(model, clouds, data, evaluation['val_chunks']))
            result, physical = rollouts(model, clouds, evaluation['val_episodes'], rows)
            validation['rollout'] = result['summary']; history.append(validation)
            write(output/f'val_rollout_{step}.json', result)
            write(output/'validation_history.json', history)
            score = [result['summary']['success_5mm'], -result['summary']['final_median_mm']]
            if score > best:
                best = score
                save(output/'best.pt', dict(model=model.state_dict(), step=step, config=config,
                    statistics=statistics, selection=validation))
                np.savez_compressed(output/'best_validation_rollout.npz', **physical)
            tqdm.write(json.dumps(validation)); model.train(); block = time.perf_counter()
        if step == 100 or step % 1000 == 0:
            save(output/'latest.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step,
                config=config, statistics=statistics, best=best, validation_history=history,
                rng=dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(), numpy_sampler=rng.bit_generator.state),
                sampler=dict(order=order, cursor=cursor, samples_seen=seen)))
    os.link(output/'latest.pt', output/'final.pt')
    config.update(status='training_complete_testing', completed_updates=120000, samples_seen=seen,
                  effective_train_passes=seen/len(train_ids), elapsed_this_launch_s=time.perf_counter()-started)
    write(output/'config.json', config)
    best_saved = torch.load(output/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best_saved['model'], strict=True); best_step = best_saved['step']; del best_saved
    model.eval(); summary = dict(status='complete', completed_updates=120000, best_step=best_step, test={})
    # Keep the same held-out goals and flow-noise seeds as E88.
    for seed in (0, 1):
        for name, ids in [('initial', np.load(DATA/'trajectory_chunk_offsets.npy')[np.load(REFERENCE/'split.npz')['test_trajectories']]),
                          ('chunks', evaluation['test_chunks'])]:
            summary['test'][f'{name}_n{seed}'] = chunks(model, clouds, data, ids, seed, sample=True)
        result, physical = rollouts(model, clouds, evaluation['test_episodes'], rows, seed)
        summary['test'][f'rollout_n{seed}'] = result['summary']
        write(output/f'test_rollout_n{seed}.json', result)
        np.savez_compressed(output/f'test_rollout_n{seed}.npz', **physical)
    for name, ids, oracle in [('train_rollout', evaluation['train_episodes'], False),
                              ('oracle_EE_IK', evaluation['test_episodes'], True)]:
        result, physical = rollouts(model, clouds, ids, rows, oracle=oracle)
        summary['test'][name] = result['summary']; write(output/f'{name}.json', result)
        np.savez_compressed(output/f'{name}.npz', **physical)
    config.update(status='complete', best_step=best_step); write(output/'config.json', config)
    write(output/'summary.json', summary); write(output/'progress.json', summary)
    with (output/'README.md').open('a') as f:
        f.write(f'\nCompleted120000 updates. Selected step:{best_step}. Exact results: summary.json.\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    clouds = np.load(ROOT/'clouds/clouds.npy', mmap_mode='r')
    if args.preflight:
        preflight(ROOT/'preflight', clouds)
    else:
        train(ROOT/'train', clouds, args.resume)
