"""Resume the joint-input10D EE-delta flow on the retained voxel chunks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.reach import ReachPolicy


DATA = Path('/workspace/robotics_assets/datasets/reach_voxel_grid/ee_delta_25hz_h25_s2')
SOURCE = Path('/workspace/robotics_assets/models/reach_joint_delta_ee/latest.pt')


def train(source, output, updates):
    if updates < 1:
        raise ValueError('updates must be positive')
    description = ('# Joint-input delta-EE flow continuation\n\nStatus: running.\n'
        'All model parameters train. H25/25Hz, output deltaXYZ+relative rotation6D+absolute jaw. '
        'Train split, normalization, AdamW, RNG and sampler restored from source checkpoint. '
        'Masked flow-matching loss, BF16, clip1; constant restored LR. '
        'This minimal trainer saves latest; it does not select a best checkpoint or run rollouts. '
        'Actual source/steps/batch/LR: config.json; losses:metrics.jsonl.\n')
    output.mkdir(parents=True, exist_ok=False)
    (output/'README.md').write_text(description)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    assert saved['config']['state_dim'] == 6 and saved['config']['action_dim'] == 10
    model = ReachPolicy().cuda(); model.load_state_dict(saved['model'], strict=True)
    names = list(dict(model.named_parameters()))
    assert names == [name for name in saved['model'] if name in dict(model.named_parameters())]
    assert float(model.flow.action_mean[-1]) == 0 and float(model.flow.action_std[-1]) == 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(.9,.95), weight_decay=1e-4, fused=True)
    optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:
        for parameter in group['params']:
            assert optimizer.state[parameter]['exp_avg'].shape == parameter.shape
    actions = np.load(DATA/'actions.npy', mmap_mode='r')
    masks = np.load(DATA/'action_mask.npy', mmap_mode='r')
    joints = np.load(DATA/'joint_states/state_joint_position.npy', mmap_mode='r')
    episode_ids = np.load(DATA/'chunk_trajectory_index.npy', mmap_mode='r')
    goals = np.load(DATA/'target_base_m.npy', mmap_mode='r')
    split = Path('/workspace/robotics_assets/models/reach_joint_delta_ee/split.npz')
    with np.load(split) as z:
        train_ids = z['train_chunks'].copy()
    rng = np.random.default_rng(); rng.bit_generator.state = saved['rng']['numpy_sampler']
    order = saved['sampler']['order'].copy(); cursor = saved['sampler']['cursor']
    seen = saved['sampler']['samples_seen']; first_step = saved['step']
    np.testing.assert_array_equal(np.sort(order), np.sort(train_ids))
    torch.set_rng_state(saved['rng']['torch_cpu'])
    torch.cuda.set_rng_state(saved['rng']['torch_cuda'])
    config = dict(saved['config'], dataset=str(DATA), source=str(source),
        source_step=first_step, max_updates=first_step+updates,
        trainer='OATFlow.reach.train: constant restored LR, masked10D flow loss; no best selection',
        lr=float(optimizer.param_groups[0]['lr']), schedule='constant restored LR',
        evaluation_every=None, validation_chunks=0, validation_rollouts=0,
        model_selection='not performed by minimal continuation trainer',
        output=str(output))
    statistics = saved['statistics']; batch = config['batch_size']
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    del saved
    model.train()
    for step in tqdm(range(first_step+1, first_step+updates+1), desc='Joint-input10D FM updates', mininterval=5):
        if cursor + batch > len(order):
            order = rng.permutation(train_ids); cursor = 0
        ids = order[cursor:cursor+batch]; cursor += batch; seen += len(ids)
        goal = torch.tensor(np.array(goals[episode_ids[ids]]),dtype=torch.float32,device='cuda')
        state = torch.tensor(np.array(joints[ids]),device='cuda')
        target = torch.tensor(np.array(actions[ids]),device='cuda')
        mask = torch.tensor(np.array(masks[ids]),device='cuda')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss = model.loss(goal,state,target,mask)
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step,loss=float(loss.detach())))+'\n')
    payload = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
        config=config,statistics=statistics,
        rng=dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(),numpy_sampler=rng.bit_generator.state),
        sampler=dict(order=order,cursor=cursor,samples_seen=seen))
    temporary = output/'latest.tmp'; torch.save(payload,temporary); temporary.replace(output/'latest.pt')
    (output/'README.md').write_text(description.replace('Status: running.','Status: complete.')+f'\nFinal update:{step}. Evaluation not run.\n')
    return dict(step=step,loss=float(loss.detach()),cursor=cursor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=1000)
    args=parser.parse_args(); print(json.dumps(train(args.resume,args.output,args.updates)),flush=True)


if __name__=='__main__':
    main()
