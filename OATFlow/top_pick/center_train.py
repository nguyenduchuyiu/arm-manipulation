"""Matched target-center training with optional random FM initialization."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.top_pick.collect import DATA
from OATFlow.top_pick.center import CenterPolicy,BASE,ROOT


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare_centers():
    rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    with np.load(BASE/'cache/labels.npz') as z:
        reference_state=z['state'];reference_target=z['target'];train=z['is_train']
    rng=np.random.default_rng(102);values=[];offset=0
    for row in tqdm(rows,desc='Align current target centers to e102 cache',mininterval=5):
        if row['evaluation_group'] not in ('train','val'):continue
        n=row['frames']-1
        starts=np.arange(n) if row['evaluation_group']=='train' else np.sort(rng.choice(n,min(8,n),replace=False))
        path=DATA/row['path']
        with np.load(path/'labels.npz') as z:
            centers=z['entity_xyz_base_m'][starts,row['target_id']].astype(np.float32)
            times=z['timestamp_s'][starts]
        with np.load(path/'observation.npz') as z:
            np.testing.assert_array_equal(z['joint_position'][starts].astype(np.float32),reference_state[offset:offset+len(starts)])
            np.testing.assert_array_equal(z['timestamp_s'][starts],times)
        np.testing.assert_array_equal(reference_target[offset:offset+len(starts)],np.full(len(starts),row['target_id']))
        values.append(centers);offset+=len(starts)
    assert offset==len(train)
    centers=np.concatenate(values)
    np.save(ROOT/'centers.npy',centers)
    mean=centers[train].astype(np.float64).mean(0).astype(np.float32)
    std=centers[train].astype(np.float64).std(0).astype(np.float32)
    assert np.isfinite(centers).all() and (std>1e-4).all()
    statistics=dict(mean_base_m=mean.tolist(),std_m=std.tolist(),train_frames=int(train.sum()),
                    frames=len(centers),normalization='train frames only',
                    target='current physical target body center in robot-base metres; same frame as top RGB',
                    alignment='all cached measured states,target IDs,timestamps checked exactly',
                    observation_input=False)
    write(ROOT/'center_statistics.json',statistics)
    return centers,mean,std


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scratch-fm',action='store_true')
    p.add_argument('--epochs',type=int)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.epochs is not None and args.epochs<1:p.error('epochs must be positive')
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.scratch_fm:
        centers=np.load(ROOT/'centers.npy')
        stats=json.loads((ROOT/'center_statistics.json').read_text())
        mean=np.array(stats['mean_base_m'],dtype=np.float32);std=np.array(stats['std_m'],dtype=np.float32)
    else:
        centers,mean,std=prepare_centers()
    out=args.output or ROOT/'train';out.mkdir(parents=True,exist_ok=False)
    config=json.loads((BASE/'train/config.json').read_text())
    for key in ('elapsed_s','samples_seen','best_val','selected_checkpoint_step','test_summary'):
        config.pop(key,None)
    config.update(status='training',test_status='not yet run',source=str(BASE/'cache'),
                  baseline=str(BASE),seed=102,steps=3000,batch=128,
                  aux_center=dict(weight=1.,loss='mean squared error in train-normalized XYZ',
                                  head='LayerNorm256 + Linear256to3',
                                  attachment='256D target decoder hidden,before existing goal projection256to960',
                                  head_seed=103,statistics=str(ROOT/'center_statistics.json'),
                                  label='current target object center,robot-base metres',ground_truth_input=False),
                  trainable='e102 visual conditioning and entire FM plus auxiliary center head',
                  validation='same fixed512 chunks/noise/time; select best by action FM loss only',
                  initialization='fresh matched e102 initialization,not e102 trained checkpoint',
                  total_loss='L_FM + 1.0 * L_center',test_checkpoint_source_step=115000)
    if args.scratch_fm:
        config.update(fm_pretrained=False,pretrained_source=None,pretrained_sha256=None,
                      train_checkpoint_source_step=None,test_checkpoint_source_step=None,
                      normalization_source=config['pretrained_source'],normalization_sha256=config['pretrained_sha256'],
                      baseline=str(ROOT/'train_2'),baseline_test=str(ROOT/'test_k10_2'),
                      initialization='random entire FlowMatchingHead seed102;matched e103 visual/center initial tensors and RNG',
                      center_labels=str(ROOT/'centers.npy'),
                      planned_test='final exact10epochs,K10,no temporal ensemble,same128cases')
    write(out/'config.json',config)
    readme=('# e103_center training\n\nStatus: running. Same e102 data/cache/initialization,seed102,3000updates,'
            'batch128,1.344passes,FM LR2e-5/visual1e-4,warmup100,AdamW(.9,.95),wd1e-4,clip1,BF16. '
            'Frozen ViT-B/16;reach FM source115000;H25x10D anchor EE delta plus absolute jaw. '
            'One auxiliary LN256+Linear3 predicts current target center from target decoder hidden256 before goal projection. '
            'L=L_FM+MSE(train-normalized XYZ),weight1. Ground-truth center is only a training/validation label. '
            'All shared initial parameters and RNG match e102;head initialization is isolated in fork_rng. '
            'Same512validation chunks/noise/time;best selected using FM action loss only. '
            'Planned learned testH25/K10,Euler10,seed0,300frames,25/50Hz on same128cases;'
            'same928scene target-conditioning probe. Test/probe not run yet. '
            'Exact config.json,metrics.jsonl,validation.jsonl,../center_statistics.json.\n')
    (out/'README.md').write_text(readme)
    torch.set_num_threads(4);torch.manual_seed(102);torch.backends.cuda.matmul.allow_tf32=True
    model=CenterPolicy(fm_pretrained=not args.scratch_fm).cuda().train()
    model.center_mean.copy_(torch.from_numpy(mean).cuda());model.center_std.copy_(torch.from_numpy(std).cuda())
    features=np.load(BASE/'cache/features.npy',mmap_mode='r')
    with np.load(BASE/'cache/labels.npz') as z:labels={k:z[k] for k in z.files}
    train_ids=np.flatnonzero(labels.pop('is_train'));val_ids=np.setdiff1d(np.arange(len(features)),train_ids)
    val_ids=np.random.default_rng(102).choice(val_ids,min(512,len(val_ids)),replace=False)
    labels={k:torch.from_numpy(v) for k,v in labels.items()};centers=torch.from_numpy(centers)
    assert len(centers)==len(features)
    config.update(parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  train_chunks=len(train_ids),validation_chunks=len(val_ids))
    steps_per_epoch=math.ceil(len(train_ids)/config['batch'])
    if args.epochs is not None:
        config.update(epochs=args.epochs,steps_per_epoch=steps_per_epoch,steps=args.epochs*steps_per_epoch)
    full_epochs,remainder=divmod(config['steps'],steps_per_epoch)
    config.update(train_passes=full_epochs+min(remainder*config['batch'],len(train_ids))/len(train_ids))
    if args.scratch_fm:
        readme=(f'# {out.parent.name} training\n\nStatus: running. Same e103 data/cache,seed102,'
                f'{config["steps"]}updates/{args.epochs}epochs,batch128,final32-sample batch per epoch. '
                'Entire FM randomly initialized;no learned reach FM weights loaded. '
                'Same state/action normalization and ImageNet frozen ViT-B/16. '
                'Matched initial visual/target/decoder/goal/center parameters and RNG. '
                'FM LR2e-5/visual1e-4,100warmup then constant,AdamW(.9,.95),wd1e-4,clip1,BF16. '
                'Same hidden256 center MSE weight1,H25x10D anchor deltaXYZ,rotation6D,physical jaw. '
                'Fixed512validation chunks/noise/time,every250steps plus final;best by action FM loss. '
                'Final exact10epoch K10 test planned,same128cases,Euler10,seed0,300frames,25/50Hz. '
                'Actual config.json,metrics.jsonl,validation.jsonl.\n')
        (out/'README.md').write_text(readme)
    write(out/'config.json',config)
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('flow.')],lr=2e-5),
            dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('flow.')],lr=1e-4)]
    optimizer=torch.optim.AdamW(groups,betas=(.9,.95),weight_decay=1e-4,fused=True)
    rng=np.random.default_rng(102);order=rng.permutation(train_ids);cursor=0;best=math.inf;best_step=None;samples_seen=0
    started=time.perf_counter()

    def batch(ids):
        return (torch.from_numpy(np.array(features[ids],dtype=np.float32)).cuda(),
                *(labels[k][ids].cuda() for k in ('state','target','action','mask')),centers[ids].cuda())

    def save(name,step,include_optimizer):
        config.update(samples_seen=samples_seen,selected_checkpoint_step=best_step)
        payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},step=step,config=config,best_val=best)
        if include_optimizer:
            payload.update(optimizer=optimizer.state_dict(),rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=rng.bit_generator.state),order=order,cursor=cursor)
        tmp=out/(name+'.tmp');torch.save(payload,tmp);tmp.replace(out/(name+'.pt'))

    for step in tqdm(range(1,config['steps']+1),desc=out.parent.name+' FM + current target center',mininterval=10):
        if cursor>=len(order):order=rng.permutation(train_ids);cursor=0
        ids=order[cursor:cursor+128];cursor+=len(ids);samples_seen+=len(ids)
        optimizer.zero_grad(set_to_none=True)
        for group,lr in zip(optimizer.param_groups,(2e-5,1e-4)):group['lr']=min(step/100,1.)*lr
        with torch.autocast('cuda',dtype=torch.bfloat16):flow,aux,error_mm=model.loss_parts(*batch(ids));loss=flow+aux
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        with (out/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step,samples_seen=samples_seen,epoch=samples_seen/len(train_ids),loss=float(loss.detach()),flow_loss=float(flow.detach()),center_loss=float(aux.detach()),center_error_mm=float(error_mm.detach()),elapsed_s=time.perf_counter()-started))+'\n')
        if step==1 or step%250==0 or step==config['steps']:
            model.eval();sums=[]
            with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                torch.manual_seed(102000)
                for start in range(0,len(val_ids),64):
                    ids=val_ids[start:start+64]
                    with torch.autocast('cuda',dtype=torch.bfloat16):values=model.loss_parts(*batch(ids))
                    sums.append((*[float(v) for v in values],len(ids)))
            score,center_score,center_error=[sum(r[k]*r[3] for r in sums)/sum(r[3] for r in sums) for k in range(3)]
            if score<best:best=score;best_step=step;save('best',step,False)
            save('latest',step,True)
            record=dict(step=step,train_flow=float(flow.detach()),train_center=float(aux.detach()),
                        val_loss=score,val_center_loss=center_score,val_center_error_mm=center_error,
                        best=best,elapsed_s=time.perf_counter()-started)
            with (out/'validation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True);model.train()
    config.update(status='complete',elapsed_s=time.perf_counter()-started,best_val=best,
                  samples_seen=samples_seen,train_passes=samples_seen/len(train_ids),
                  final_step=step,selected_checkpoint_step=best_step)
    if args.epochs is not None:assert samples_seen==args.epochs*len(train_ids) and cursor==len(order)
    save('final',step,False);write(out/'config.json',config)
    (out/'README.md').write_text(readme.replace('Status: running','Status: complete')+f'\nBest action validation loss {best}. Learned tests/probes are separate.\n')


if __name__=='__main__':main()
