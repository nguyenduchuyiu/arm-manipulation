"""Stage one: full-data TCOW XYZ training, without instantiating an action policy."""
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

from OATFlow.shuffle_bc.data import DATA, splits, clip
from OATFlow.tracking.data import evaluation_plan
from OATFlow.tracking.point import PointXYZ, load_tcow
from OATFlow.tracking.point_data import load_points
from OATFlow.tracking.utils import write_json

ROOT=Path('/workspace/robotics_assets/experiments/e106_track')
WEIGHTS=Path('/workspace/robotics_assets/pretrained/vision')


def inputs(episodes, points):
    batches=[clip(episodes[i],end) for i,end in points]
    return tuple(torch.cat([b[j] for b in batches]).cuda() for j in range(3))


def endpoints(episodes):
    return [(i,end) for i,ep in enumerate(episodes) for end in range(len(ep['raw_index']))]


@torch.no_grad()
def evaluate(model, episodes):
    model.eval();records=[]
    plan=evaluation_plan(episodes)
    for start in tqdm(range(0,len(plan),2),desc='XYZ validation',mininterval=5,leave=False):
        points=plan[start:start+2]
        rgb,query,xyz=inputs(episodes,[(i,end) for i,end,_ in points])
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model.predict(rgb,query)
        errors=(prediction-xyz[:,-1]).norm(dim=-1).cpu().numpy()*1000
        for (i,end,phase),error,value in zip(points,errors,prediction.cpu().numpy()):
            records.append(dict(episode=episodes[i]['row']['path'],raw_frame=int(episodes[i]['raw_index'][end]),
                phase=phase,error_mm=float(error),prediction_base_m=value.tolist(),
                truth_base_m=episodes[i]['xyz'][end].tolist()))
    phases={phase:float(np.mean([r['error_mm'] for r in records if r['phase']==phase]))
            for phase in sorted({r['phase'] for r in records})}
    errors=np.array([r['error_mm'] for r in records])
    return dict(macro_phase_mm=float(np.mean(list(phases.values()))),mean_mm=float(errors.mean()),
        median_mm=float(np.median(errors)),by_phase_mm=phases,points=len(records)),records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.batch<1 or args.epochs<1:parser.error('batch and epochs must be positive')
    torch.set_num_threads(4);torch.manual_seed(106)
    torch.backends.cuda.matmul.allow_tf32=True
    rng=np.random.default_rng(106)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile('docs/research/e106_README.md',out/'README.md')
    groups=splits();train_rows=groups['train'][:4] if args.preflight else groups['train']
    val_rows=[r for r in groups['val'] if r['layout_id'] in (32,33) and r['permutation'] in (0,9,16,18)]
    if args.preflight:val_rows=val_rows[:4]
    write_json(out/'progress.json',dict(status='loading top RGB',step=0,train_episodes=len(train_rows)))
    train,segments=load_points(DATA,train_rows,stride=10)
    val,segments=load_points(DATA,val_rows,segments=segments,stride=10)
    points=endpoints(train)
    assert len(points)==sum(len(ep['raw_index']) for ep in train)
    assert len(set(points))==len(points)
    for ep in train+val:
        assert ep['raw_index'][0]==0 and np.all(np.diff(ep['raw_index'])>0)
        assert ep['raw_index'][-1]==ep['row']['frames']-1
    labels=np.concatenate([ep['xyz'] for ep in train]).astype(np.float64)
    model=PointXYZ()
    load_tcow(model,WEIGHTS/'tracking_backbone.pt',WEIGHTS/'tracking_query.pt')
    model.set_xyz_statistics(labels.mean(0),labels.std(0).clip(.01))
    model=model.cuda()
    optimizer=torch.optim.AdamW([
        dict(params=model.backbone.parameters(),lr=1e-5),
        dict(params=[p for name,p in model.named_parameters() if not name.startswith('backbone.')],lr=1e-4)],
        betas=(.9,.95),weight_decay=1e-4,fused=True)
    assert all(p.requires_grad for p in model.parameters())
    assert not any(name.startswith(('flow.','goal.')) for name,_ in model.named_parameters())
    epoch_steps=math.ceil(len(points)/args.batch)
    steps=8 if args.preflight else args.epochs*epoch_steps
    config=dict(status='preflight' if args.preflight else 'training',seed=106,data=str(DATA),
        train_episodes=len(train),val_episodes=len(val),train_points=len(points),steps=steps,
        epochs=None if args.preflight else args.epochs,epoch_updates=epoch_steps,batch=args.batch,
        frames=30,endpoint_stride=10,context_stride=10,expert_stride=10,
        sampling='full shuffled pass over every train context/expert endpoint including terminal XYZ labels; final small batch retained',
        prefix='uniform raw time, frame0 to current; nearest available past RGB; initial dot only',
        rgb_cache='raw uint8 top frames, stride10 and segment-terminal frames, RAM only; no feature cache',
        rgb_cache_bytes=sum(s['rgb'].numel() for s in segments.values()),crop='320square rows40:280 -> 240x320',
        inputs='past/current top RGB and initial radius3 UV dot only; no ID, joints, XYZ or camera input',
        initialization='original official TCOW RGB+query weights; new random pooling/XYZ head',
        model_config=model.config,parameters=sum(p.numel() for p in model.parameters()),frozen='none',
        loss='0.5 normalized current-frame XYZ MSE + 0.5 normalized 30-frame prefix XYZ MSE',
        action_policy=False,mask_head=False,mask_files_loaded=False,
        lrs=dict(backbone=1e-5,heads=1e-4),optimizer='AdamW(.9,.95),wd1e-4,clip1',
        warmup=100,schedule='constant after warmup',precision='BF16',gradient_checkpointing=False,gradient_accumulation=False,
        xyz_mean=model.xyz_mean.tolist(),xyz_std=model.xyz_std.tolist(),
        validation='fixed phase endpoints on 32 val episodes; select equal-weight mean of visible/shuffle/post_shuffle/exposed/lift mean XYZ errors',
        test_status='tracking validation only during training; full held-out tracker test not run; no action-policy test',
        weights_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (WEIGHTS/'tracking_backbone.pt',WEIGHTS/'tracking_query.pt')},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path('OATFlow/tracking/point.py'),Path('OATFlow/tracking/point_data.py'),
             Path('OATFlow/tracking/data.py'),Path('OATFlow/tracking/backbone.py'),Path('OATFlow/shuffle_bc/data.py')]})
    write_json(out/'config.json',config)
    readme=(out/'README.md').read_text().replace('Status: starting.',
        'Status: running preflight.' if args.preflight else 'Status: training.')
    (out/'README.md').write_text(readme)
    with (out/'README.md').open('a') as stream:
        stream.write(f'\nActual run: preflight={args.preflight}; train={len(train)}, validation={len(val)} episodes; '
            f'{len(points)} train endpoints; batch={args.batch}; {steps} updates; '
            f'RGB cache={config["rgb_cache_bytes"]/2**30:.2f}GiB.\n')
    write_json(out/'samples.json',dict(train=[r['path'] for r in train_rows],validation=[r['path'] for r in val_rows]))
    order=rng.permutation(len(points));cursor=0;start_step=0;samples_seen=0;best=math.inf;best_step=0
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True)
        assert saved['config']['batch']==args.batch and saved['config']['train_points']==len(points)
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        order=saved['order'];cursor=saved['cursor'];samples_seen=saved['samples_seen'];start_step=saved['step']
        best=saved['best_val_mm'];best_step=saved['best_step'];rng.bit_generator.state=saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        shutil.copy2(args.resume.parent/'best.pt',out/'best.pt')
        config.update(resume_checkpoint=str(args.resume),resume_step=start_step)
        write_json(out/'config.json',config);del saved
    started=time.perf_counter();timings=[]

    def save(name,step,resumable=False):
        payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},model_config=model.config,
            config=config,step=step,samples_seen=samples_seen,best_val_mm=best,best_step=best_step)
        if resumable:payload.update(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state,order=order,cursor=cursor)
        tmp=out/(name+'.tmp');torch.save(payload,tmp);tmp.replace(out/(name+'.pt'))

    if not args.preflight and start_step==0:
        initial,records=evaluate(model,val);best=initial['macro_phase_mm']
        write_json(out/'initial_val.json',dict(metrics=initial,records=records));save('best',0)
        print(json.dumps(dict(step=0,validation=initial)),flush=True)
    progress=tqdm(range(start_step+1,steps+1),initial=start_step,total=steps,desc='e106 TCOW XYZ',mininterval=5)
    try:
        for step in progress:
            if cursor==len(order):order=rng.permutation(len(points));cursor=0
            ids=order[cursor:cursor+args.batch];cursor+=len(ids)
            tick=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
            rgb,query,xyz=inputs(train,[points[i] for i in ids])
            for group,lr in zip(optimizer.param_groups,(1e-5,1e-4)):group['lr']=min(step/100,1)*lr
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(rgb,query)
            error=(prediction.float()-model.normalize_xyz(xyz)).square()
            loss=.5*error[:,-1].mean()+.5*error.mean()
            assert torch.isfinite(loss)
            loss.backward()
            if step==start_step+1:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                assert model.backbone.patch_embed.proj.weight.grad[:,3].abs().sum()>0
                write_json(out/'gradient_checks.json',dict(all_parameters_receive_finite_gradients=True,
                    query_channel_gradient_nonzero=True,FM_instantiated=False,mask_head=False))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();samples_seen+=len(ids)
            mm=(prediction[:,-1].float()*model.xyz_std+model.xyz_mean-xyz[:,-1]).norm(dim=-1).mean()*1000
            record=dict(status='training',step=step,expected=steps,samples_seen=samples_seen,
                train_passes=samples_seen/len(points),batch_size=len(ids),loss=float(loss.detach()),
                current_xyz_mm=float(mm.detach()),grad_norm=float(norm),batch_seconds=time.perf_counter()-tick,
                elapsed_s=time.perf_counter()-started,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            timings.append(record['batch_seconds'])
            with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            write_json(out/'progress.json',record)
            progress.set_postfix(loss=f"{record['loss']:.4f}",mm=f"{record['current_xyz_mm']:.1f}",refresh=False)
            if step==1 or step%10==0:print(json.dumps(record),flush=True)
            if not args.preflight and (step==1 or step%500==0 or step==steps):
                optimizer.zero_grad(set_to_none=True)
                metric,records=evaluate(model,val)
                if metric['macro_phase_mm']<best:
                    best=metric['macro_phase_mm'];best_step=step;save('best',step)
                save('latest',step,True)
                value=dict(step=step,best_step=best_step,metrics=metric)
                with (out/'validation.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
                write_json(out/'last_validation.json',dict(**value,records=records))
                print(json.dumps(value),flush=True)
    except KeyboardInterrupt:
        # The latest atomic checkpoint contains the last completed validation boundary.
        config.update(status='interrupted_by_user',last_logged_step=step)
        write_json(out/'config.json',config)
        with (out/'README.md').open('a') as stream:stream.write(f'\nInterrupted on user request near step {step}; resume from latest.pt.\n')
        raise
    timing=float(np.median(timings[2:] or timings))
    if args.preflight:
        result=dict(status='passed',batch=args.batch,updates=steps,median_batch_seconds=timing,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,parameters=config['parameters'])
        write_json(out/'preflight.json',result);print(json.dumps(result),flush=True)
        (out/'README.md').write_text((out/'README.md').read_text().replace('Status: running preflight.','Status: preflight passed.'))
    else:
        assert samples_seen==args.epochs*len(points) and cursor==len(order)
        save('final',steps)
        config.update(status='complete',elapsed_s=time.perf_counter()-started,samples_seen=samples_seen,
            train_passes=samples_seen/len(points),best_step=best_step,best_val_mm=best)
        write_json(out/'config.json',config)
        (out/'README.md').write_text((out/'README.md').read_text().replace('Status: training.','Status: complete.'))
        print(json.dumps(dict(status='complete',best_step=best_step,best_macro_phase_mm=best)),flush=True)
    with (out/'README.md').open('a') as stream:
        stream.write(f'\nCompleted {steps-start_step} updates; batch{args.batch}; median {timing:.3f}s/update. Exact configuration: config.json.\n')


if __name__=='__main__':main()
