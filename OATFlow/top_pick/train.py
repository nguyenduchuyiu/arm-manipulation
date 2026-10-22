"""Train visual target conditioning and the pretrained reach FM; keep ViT frozen."""
import json
import math
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.top_pick.collect import ROOT
from OATFlow.top_pick.model import TopPickPolicy


def main():
    out=ROOT/'train';out.mkdir(exist_ok=False)
    config=dict(status='training',source=str(ROOT/'cache'),steps=3000,batch=128,
                fm_lr=2e-5,visual_lr=1e-4,warmup_steps=100,seed=102,
                frozen='entire ViT-B/16',trainable='visual target query decoder and entire pretrained FM',
                horizon=25,action_dim=10,action_representation='EE chunk-anchor deltaXYZ,deltaRotation6D,absolute jaw',
                normalization='preserved reach checkpoint',optimizer='AdamW beta(.9,.95),wd1e-4,clip1',
                validation='fixed 512 validation chunks and noise/time RNG; best masked flow loss',
                precision='BF16',train_checkpoint_source_step=115000,policy_inputs='current top RGB,color target,measured joints',
                flow_steps=10,execute_chunk=10,test_budget=300,test_status='not yet run')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    readme=('# e102_top training\n\nStatus: running. Frozen ViT-B/16 top320px,64 pooled tokens. '
            'Target-query visual token plus pretrained measured-joint token; entire H25x10D reach FM from best step115000. '
            'All FM and visual target modules train; ViT frozen. 3000updates,batch128,seed102,BF16,AdamW '
            'beta(.9,.95),wd1e-4,clip1,100-step warmup then constant FM2e-5/visual1e-4. '
            'Stride1 train-only cache, saved reach normalization. '
            'Best selected by fixed512 validation chunks/noise/time, masked flow loss every250updates. '
            'No test split enters selection. Intended rollout test H25/K10,Euler10,seed0,300frames,25/50Hz,PoseIK. '
            'Test not run yet. Actual config.json,metrics.jsonl,validation.jsonl.\n')
    (out/'README.md').write_text(readme)
    torch.set_num_threads(4);torch.manual_seed(102);torch.backends.cuda.matmul.allow_tf32=True
    model=TopPickPolicy().cuda().train()
    features=np.load(ROOT/'cache/features.npy',mmap_mode='r')
    with np.load(ROOT/'cache/labels.npz') as z:labels={k:z[k] for k in z.files}
    train_ids=np.flatnonzero(labels.pop('is_train'));val_ids=np.setdiff1d(np.arange(len(features)),train_ids)
    val_ids=np.random.default_rng(102).choice(val_ids,min(512,len(val_ids)),replace=False)
    labels={k:torch.from_numpy(v) for k,v in labels.items()}
    cache_config=json.loads((ROOT/'cache/config.json').read_text())
    assert cache_config['status']=='complete' and cache_config['frames']==len(features)
    config.update(pretrained_source=cache_config['source'],pretrained_sha256=cache_config['source_sha256'],
                  vision_source=cache_config['vision'],vision_sha256=cache_config['vision_sha256'],
                  manifest_sha256=cache_config['manifest_sha256'],
                  train_chunks=len(train_ids),validation_chunks=len(val_ids),
                  train_passes=3000*128/len(train_ids),
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('flow.')],lr=2e-5),
            dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('flow.')],lr=1e-4)]
    optimizer=torch.optim.AdamW(groups,betas=(.9,.95),weight_decay=1e-4,fused=True)
    rng=np.random.default_rng(102);order=rng.permutation(train_ids);cursor=0;best=math.inf;samples_seen=0
    started=time.perf_counter()

    def batch(ids):
        return (torch.from_numpy(np.array(features[ids],dtype=np.float32)).cuda(),
                *(labels[k][ids].cuda() for k in ('state','target','action','mask')))

    def save(name,step,include_optimizer):
        payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},step=step,config=config,best_val=best)
        if include_optimizer:
            payload.update(optimizer=optimizer.state_dict(),rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=rng.bit_generator.state),order=order,cursor=cursor)
        tmp=out/(name+'.tmp');torch.save(payload,tmp);tmp.replace(out/(name+'.pt'))

    for step in tqdm(range(1,config['steps']+1),desc='e102 top FM',mininterval=5):
        if cursor>=len(order):order=rng.permutation(train_ids);cursor=0
        ids=order[cursor:cursor+config['batch']];cursor+=len(ids);samples_seen+=len(ids)
        optimizer.zero_grad(set_to_none=True)
        factor=min(step/100,1.)
        for group,lr in zip(optimizer.param_groups,(2e-5,1e-4)):group['lr']=factor*lr
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(*batch(ids))
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,loss=float(loss.detach()),elapsed_s=time.perf_counter()-started))+'\n')
        if step==1 or step%250==0 or step==config['steps']:
            model.eval();sums=[]
            with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                torch.manual_seed(102000)
                for start in range(0,len(val_ids),64):
                    ids=val_ids[start:start+64]
                    with torch.autocast('cuda',dtype=torch.bfloat16):value=model.loss(*batch(ids))
                    sums.append((float(value),len(ids)))
            score=sum(a*b for a,b in sums)/sum(b for a,b in sums)
            improved=score<best
            if improved:best=score;save('best',step,False)
            save('latest',step,True)
            record=dict(step=step,train_loss=float(loss.detach()),val_loss=score,best=best,elapsed_s=time.perf_counter()-started)
            with (out/'validation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True);model.train()
    config.update(status='complete',elapsed_s=time.perf_counter()-started,best_val=best,
                  samples_seen=samples_seen,train_passes=samples_seen/len(train_ids))
    save('final',step,False)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'README.md').write_text(readme.replace('Status: running','Status: complete')+f'\nBest validation loss {best}; train coverage {config["train_passes"]:.3f} passes. Learned rollout test is separate.\n')


if __name__=='__main__':main()
