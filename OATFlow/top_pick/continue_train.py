"""Resume e103 to ten complete dataset passes, retaining optimizer and RNG."""
import hashlib
import json
import math
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.top_pick.center import BASE,ROOT,load_policy
from OATFlow.top_pick.center_train import write


def main():
    checkpoint=ROOT/'train/latest.pt';out=ROOT/'train_2';epochs=10
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    model=load_policy(saved).cuda().train()
    features=np.load(BASE/'cache/features.npy',mmap_mode='r')
    with np.load(BASE/'cache/labels.npz') as z:labels={k:z[k] for k in z.files}
    train_ids=np.flatnonzero(labels.pop('is_train'))
    val_ids=np.setdiff1d(np.arange(len(features)),train_ids)
    val_ids=np.random.default_rng(102).choice(val_ids,min(512,len(val_ids)),replace=False)
    labels={k:torch.from_numpy(v) for k,v in labels.items()}
    centers=torch.from_numpy(np.load(ROOT/'centers.npy'))
    assert len(centers)==len(features)
    config=json.loads((ROOT/'train/config.json').read_text())
    batch_size=config['batch'];steps_per_epoch=math.ceil(len(train_ids)/batch_size)
    total_steps=epochs*steps_per_epoch;start_step=saved['step']
    order=saved['order'].copy();cursor=saved['cursor']
    assert 0<start_step<total_steps
    np.testing.assert_array_equal(np.sort(order),train_ids)
    assert cursor==min(((start_step-1)%steps_per_epoch+1)*batch_size,len(order))
    samples_seen=((start_step-1)//steps_per_epoch)*len(train_ids)+cursor
    assert samples_seen==config['samples_seen']
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('flow.')],lr=2e-5),
            dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('flow.')],lr=1e-4)]
    optimizer=torch.optim.AdamW(groups,betas=(.9,.95),weight_decay=1e-4,fused=True)
    optimizer.load_state_dict(saved['optimizer'])
    assert all(float(v['step'])==start_step for v in optimizer.state.values())
    assert [g['lr'] for g in optimizer.param_groups]==[2e-5,1e-4]
    rng=np.random.default_rng();rng.bit_generator.state=saved['rng']['numpy']
    torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state(saved['rng']['cuda'])
    assert torch.equal(torch.get_rng_state(),saved['rng']['torch'])
    assert torch.equal(torch.cuda.get_rng_state(),saved['rng']['cuda'])
    for name,value in model.state_dict().items():
        assert torch.equal(value.detach().cpu(),saved['model'][name]),name
    for key in ('elapsed_s','best_val','test_summary','selected_checkpoint_step','probe_status'):
        config.pop(key,None)
    config.update(status='training',steps=total_steps,epochs=epochs,steps_per_epoch=steps_per_epoch,
                  train_passes=epochs,resume_checkpoint=str(checkpoint),resume_step=start_step,
                  resume_samples_seen=samples_seen,continuation_updates=total_steps-start_step,
                  samples_seen=samples_seen,test_status='not yet run',
                  initialization='resume e103 latest: model,AdamW moments,CPU/CUDA/NumPy RNG,order,cursor',
                  warmup='original global100-step warmup already complete;constant learning rates retained',
                  epoch_definition='one full shuffled pass over285600training chunks;keep final32-sample batch',
                  planned_test='final exact10epochs,K10,no temporal ensemble,same128cases')
    out.mkdir(exist_ok=False)
    write(out/'config.json',config)
    readme=('# e103 train_2\n\nStatus: running. Resume step3000 to22320,exactly10total epochs,not10additional. '
            'Same285600training chunks,final32-sample batch each epoch,batch128. '
            'Model,optimizer moments,CPU/CUDA/NumPy RNG,order,cursor restored. '
            'Frozen ViT-B/16;visual decoder/full reach-initialized FM and center head trainable. '
            'FM LR2e-5/visual1e-4,original100warmup already complete;AdamW(.9,.95),wd1e-4,clip1,BF16. '
            'Same hidden256 center MSE weight1;H25x10D EE anchor deltaXYZ,rotation6D,physical jaw. '
            'Same fixed512 validation chunks/noise/time,action FM loss selects best. '
            'Final exact10epoch checkpoint will be evaluated K10 without temporal ensemble on same128cases. '
            'Source reachbest115000,ImageNet ViT weights and unchanged data/cache are recorded in config.json. '
            'Test planned:H25/K10,Euler10,seed0,300frames,25Hzpolicy/50Hzservo. '
            'Actual config.json,validation.jsonl,metrics.jsonl,resume_check.json.\n')
    (out/'README.md').write_text(readme)
    best=saved['best_val']
    prior_best=torch.load(ROOT/'train/best.pt',map_location='cpu',weights_only=False,mmap=True)
    best_step=prior_best['step'];del prior_best
    # Keep the original selected checkpoint available if continuation never improves it.
    (out/'best.pt').hardlink_to(ROOT/'train/best.pt')
    first_ids=order[cursor:cursor+batch_size]
    write(out/'resume_check.json',dict(status='passed',step=start_step,cursor=cursor,
          samples_seen=samples_seen,epochs=samples_seen/len(train_ids),optimizer_states=len(optimizer.state),
          model_all_tensors_exact=True,optimizer_steps_restored=True,learning_rates_restored=True,
          cpu_cuda_rng_exact=True,numpy_rng_restored=True,order_is_exact_train_permutation=True,
          first_batch_ids_sha256=hashlib.sha256(first_ids.tobytes()).hexdigest(),
          total_steps=total_steps,target_samples=epochs*len(train_ids)))
    del saved
    started=time.perf_counter()

    def batch(ids):
        return (torch.from_numpy(np.array(features[ids],dtype=np.float32)).cuda(),
                *(labels[k][ids].cuda() for k in ('state','target','action','mask')),centers[ids].cuda())

    def save(name,step,include_optimizer):
        config.update(samples_seen=samples_seen,train_passes=samples_seen/len(train_ids),
                      selected_checkpoint_step=best_step)
        payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                     step=step,config=config,best_val=best)
        if include_optimizer:
            payload.update(optimizer=optimizer.state_dict(),
                rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=rng.bit_generator.state),
                order=order,cursor=cursor)
        tmp=out/(name+'.tmp');torch.save(payload,tmp);tmp.replace(out/(name+'.pt'))

    for step in tqdm(range(start_step+1,total_steps+1),desc='e103 to10epochs',initial=start_step,total=total_steps,mininterval=10):
        if cursor>=len(order):order=rng.permutation(train_ids);cursor=0
        ids=order[cursor:cursor+batch_size];cursor+=len(ids);samples_seen+=len(ids)
        optimizer.zero_grad(set_to_none=True)
        for group,lr in zip(optimizer.param_groups,(2e-5,1e-4)):group['lr']=min(step/100,1.)*lr
        with torch.autocast('cuda',dtype=torch.bfloat16):
            flow,aux,error_mm=model.loss_parts(*batch(ids));loss=flow+aux
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        with (out/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step,epoch=samples_seen/len(train_ids),loss=float(loss.detach()),
                flow_loss=float(flow.detach()),center_loss=float(aux.detach()),center_error_mm=float(error_mm.detach()),
                elapsed_s=time.perf_counter()-started))+'\n')
        if step%250==0 or step==total_steps:
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
            record=dict(step=step,epoch=samples_seen/len(train_ids),train_flow=float(flow.detach()),
                        train_center=float(aux.detach()),val_loss=score,val_center_loss=center_score,
                        val_center_error_mm=center_error,best=best,best_step=best_step,elapsed_s=time.perf_counter()-started)
            with (out/'validation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True);model.train()
    assert samples_seen==epochs*len(train_ids) and cursor==len(order)
    config.update(status='complete',elapsed_s=time.perf_counter()-started,best_val=best,
                  samples_seen=samples_seen,train_passes=samples_seen/len(train_ids),
                  selected_checkpoint_step=best_step,final_step=step)
    save('final',step,False);write(out/'config.json',config)
    (out/'README.md').write_text(readme.replace('Status: running','Status: complete')+
        f'\nFinished exactly{epochs}epochs,{samples_seen}samples,best step{best_step},FM validation{best}. Test not yet run.\n')
    print(json.dumps(config),flush=True)


if __name__=='__main__':main()
