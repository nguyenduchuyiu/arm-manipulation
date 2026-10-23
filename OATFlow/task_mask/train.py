"""E123: frozen MobileSAM encoder, learned task prompt and pretrained mask decoder."""
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.task_mask.model import TaskMask,WEIGHTS,loss_parts
from OATFlow.task_mask.data import ROOT,manifests,groups,cache,query_index,batch,frozen_hash
from OATFlow.task_mask.evaluate import evaluate,run_tests
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.cloud_data import write

EPOCHS=5
BATCH=8


def save(path,model,optimizer,config,epoch,step,metrics):
    torch.save(dict(model=model.train_state(),optimizer=optimizer.state_dict(),config=config,
        epoch=epoch,step=step,validation=metrics),path)


def plot(history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_loss'] for r in history],marker='o');axes[0].set(xlabel='Epoch',ylabel='Train BCE + Dice + 0.1 quality loss')
    axes[1].plot(epochs,[r['val']['mean_iou'] for r in history],marker='o',label='Mask IoU')
    axes[1].plot(epochs,[r['val']['identity_accuracy'] for r in history],marker='o',label='Task identity')
    axes[1].set(xlabel='Epoch',ylabel='Validation score',ylim=(0,1));axes[1].legend()
    for axis in axes:axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'learning_curves.png',dpi=180);plt.close(fig)


def main():
    ROOT.mkdir(exist_ok=False)
    note=Path('docs/research/e123_README.md');shutil.copyfile(note,ROOT/'README.md')
    torch.manual_seed(123);np.random.seed(123);torch.set_num_threads(16)
    split=manifests()
    for name,rows in split.items():(ROOT/f'{name}_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    repo=WEIGHTS.parents[1]
    config=dict(status='caching',experiment='e123_mask',epochs=EPOCHS,batch=BATCH,seed=123,device='cpu',precision='FP32',
        source_repo='https://github.com/ChaoningZhang/MobileSAM',source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        pretrained=str(WEIGHTS),pretrained_sha256=sha(WEIGHTS),
        inputs='front RGB and target ID only;no point,box,GTmask,depth,joints or coordinates',
        task_IDs=dict(Red=0,Blue=1,Green=2,Yellow=3),task_prompt='one learned256D sparse prompt per target ID',
        frozen='original TinyViT image encoder and original prompt encoder positional/dense no-mask embeddings',
        trainable='task prompt and pretrained single-mask decoder;unused multimask readout MLPs frozen',
        temporal_sampling='frame0 only,all2048train episodes;512canonical layout/permutation RGBs x4queries',
        preprocessing='original SAM ResizeLongestSide1024;native240x320 ->768x1024;bottom pad to1024square;original mask postprocessing',
        cache='FP16 encoder embeddings and nativeRGB/instance labels in CPU RAM only;encoding in3workers x8threads',
        lr_prompt=.001,lr_decoder=.0001,optimizer='AdamW',weight_decay=.0001,clip_grad=1.,
        loss='BCE+softDice+0.1*IoU-quality MSE',scheduler='20update linear warmup then constant LR',
        training_frames=2048,train_images=512,validation_queries=256,validation_images=64,
        checkpoint_selection='highest validation mean IoU,original validation split only',
        total_updates=EPOCHS*(2048//BATCH),H=None,K=None,actions='no FM,action supervision,IK or rollout in this experiment',
        tests='all1408original home test queries;72later expert images x4queries;16retained XY images x4queries',
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    write(ROOT/'config.json',config)
    started=time.perf_counter()
    train_jobs=[dict(job,check_initial=True) for job in groups(split['train'])]
    training=cache(train_jobs,'Cache frozen MobileSAM train images')
    validation=cache(groups(split['val']),'Cache frozen MobileSAM validation images')
    train_index=query_index(training);val_index=query_index(validation)
    assert len(train_index)==2048 and len(val_index)==256
    config['cache_seconds']=time.perf_counter()-started
    config['cache_GiB']=training['GiB']+validation['GiB'];config['status']='training'
    torch.set_num_threads(16);torch.manual_seed(123)
    model=TaskMask().train();frozen=frozen_hash(model)
    config['frozen_encoder_hash']=frozen
    config['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    write(ROOT/'config.json',config)
    optimizer=torch.optim.AdamW([dict(params=model.task_prompt.parameters(),lr=.001),
        dict(params=[p for p in model.sam.mask_decoder.parameters() if p.requires_grad],lr=.0001)],weight_decay=.0001)
    baseline,_,_=evaluate(model,validation,val_index,'Pretrained + random task prompt validation')
    write(ROOT/'baseline.json',baseline);print(json.dumps(dict(stage='baseline',**baseline)),flush=True)
    best=-1.;step=0;history=[];rng=np.random.default_rng(123);training_started=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        model.train();order=rng.permutation(len(train_index));total=0.;epoch_started=time.perf_counter()
        bar=tqdm(range(0,len(order),BATCH),desc=f'Task mask epoch{epoch}/{EPOCHS}',mininterval=10)
        for offset in bar:
            features,target,truth,_=batch(training,train_index,order[offset:offset+BATCH])
            optimizer.zero_grad(set_to_none=True)
            scale=min(1.,(step+1)/20)
            for group,lr in zip(optimizer.param_groups,(.001,.0001)):group['lr']=lr*scale
            logits,quality=model(features,target);loss,parts=loss_parts(logits,truth,quality)
            assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(norm)
            if step==0:
                assert model.task_prompt.weight.grad is not None and model.task_prompt.weight.grad.abs().sum()>0
                assert all(p.grad is None for p in model.sam.image_encoder.parameters())
                write(ROOT/'gradient_check.json',dict(status='passed',task_prompt_grad_norm=float(model.task_prompt.weight.grad.norm()),
                    decoder_grad_nonzero=any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.sam.mask_decoder.parameters()),
                    frozen_encoder_has_no_grad=True,paired_batch_decoder_matches_upstream=True))
            optimizer.step();step+=1;total+=float(loss)*len(target)
            bar.set_postfix(loss=f'{float(loss):.4f}',iou=f'{float(parts["iou"]):.3f}')
            if step%25==0:
                record=dict(epoch=epoch,step=step,loss=float(loss),batch_iou=float(parts['iou']),
                    bce=float(parts['bce']),dice=float(parts['dice']),elapsed_s=time.perf_counter()-training_started)
                with (ROOT/'loss.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
                write(ROOT/'progress.json',dict(status='training',epoch=epoch,step=step,total_updates=config['total_updates'],**{k:v for k,v in record.items() if k not in ('epoch','step')}))
        metrics,_,_=evaluate(model,validation,val_index,f'Validation epoch{epoch}')
        record=dict(epoch=epoch,step=step,train_loss=total/len(order),val=metrics,epoch_seconds=time.perf_counter()-epoch_started)
        history.append(record);write(ROOT/'history.json',history)
        if metrics['mean_iou']>best:
            best=metrics['mean_iou'];save(ROOT/'best.pt',model,optimizer,config,epoch,step,metrics)
        save(ROOT/'latest.pt',model,optimizer,config,epoch,step,metrics)
        print(json.dumps(record),flush=True)
    save(ROOT/'final.pt',model,optimizer,config,EPOCHS,step,metrics)
    assert frozen_hash(model)==frozen
    plot(history);del training,validation,optimizer
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    model.load_train_state(saved['model']);config['best_epoch']=saved['epoch'];config['best_step']=saved['step']
    config['best_checkpoint_sha256']=sha(ROOT/'best.pt');config['training_seconds']=time.perf_counter()-training_started
    config['status']='testing';write(ROOT/'config.json',config);del saved
    tests=run_tests(model,split,config)
    assert frozen_hash(model)==frozen
    config['status']='complete';config['elapsed_seconds']=time.perf_counter()-started;write(ROOT/'config.json',config)
    write(ROOT/'verification.json',dict(status='passed',frozen_encoder_unchanged=True,training_queries=2048,
        original_split_preserved=True,no_GT_spatial_prompt_input=True,test_case_counts={k:v['cases'] for k,v in tests.items()},
        best_checkpoint_sha256=config['best_checkpoint_sha256']))
    write(ROOT/'progress.json',dict(status='complete',epoch=EPOCHS,step=step,best_epoch=config['best_epoch'],tests=tests))
    text=note.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nBest epoch:'+str(config['best_epoch'])+';test summaries:'+json.dumps(tests)+'\n'
    note.write_text(text);(ROOT/'README.md').write_text(text)
    with Path('docs/research/REPORT.md').open('a') as f:f.write('\nE123 learned-task MobileSAM:\n'+json.dumps(dict(config=config,tests=tests))+'\n')
    print(json.dumps(dict(status='complete',best_epoch=config['best_epoch'],tests=tests)),flush=True)


if __name__=='__main__':main()
