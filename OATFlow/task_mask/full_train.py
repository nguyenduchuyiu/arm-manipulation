"""E124: continue task-prompt MobileSAM on every train trajectory frame."""
import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from OATFlow.task_mask.model import TaskMask,WEIGHTS,loss_parts
from OATFlow.task_mask.data import manifests,groups,query_index,frozen_hash
from OATFlow.task_mask.full_data import ROOT,Frames,native_cache,temporal_jobs,gpu_cache,read_episode
from OATFlow.task_mask.evaluate import evaluate,sample_sheet,extra_jobs,outside_cache
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.features import sha

SOURCE=ROOT.parent/'e123_mask/best.pt'
EPOCHS=5
IMAGES=64
NOTE=Path('docs/research/e124_README.md')


def fresh():
    torch.manual_seed(123)
    model=TaskMask().cuda()
    checkpoint=torch.load(SOURCE,map_location='cpu',weights_only=False)
    model.load_train_state(checkpoint['model'])
    return model.train()


def optimizer(model):
    return torch.optim.AdamW([dict(params=model.task_prompt.parameters(),lr=.001),
        dict(params=[p for p in model.sam.mask_decoder.parameters() if p.requires_grad],lr=.0001)],weight_decay=.0001)


def update(model,opt,pixels,instances,step):
    pixels=pixels.cuda(non_blocking=True);instances=instances.cuda(non_blocking=True)
    target=torch.arange(4,device='cuda').repeat(len(pixels))
    truth=instances.repeat_interleave(4,dim=0)==target[:,None,None]+1
    opt.zero_grad(set_to_none=True)
    for group,lr in zip(opt.param_groups,(.001,.0001)):group['lr']=lr*min(1.,(step+1)/20)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        with torch.no_grad():features=model.sam.image_encoder(model.sam.preprocess(pixels))
        logits,quality=model(features.repeat_interleave(4,dim=0),target)
    loss,parts=loss_parts(logits.float(),truth,quality.float())
    assert torch.isfinite(loss)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    assert torch.isfinite(norm)
    if step==0:
        assert model.task_prompt.weight.grad is not None and model.task_prompt.weight.grad.abs().sum()>0
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.sam.mask_decoder.parameters())
        assert all(p.grad is None for p in model.sam.image_encoder.parameters())
    opt.step()
    return {k:float(v.detach()) for k,v in dict(loss=loss,grad_norm=norm,**parts).items()}


def save(name,model,opt,config,epoch,step,cursor,order,generator,best,metrics,full=False,epoch_stats=None):
    payload=dict(model=model.train_state(),config=config,epoch=epoch,step=step,validation=metrics,best_iou=best)
    if full:
        payload.update(optimizer=opt.state_dict(),cursor=cursor,order=order,
                       rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),
                                numpy=generator.bit_generator.state))
        if epoch_stats is not None:payload['epoch_stats']=epoch_stats
    path=ROOT/(name+'.pt');temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def preflight(row):
    model=fresh();frozen=frozen_hash(model)
    rgb,instance=read_episode(row)
    dataset=Frames(rgb,instance)
    pixels,masks,_=next(iter(DataLoader(dataset,batch_size=IMAGES,shuffle=False)))
    # Check batched preprocessing and the paired four-query decoder against SAM.
    model.eval()
    with torch.no_grad():
        expected=model.encode(rgb[0]);actual=model.sam.image_encoder(model.sam.preprocess(pixels[:1].cuda()))
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        ids=torch.arange(4,device='cuda');logits,quality=model(actual.expand(4,-1,-1,-1),ids)
        low,q=model.sam.mask_decoder(image_embeddings=actual,image_pe=model.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=model.task_prompt(ids)[:,None],
            dense_prompt_embeddings=model.sam.prompt_encoder.no_mask_embed.weight.reshape(1,256,1,1).expand(4,-1,-1,-1),
            multimask_output=False)
        original=model.sam.postprocess_masks(low,(768,1024),(240,320))[:,0]
        torch.testing.assert_close(logits,original,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(quality,q[:,0],rtol=1e-5,atol=1e-5)
    model.train();opt=optimizer(model);times=[];torch.cuda.reset_peak_memory_stats()
    for i in range(4):
        torch.cuda.synchronize();started=time.perf_counter()
        parts=update(model,opt,pixels,masks,i)
        torch.cuda.synchronize();times.append(time.perf_counter()-started)
    assert frozen_hash(model)==frozen
    result=dict(status='passed',images=IMAGES,queries=4*IMAGES,seconds=times,
                steady_update_s=float(np.median(times[1:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                frozen_encoder_unchanged=True,original_resize_bit_exact=True,paired_decoder_matches_original=True,
                nonzero_task_and_decoder_gradients=True,last_loss=parts)
    write(ROOT/'preflight.json',result);print(json.dumps(dict(stage='preflight',**result)),flush=True)
    del model,opt,dataset,pixels,masks,rgb,instance;gc.collect();torch.cuda.empty_cache()
    return result


def test(model,split,config):
    test_root=ROOT/'test';test_root.mkdir(exist_ok=False)
    note=Path('docs/research/e124_test_README.md').read_text()
    (test_root/'README.md').write_text(note)
    results={}
    for name,jobs in (('home',groups(split['test'])),('dynamic',extra_jobs(split['test'])),('xy',None)):
        out=test_root/name;out.mkdir();(out/'README.md').write_text(note)
        cached=outside_cache(model) if jobs is None else gpu_cache(model,jobs,'Encode test '+name)
        index=query_index(cached)
        summary,records,packed=evaluate(model,cached,index,'Test '+name,save=out)
        sample_sheet(cached,index,records,packed,out)
        sample_sheet(cached,index,records,packed,out,filename='worst.png',worst=True)
        write(out/'config.json',dict(status='complete',checkpoint=str(ROOT/'best.pt'),
              checkpoint_sha256=sha(ROOT/'best.pt'),sampling='original E123 fixed test protocol',cases=len(index)))
        results[name]=summary;del cached;gc.collect()
    write(test_root/'summary.json',results)
    (test_root/'README.md').write_text(note.replace('Status: PLANNED.','Status: COMPLETE.'))
    for name in results:(test_root/name/'README.md').write_text((test_root/'README.md').read_text())
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true',help='Continue E124 latest.pt with its optimizer, RNG and frame cursor')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(123);np.random.seed(123)
    torch.backends.cuda.matmul.allow_tf32=True
    resumed=None
    if args.resume:
        resumed=torch.load(ROOT/'latest.pt',map_location='cpu',weights_only=False)
        assert resumed['step']<resumed['config']['total_updates'],'Training budget already complete'
    else:ROOT.mkdir(exist_ok=False)
    NOTE.write_text(NOTE.read_text().replace('Status: PLANNED.','Status: RUNNING.'))
    shutil.copyfile(NOTE,ROOT/'README.md')
    started=time.perf_counter();split=manifests()
    for name,rows in split.items():
        path=ROOT/f'{name}_manifest.jsonl'
        if resumed is not None:assert list(map(json.loads,path.read_text().splitlines()))==rows
        else:path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    n=sum(r['frames'] for r in split['train']);updates=math.ceil(n/IMAGES)
    memory=Path('/sys/fs/cgroup/memory')
    available=int((memory/'memory.limit_in_bytes').read_text())-int((memory/'memory.usage_in_bytes').read_text())
    val_jobs=temporal_jobs(split['val'])
    required=n*240*320*4+len(val_jobs)*(256*64*64*2+240*320*4)+16*2**30
    assert required<available,'Insufficient RAM for full native-frame cache'
    config=dict(experiment=ROOT.name,status='preflight',epochs=EPOCHS,images_per_batch=IMAGES,queries_per_batch=IMAGES*4,
        source=str(SOURCE),source_sha256=sha(SOURCE),pretrained=str(WEIGHTS),pretrained_sha256=sha(WEIGHTS),
        initialization='E123 best epoch5 step1280 task prompt + decoder; same original frozen TinyViT; fresh AdamW',
        inputs='front RGB + target ID only; no GT mask,points,boxes,depth,joints,XYZ',
        frozen='original TinyViT image encoder and prompt encoder',trainable='four learned256D task tokens + original single-mask decoder',
        train_episodes=len(split['train']),train_images=n,queries_per_epoch=4*n,temporal_stride=1,
        sampling='every frame including final frame,all4task queries per image,global shuffled frame order,each frame exactly once per epoch',
        validation='8uniform temporal frames per each original val episode,all4queries;fixed validation only selects best mean IoU',
        validation_images=len(val_jobs),validation_queries=4*len(val_jobs),test='original heldout home + fixed later-phase + XY diagnostics after training',
        precision='CUDA BF16 autocast; FP32 trainable weights and loss',device=torch.cuda.get_device_name(),seed=123,
        lr_prompt=.001,lr_decoder=.0001,optimizer='AdamW',weight_decay=.0001,clip_grad=1.,warmup_updates=20,
        loss='BCE+softDice+0.1*IoU-quality MSE;empty visible masks are supervised as empty',
        preprocessing='original PIL SAM ResizeLongestSide1024 ->768x1024 + bottom pad1024;no crop or augmentation',
        cache='native RGB uint8 + front instance labels in CPU RAM;frozen encoder online;validation embeddings FP16 RAM;no disk features',
        train_cache_GiB=n*240*320*4/2**30,embedding_cache_would_be_GiB=n*256*64*64*2/2**30,
        RAM_available_GiB=available/2**30,RAM_required_with_headroom_GiB=required/2**30,
        updates_per_epoch=updates,total_updates=updates*EPOCHS,H=None,K=None,actions='segmentation only,no FM or manipulation rollout',
        source_files_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    if resumed is not None:
        original=json.loads((ROOT/'config.json').read_text())
        for key in ('epochs','images_per_batch','train_images','temporal_stride','total_updates','source_sha256'):
            assert original[key]==config[key],key
        config=original
        config.update(resume_step=resumed['step'],resume_epoch=resumed['epoch'],
                      resume_source_files_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    write(ROOT/'config.json',config)
    if resumed is None:
        timing=preflight(split['train'][0]);config['estimated_train_hours']=timing['steady_update_s']*updates*EPOCHS/3600
    config['status']='recaching_resume' if resumed is not None else 'caching';write(ROOT/'config.json',config)
    images,instances,cache_info=native_cache(split['train'])
    write(ROOT/('cache_info_resume.json' if resumed is not None else 'cache_info.json'),cache_info)
    model=fresh();frozen=frozen_hash(model)
    if resumed is not None:
        model.load_train_state(resumed['model'])
        assert frozen==config['frozen_encoder_hash']
    validation=gpu_cache(model,val_jobs,'Encode fixed temporal validation');val_index=query_index(validation)
    if resumed is None:
        baseline,_,_=evaluate(model,validation,val_index,'E123 initialization temporal validation')
        write(ROOT/'baseline.json',baseline);print(json.dumps(dict(stage='baseline',**baseline)),flush=True)
    opt=optimizer(model);rng=np.random.default_rng(124);best=-1.;history=[];step=0
    start_epoch=1;previous_elapsed=0.
    if resumed is not None:
        opt.load_state_dict(resumed['optimizer'])
        step=int(resumed['step']);best=resumed['best_iou'];start_epoch=int(resumed['epoch'])
        assert all(float(v['step'])==step for v in opt.state.values())
        assert len(resumed['order'])==n and np.array_equal(np.sort(resumed['order']),np.arange(n))
        assert step==(start_epoch-1)*updates+math.ceil(resumed['cursor']/IMAGES)
        rng.bit_generator.state=resumed['rng']['numpy']
        history=json.loads((ROOT/'history.json').read_text())
        if resumed['cursor']==n:start_epoch+=1
        assert len(history)==start_epoch-1
        previous=[json.loads(line) for line in (ROOT/'loss.jsonl').read_text().splitlines()]
        previous_elapsed=resumed.get('epoch_stats',{}).get('elapsed_s')
        if previous_elapsed is None:
            previous_elapsed=next(r['elapsed_s'] for r in reversed(previous) if r['step']<=step)
        verification=dict(status='passed',checkpoint_step=step,epoch=start_epoch,cursor=resumed['cursor'],
                          optimizer_steps_restored=True,unique_order_frames=n,original_manifests_unchanged=True,
                          frozen_encoder_unchanged=True,original_budget_updates=updates*EPOCHS,
                          RNG='CPU/CUDA restored after loader creation;NumPy frame-order generator restored',
                          partial_epoch_loss='Missing pre-interruption loss sum is not reconstructed or reported as a full-epoch mean')
        write(ROOT/'resume_verification.json',verification);print(json.dumps(dict(stage='resume',**verification)),flush=True)
    config.update(status='training',cache_seconds=cache_info['seconds'],validation_cache_GiB=validation['GiB'],
                  frozen_encoder_hash=frozen,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    write(ROOT/'config.json',config);training_started=time.perf_counter()
    for epoch in range(start_epoch,EPOCHS+1):
        continuing=resumed is not None and epoch==resumed['epoch']
        order=resumed['order'] if continuing else rng.permutation(n)
        seen=int(resumed['cursor']) if continuing else 0
        stats=resumed.get('epoch_stats',dict(loss_sum=0.,start_cursor=seen)) if continuing else dict(loss_sum=0.,start_cursor=0)
        model.train();total=stats['loss_sum'];loss_start=stats['start_cursor'];epoch_started=time.perf_counter()
        loader=DataLoader(Frames(images,instances),batch_size=IMAGES,sampler=order[seen:],num_workers=4,
                          pin_memory=True,prefetch_factor=1,multiprocessing_context='fork')
        iterator=iter(loader)
        if resumed is not None and epoch==start_epoch:
            torch.set_rng_state(resumed['rng']['torch']);torch.cuda.set_rng_state(resumed['rng']['cuda'])
        bar=tqdm(iterator,total=updates,initial=math.ceil(seen/IMAGES),desc=f'E124 epoch{epoch}/{EPOCHS}',mininterval=10)
        for pixels,masks,indices in bar:
            np.testing.assert_array_equal(indices.numpy(),order[seen:seen+len(indices)])
            parts=update(model,opt,pixels,masks,step);step+=1;seen+=len(indices);total+=parts['loss']*len(indices)
            bar.set_postfix(loss=f'{parts["loss"]:.4f}',iou=f'{parts["iou"]:.3f}')
            if step==1 or step%25==0 or (resumed is not None and step==resumed['step']+1):
                elapsed=previous_elapsed+time.perf_counter()-training_started
                record=dict(status='training',epoch=epoch,step=step,total_updates=config['total_updates'],
                            images_seen=(epoch-1)*n+seen,epoch_frames=seen,elapsed_s=elapsed,
                            ETA_hours=elapsed/step*(config['total_updates']-step)/3600,
                            peak_VRAM_GiB=torch.cuda.max_memory_allocated()/2**30,**parts)
                with (ROOT/'loss.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                write(ROOT/'progress.json',record);print(json.dumps(record),flush=True)
            if step==1 or step%250==0 or (resumed is not None and step==resumed['step']+1):
                save('latest',model,opt,config,epoch,step,seen,order,rng,best,None,full=True,
                     epoch_stats=dict(loss_sum=total,start_cursor=loss_start,
                                      elapsed_s=previous_elapsed+time.perf_counter()-training_started))
        assert seen==n
        out=ROOT/f'validation/epoch_{epoch}';out.mkdir(parents=True)
        (out/'README.md').write_text('Temporal validation only; original256validation episodes,8uniform frames each,all4task queries.\n'
            f'E124 epoch{epoch},step{step}; frozen TinyViT + learned task prompt + mask decoder.\n'
            'No actions, FM, H/K, flow steps or rollout. Configuration: ../../config.json. Metrics: summary.json.\n')
        metrics,records,packed=evaluate(model,validation,val_index,f'Temporal validation epoch{epoch}',save=out)
        sample_sheet(validation,val_index,records,packed,out)
        del records,packed
        assert frozen_hash(model)==frozen
        record=dict(epoch=epoch,step=step,frames_seen=seen,train_loss=total/n if loss_start==0 else None,
                    train_loss_segment=total/(n-loss_start),loss_start_cursor=loss_start,
                    val=metrics,epoch_seconds=time.perf_counter()-epoch_started)
        history.append(record);write(ROOT/'history.json',history);print(json.dumps(record),flush=True)
        if metrics['mean_iou']>best:
            best=metrics['mean_iou'];save('best',model,opt,config,epoch,step,seen,order,rng,best,metrics)
        save('latest',model,opt,config,epoch,step,seen,order,rng,best,metrics,full=True,
             epoch_stats=dict(loss_sum=total,start_cursor=loss_start,
                              elapsed_s=previous_elapsed+time.perf_counter()-training_started))
        write(ROOT/'progress.json',dict(status='epoch_complete',**record,total_epochs=EPOCHS))
    del loader,images,instances,validation,opt;gc.collect();torch.cuda.empty_cache()
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False);model.load_train_state(saved['model'])
    config.update(status='testing',best_epoch=saved['epoch'],best_step=saved['step'],training_seconds=previous_elapsed+time.perf_counter()-training_started)
    write(ROOT/'config.json',config);del saved
    results=test(model,split,config)
    assert frozen_hash(model)==frozen
    config.update(status='complete',elapsed_seconds=time.perf_counter()-started);write(ROOT/'config.json',config)
    write(ROOT/'verification.json',dict(status='passed',all_train_frames_once_per_epoch=True,train_frames=n,
        queries_per_epoch=4*n,epochs=EPOCHS,total_updates=step,frozen_encoder_unchanged=True,original_split_preserved=True))
    write(ROOT/'progress.json',dict(status='complete',epoch=EPOCHS,step=step,best_epoch=config['best_epoch'],tests=results))
    NOTE.write_text(NOTE.read_text().replace('Status: RUNNING.','Status: COMPLETE.')+'\nResults: '+json.dumps(results)+'\n')
    shutil.copyfile(NOTE,ROOT/'README.md')


if __name__=='__main__':main()
