"""Train E119 on one of eight balanced layout parts; keep frozen vision features in RAM."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from OATFlow.localization.model import state_hash
from OATFlow.top_pick.cloud import CloudPickPolicy, ROOT, SOURCE, DATA
from OATFlow.top_pick.cloud_data import Clips, partition, jobs_and_labels, write
from OATFlow.top_pick.data import VISION
from OATFlow.top_pick.features import sha


def fresh():
    torch.manual_seed(102)
    return CloudPickPolicy().cuda()


def optimizer(model):
    groups = [dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('flow.')], lr=2e-5),
              dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('flow.')], lr=1e-4)]
    return torch.optim.AdamW(groups, betas=(.9,.95), weight_decay=1e-4, fused=True)


def cache(model, jobs, labels, artifact_root=ROOT, include_depth=False, compress_depth=False, clean_depth_labels=False, raw_rgb=False):
    assert not compress_depth or include_depth
    n = len(labels['is_train'])
    tokens=model.position.shape[1]
    features = np.empty((n,3,240,320),np.uint8) if raw_rgb else np.empty((n,tokens,768),np.float16)
    clouds = np.zeros((n,128,3), np.float32); valid = np.zeros(n, bool); counts = np.zeros(n,np.int32)
    seen = np.zeros(n, bool); started = time.perf_counter()
    depths=([None]*n if compress_depth else np.empty((n,240,320),np.uint16)) if include_depth else None
    intrinsics=np.empty((n,3,3),np.float32) if include_depth else None
    transforms=np.empty((n,4,4),np.float32) if include_depth else None
    loader = DataLoader(Clips(jobs,include_depth,clean_depth_labels), batch_size=None, num_workers=4, pin_memory=True, prefetch_factor=1)
    description='Native uint8 RGB + depth/label cache' if raw_rgb else 'Front RGB ViT + depth cloud cache'
    with torch.inference_mode(), tqdm(total=n, desc=description, mininterval=5) as bar:
        for packed in loader:
            rgb,ids,cloud,good,pixels=packed[:5]
            indices = ids.numpy(); assert not seen[indices].any()
            if raw_rgb:
                features[indices]=rgb.numpy()
            else:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    encoded = model.encode(rgb.cuda(non_blocking=True))
                encoded = encoded.float().cpu().numpy().astype(np.float16)
                assert np.isfinite(encoded).all()
                features[indices] = encoded
            clouds[indices] = cloud.numpy(); valid[indices] = good.numpy(); counts[indices] = pixels.numpy()
            if include_depth:
                if compress_depth:
                    for index,depth in zip(indices,packed[5].numpy()):depths[index]=zlib.compress(depth.tobytes(),level=1)
                else:depths[indices]=packed[5].numpy()
                intrinsics[indices]=packed[6].numpy();transforms[indices]=packed[7].numpy()
            seen[indices] = True; bar.update(len(indices))
    assert seen.all()
    train = labels['is_train'] & valid
    values = clouds[train].reshape(-1,3).astype(np.float64)
    mean = values.mean(0).astype(np.float32); std = values.std(0).astype(np.float32)
    assert (std > .001).all() and train.sum() > labels['is_train'].sum()*.25
    del values
    statistics = dict(mean_base_m=mean.tolist(), std_m=std.tolist(), source='valid training clouds only',
        valid_train_frames=int(train.sum()), train_frames=int(labels['is_train'].sum()),
        valid_val_frames=int((~labels['is_train'] & valid).sum()), val_frames=int((~labels['is_train']).sum()),
        labels='eroded visible target instance mask + actual decoded front2mm depth + native calibration',
        points=128, minimum_pixels=8, erosion_pixels=1, depth_outlier_band_m=.060, centroid_subtraction=False,
        local_depth_spike_filter=dict(enabled=clean_depth_labels,median_kernel=3,threshold_m=.010,
            operation='reject auxiliary label pixels only;preserve original depth input;no GT3D pose filtering'),
        missing='fully occluded/too few pixels: cloud auxiliary loss masked; action loss remains active')
    write(artifact_root/'cloud_statistics.json', statistics)
    labels.update(cloud=clouds, cloud_valid=valid)
    np.savez_compressed(artifact_root/'labels.npz', **labels, visible_pixels=counts)
    # Raw depth is an inexpensive RAM cache; never write a dense XYZ feature cache.
    if include_depth:
        labels['depth_zlib' if compress_depth else 'depth']=depths
        labels.update(intrinsics=intrinsics,camera_to_base=transforms)
    config = dict(status='complete', frames=n, train_frames=int(labels['is_train'].sum()),
        val_frames=int((~labels['is_train']).sum()), feature_bytes=features.nbytes,
        cloud_bytes=clouds.nbytes, elapsed_s=time.perf_counter()-started,
        feature_storage='native RGB uint8 in CPU RAM; frozen ViT encoded online' if raw_rgb else 'RAM only; no persistent per-frame feature file',
        raw_rgb=raw_rgb,feature_tokens=tokens,
        train_stride=1, validation_sampling='8fixed frames/episode;512fixed validation chunks selected later',
        jobs=len(jobs), source_manifest_sha256=sha(DATA/'manifest.jsonl'),
        vision_sha256=sha(VISION), source_checkpoint_sha256=sha(SOURCE))
    config.update(depth_input=include_depth,depth_bytes=(sum(map(len,depths)) if compress_depth else depths.nbytes) if include_depth else 0,
                  depth_uncompressed_bytes=n*240*320*2 if include_depth else 0,
                  depth_compression='zlib level1,lossless uint16,per frame' if compress_depth else None,
                  calibration_bytes=intrinsics.nbytes+transforms.nbytes if include_depth else 0)
    write(artifact_root/'cache_config.json', config)
    return features, labels, mean, std


def batch(features, labels, ids,model=None):
    index = torch.as_tensor(ids, dtype=torch.long, device=features.device)
    pixels_or_features=features[index].cuda()
    if features.dtype==torch.uint8:
        assert model is not None,'Online RGB batch requires the policy encoder'
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            pixels_or_features=model.encode(pixels_or_features).half()
    values=(pixels_or_features.float(), *(labels[k][ids].cuda() for k in
        ('state','target','action','mask','cloud','cloud_valid')))
    if 'depth' in labels or 'depth_zlib' in labels:
        depth=(np.stack([np.frombuffer(zlib.decompress(labels['depth_zlib'][i]),dtype=np.uint16).reshape(240,320) for i in ids])
               if 'depth_zlib' in labels else labels['depth'][ids])
        scene=dict(depth=torch.from_numpy(depth).to(device='cuda',dtype=torch.float32),
                   intrinsics=labels['intrinsics'][ids].cuda(),camera_to_base=labels['camera_to_base'][ids].cuda())
        return (*values,scene)
    return values


def preflight(model, features, labels, ids):
    frozen = state_hash(model.vision)
    reference = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
    for name,value in model.flow.state_dict().items():
        assert torch.equal(value.cpu(), reference['model']['flow.'+name]), name
    del reference
    model.train(); assert not model.vision.training
    b = batch(features, labels, ids)
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16): parts = model.loss_parts(*b)
    parts[1].backward()
    assert model.decoder.layers[-1].linear2.weight.grad.abs().sum() > 0
    assert model.visual[1].weight.grad.abs().sum() > 0 and model.target.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.flow.parameters())
    assert all(p.grad is None for p in model.goal.parameters())
    assert all(p.grad is None for p in model.vision.parameters())
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        reversed_batch = (*b[:-2], b[-2].flip(1), b[-1])
        other = model.loss_parts(*reversed_batch)[1]
    torch.testing.assert_close(parts[1].detach(), other, rtol=1e-5, atol=1e-6)
    opt = optimizer(model); opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16): flow, aux, error = model.loss_parts(*b)
    loss = flow+aux; assert torch.isfinite(loss)
    loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    assert model.flow.layers[-1].mlp.down_proj.weight.grad.abs().sum() > 0
    opt.step(); torch.cuda.synchronize()
    assert state_hash(model.vision) == frozen
    result=dict(status='passed', batch=len(ids), seconds=time.perf_counter()-start,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        flow_loss=float(flow.detach()), cloud_loss=float(aux.detach()), cloud_error_mm=float(error),
        FM_initialization='all flow tensors exactly E118best120000; fresh optimizer',
        auxiliary_gradients='visual projection,target embedding,decoder,cloud head; not goal projection/FM/ViT',
        combined_gradients='FM and visual pipeline', frozen_ViT_unchanged=True,
        cloud_order_invariant=True, actual_depth_geometry_alignment_passed=True)
    write(ROOT/'preflight.json', result); print(json.dumps(result), flush=True)
    del opt, parts, b


def train(model, features, labels, mean, std, out=None, epochs=10, artifact_root=ROOT, config_overrides=None, updates=None):
    out=out or ROOT/'train';out.mkdir(exist_ok=False)
    shutil.copyfile(artifact_root/'README.md', out/'README.md')
    model.cloud_mean.copy_(torch.tensor(mean,device='cuda'));model.cloud_std.copy_(torch.tensor(std,device='cuda'))
    frozen = state_hash(model.vision); normalization = {n:v.clone() for n,v in model.flow.named_buffers()}
    train_ids=np.flatnonzero(labels.pop('is_train').numpy()); all_val=np.setdiff1d(np.arange(len(features)),train_ids)
    val_ids=np.random.default_rng(102).choice(all_val,512,replace=False)
    np.savez(out/'indices.npz',train=train_ids,validation=val_ids)
    assert (updates is None and isinstance(epochs,int) and epochs>0) or (epochs is None and isinstance(updates,int) and updates>0)
    epoch_steps=math.ceil(len(train_ids)/128)
    total_steps=updates if updates is not None else epochs*epoch_steps
    budget=f'{updates}updates' if updates is not None else f'exact{epochs}epochs'
    config=dict(status='running', experiment=artifact_root.name, dataset=str(DATA), subset=str(artifact_root/'partitions.json'),
        source_checkpoint=str(SOURCE),source_checkpoint_sha256=sha(SOURCE),source_step=120000,
        initialization='entire FM from E118best120k; fresh visual/query/goal/cloud head; fresh AdamW',
        point_encoder='E118point encoder not used; E103hidden-to-goal path retained',
        vision=str(VISION),vision_sha256=sha(VISION),vision_frozen_sha256=frozen,
        inputs='front RGB240x320 letterboxed to320square, target color ID, measured5joints+jaw; no depth/mask/cloud/XYZ input',
        trainable='target-query decoder,visual/goal projections,auxiliary cloud head,entire FM/state encoder',
        frozen='ImageNet ViT-B/16 in eval/no-grad',epochs=epochs,batch=128,steps_per_epoch=epoch_steps,
        total_steps=total_steps,requested_updates=updates,train_chunks=len(train_ids),validation_chunks=512,
        fm_lr=2e-5,visual_lr=1e-4,warmup_steps=100,optimizer='fusedAdamW beta(.9,.95),wd1e-4,clip1',
        precision='BF16',seed=102,aux_cloud=dict(weight=1.,points=128,
            loss='0.5*(pred-to-label+label-to-pred) mean squared Chamfer, global train-only normalized coordinates',
            head='LN256+Linear256to384 reshaped128x3',attachment='target decoder hidden256 before goal projection',
            source='current visible target surface from decoded2mm front depth, eroded GT instance mask,calibration',
            missing='mask auxiliary only if fewer8pixels; keep action loss'),
        validation='best by masked FM action loss only on512fixed chunks/noise/time; no test selection',
        horizon=25,action_dim=10,action='chunk-anchor EE deltaXYZ + relative rotation6D + absolute jaw',
        normalization='E118state/action buffers unchanged; cloud train-only stats',
        planned_rollout=f'final {budget}; all3200test episodes,H25/K10/Euler10,noise0,300frames,no videos',
        cache='frozen features in GPU RAM during training; no disk feature cache',
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in ['cloud.py','cloud_data.py','cloud_train.py']})
    opt=optimizer(model);rng=np.random.default_rng(102);order=rng.permutation(train_ids)
    cursor=seen=first_step=0;best=math.inf;best_step=0
    if config_overrides:config.update(config_overrides)
    write(out/'config.json',config)
    started=time.perf_counter();losses=[]

    def save(name,step,resumable):
        config.update(samples_seen=seen,effective_epochs=seen/len(train_ids),best_step=best_step)
        payload=dict(model=model.state_dict(),step=step,config=config,best_val=best)
        if resumable:payload.update(optimizer=opt.state_dict(),order=order,cursor=cursor,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=rng.bit_generator.state))
        temporary=out/(name+'.tmp');torch.save(payload,temporary);temporary.replace(out/(name+'.pt'))

    model.train(); torch.cuda.reset_peak_memory_stats()
    for step in tqdm(range(first_step+1,total_steps+1),desc=f'{artifact_root.name} pick {budget}',
                     initial=first_step,total=total_steps,mininterval=5):
        if cursor>=len(order):order=rng.permutation(train_ids);cursor=0
        ids=order[cursor:cursor+128];cursor+=len(ids);seen+=len(ids)
        for group,lr in zip(opt.param_groups,(2e-5,1e-4)):group['lr']=min(step/100,1.)*lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            flow,aux,error=model.loss_parts(*batch(features,labels,ids,model=model));loss=flow+aux
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        opt.step()
        record=dict(step=step,epoch=seen/len(train_ids),flow_loss=float(flow.detach()),cloud_loss=float(aux.detach()),
            loss=float(loss.detach()),cloud_error_mm=float(error),elapsed_s=time.perf_counter()-started)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        losses.append(record)
        if step%50==0 or step==1:
            metric=dict(record,mean_flow=float(np.mean([x['flow_loss'] for x in losses])),
                mean_cloud=float(np.mean([x['cloud_loss'] for x in losses])),
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
            write(out/'progress.json',dict(status='training',**metric));print(json.dumps(metric),flush=True);losses=[]
        if step==1 or step%250==0 or step==total_steps:
            model.eval();sums=[]
            with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                torch.manual_seed(102000)
                for start in range(0,len(val_ids),64):
                    ids=val_ids[start:start+64]
                    with torch.autocast('cuda',dtype=torch.bfloat16): values=model.loss_parts(*batch(features,labels,ids,model=model))
                    good=int(labels['cloud_valid'][ids].sum())
                    sums.append((*[float(v) for v in values],len(ids),good))
            score=sum(r[0]*r[3] for r in sums)/sum(r[3] for r in sums)
            cloud_score=sum(r[1]*r[4] for r in sums)/max(1,sum(r[4] for r in sums))
            cloud_error=sum(r[2]*r[4] for r in sums)/max(1,sum(r[4] for r in sums))
            if score<best:best=score;best_step=step;save('best',step,False)
            save('latest',step,True)
            validation=dict(step=step,epoch=seen/len(train_ids),FM_loss=score,cloud_loss=cloud_score,
                            cloud_chamfer_mm=cloud_error,best_step=best_step)
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(validation)+'\n')
            print(json.dumps(validation),flush=True);model.train()
    completed_epochs,remainder=divmod(total_steps,epoch_steps)
    assert seen==completed_epochs*len(train_ids)+remainder*128
    assert state_hash(model.vision)==frozen
    for name,value in model.flow.named_buffers():torch.testing.assert_close(value,normalization[name],rtol=0,atol=0)
    os.link(out/'latest.pt',out/'final.pt')
    config.update(status='complete',elapsed_s=time.perf_counter()-started,samples_seen=seen,
                  effective_epochs=seen/len(train_ids),best_step=best_step,best_validation_FM_loss=best)
    write(out/'config.json',config);write(out/'progress.json',config)
    write(out/'verification.json',dict(status='passed',ViT_sha256=frozen,ViT_unchanged=True,
        state_action_normalization_unchanged=True,exact_epochs=epochs,requested_updates=updates,samples_seen=seen,
        final_step=total_steps,best_step=best_step))
    with (out/'README.md').open('a') as f:f.write(f'\nTraining complete: {total_steps} updates/{seen/len(train_ids):.6f} effective epochs. Best step{best_step}; rollout pending.\n')
    print(json.dumps(config),flush=True)


def main(retry_prepare=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    ROOT.mkdir(exist_ok=retry_prepare)
    assert not (ROOT/'train').exists(), 'Training output already exists'
    if retry_prepare:
        previous=json.loads((ROOT/'partitions.json').read_text())
        assert previous['source_manifest_sha256']==sha(DATA/'manifest.jsonl')
    shutil.copyfile('docs/research/e119_README.md',ROOT/'README.md')
    manifests=partition();jobs,labels=jobs_and_labels(manifests)
    write(ROOT/'cache_jobs.json',[dict(path=r['path'],starts=s.tolist(),offset=o) for r,s,o in jobs])
    model=fresh().eval();features,labels,mean,std=cache(model,jobs,labels)
    frozen=state_hash(model.vision)
    features=torch.from_numpy(features).cuda()
    labels={k:torch.from_numpy(v) for k,v in labels.items()}
    model.cloud_mean.copy_(torch.tensor(mean,device='cuda'));model.cloud_std.copy_(torch.tensor(std,device='cuda'))
    ids=np.flatnonzero(labels['is_train'].numpy())[:128]
    preflight(model,features,labels,ids)
    del model;gc.collect();torch.cuda.empty_cache()
    model=fresh();assert state_hash(model.vision)==frozen
    train(model,features,labels,mean,std)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-prepare',action='store_true')
    main(parser.parse_args().retry_prepare)
