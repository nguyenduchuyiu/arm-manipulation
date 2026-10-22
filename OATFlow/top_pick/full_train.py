"""E122: train the400-patch RGB/XYZ policy on the original full training split."""
from collections import Counter
import gc
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import torch

from OATFlow.localization.model import state_hash
from OATFlow.top_pick.cloud import DATA,ROOT as SUBSET
from OATFlow.top_pick.cloud_data import action_labels,jobs_and_labels,write
from OATFlow.top_pick.cloud_train import cache,train
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.fusion_train import preflight
from OATFlow.top_pick.patch_fusion import PatchFusionPolicy

ROOT=Path('/workspace/robotics_assets/experiments/e122_full')
UPDATES=23520


def fresh():
    torch.manual_seed(102)
    return PatchFusionPolicy().cuda()


def partition():
    rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    manifests={k:[] for k in ('train','val','test')}
    for row in rows:
        group=row['evaluation_group']
        split=group if group in ('train','val') else 'test'
        manifests[split].append(dict(row,split=split))
    assert [len(manifests[k]) for k in ('train','val','test')]==[2048,256,1408]
    assert {r['layout_id'] for r in manifests['train']}==set(range(32))
    assert Counter(r['target_id'] for r in manifests['train'])=={k:512 for k in range(4)}
    assert Counter(r['evaluation_group'] for r in manifests['test'])==dict(composition=1024,spatial=256,combined=128)
    path_sets=[{r['path'] for r in manifests[k]} for k in ('train','val','test')]
    assert all(not path_sets[i]&path_sets[j] for i in range(3) for j in range(i+1,3))
    for split,rows in manifests.items():
        (ROOT/f'{split}_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(ROOT/'partitions.json',dict(status='complete',source_manifest_sha256=sha(DATA/'manifest.jsonl'),
        train_layouts=list(range(32)),train_episodes=2048,val_episodes=256,test_episodes=1408,
        test_groups=dict(Counter(r['evaluation_group'] for r in manifests['test'])),
        selection='original E117 full split; no original test/validation episode enters training'))
    return manifests


def baseline(manifests):
    source=ROOT.parent/'e120_fusion/test_k10/results.jsonl'
    results={r['path']:r for r in map(json.loads,source.read_text().splitlines())}
    common=[dict(results[r['path']],group=r['evaluation_group']) for r in manifests['test']]
    from OATFlow.top_pick.cloud_evaluate import summarize
    result=summarize(common,0)
    result.update(status='complete',source=str(source),source_sha256=sha(source),
        elapsed_s=None,test_manifest_sha256=sha(ROOT/'test_manifest.jsonl'),
        comparison='E120 final23520 reclassified on original1408held-out episodes only; architecture and training-set size differ')
    write(ROOT/'baseline_common_test.json',result)


def alignment(jobs,labels):
    current={row['path']:(starts,offset) for row,starts,offset in jobs}
    indices=[]
    for old in json.loads((SUBSET/'cache_jobs.json').read_text()):
        starts,offset=current[old['path']]
        np.testing.assert_array_equal(starts,old['starts'])
        indices.append(offset+np.arange(len(starts)))
    indices=np.concatenate(indices)
    with np.load(SUBSET/'labels.npz') as previous:
        for key in ('state','action','mask','target','raw_frame','is_train'):
            np.testing.assert_array_equal(previous[key],labels[key][indices])
        changed=int(np.any(previous['cloud']!=labels['cloud'][indices],axis=(1,2)).sum())
    write(ROOT/'label_alignment.json',dict(status='passed',overlap_frames=len(indices),
        previous_action_state_target_sampling_labels_bit_exact=True,new_train_episodes=1792,
        changed_cloud_frames=changed,cloud_change='3x3local depth-median outlier rejection10mm; sampling may change',
        validation_endpoints_unchanged=True,spatial_pooling=False,patch_tokens=400))


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    ROOT.mkdir(exist_ok=False);shutil.copyfile('docs/research/e122_README.md',ROOT/'README.md')
    manifests=partition();baseline(manifests)
    jobs,labels=jobs_and_labels(manifests)
    assert labels['is_train'].sum()==294640 and len(labels['is_train'])==296688
    write(ROOT/'cache_jobs.json',[dict(path=r['path'],starts=s.tolist(),offset=o) for r,s,o in jobs])

    # Check a real B128 update before committing to the full RAM cache.
    row,starts,_=jobs[0];starts=starts[:128]
    assert len(starts)==128
    model=fresh().eval()
    with TemporaryDirectory(prefix='e122_preflight_') as tmp:
        small,small_labels,mean,std=cache(model,[(row,starts,0)],action_labels(row,starts),
                                         artifact_root=Path(tmp),include_depth=True,compress_depth=True,clean_depth_labels=True,raw_rgb=True)
        small_depth_bytes=sum(map(len,small_labels['depth_zlib']))
        model.cloud_mean.copy_(torch.tensor(mean,device='cuda'));model.cloud_std.copy_(torch.tensor(std,device='cuda'))
        small=torch.from_numpy(small)
        small_labels={k:(v if k=='depth_zlib' else torch.from_numpy(v)) for k,v in small_labels.items()}
        preflight(model,small,small_labels,np.arange(128),artifact_root=ROOT)
    del model,small,small_labels;gc.collect();torch.cuda.empty_cache()

    n=len(labels['is_train']);features_bytes=n*3*240*320;depth_bytes=n*240*320*2
    depth_reserve=8*2**30
    extra_bytes=n*(128*3*4+9*4+16*4+1+4)
    meminfo={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
    cgroup=Path('/sys/fs/cgroup/memory')
    limit=int((cgroup/'memory.limit_in_bytes').read_text())
    usage=int((cgroup/'memory.usage_in_bytes').read_text())
    available=min(meminfo['MemAvailable'],limit-usage)
    required=features_bytes+depth_reserve+extra_bytes+8*2**30
    plan=dict(status='passed' if available>=required else 'insufficient_RAM',
        native_RGB_uint8_GiB=features_bytes/2**30,feature_cache=False,depth_uncompressed_GiB=depth_bytes/2**30,
        depth_estimated_compressed_GiB=small_depth_bytes/128*n/2**30,depth_reserve_GiB=8,
        extra_cache_GiB=extra_bytes/2**30,headroom_GiB=8,available_GiB=available/2**30,
        host_available_GiB=meminfo['MemAvailable']/2**30,cgroup_limit_GiB=limit/2**30,cgroup_usage_GiB=usage/2**30,
        required_GiB=required/2**30,feature_tokens=400,storage='native RGB uint8 plus lossless depth in CPU RAM; ViT online;no disk feature cache')
    write(ROOT/'memory_plan.json',plan);print(json.dumps(plan),flush=True)
    assert available>=required,'Insufficient container RAM for native RGB cache'

    model=fresh().eval();frozen=state_hash(model.vision)
    features,labels,mean,std=cache(model,jobs,labels,artifact_root=ROOT,include_depth=True,compress_depth=True,clean_depth_labels=True,raw_rgb=True)
    alignment(jobs,labels)
    features=torch.from_numpy(features)
    labels={k:(v if k=='depth_zlib' else torch.from_numpy(v)) for k,v in labels.items()}
    assert state_hash(model.vision)==frozen
    overrides=dict(experiment=ROOT.name,
        inputs='full front RGB + full depth + intrinsics/extrinsics + targetID + measured joints',
        architecture='RGB400native ViT patches + XYZ400aligned patches -> channel fusion400tokens -> two256D target-query decoder blocks ->960Dgoal + state ->FM',
        point_encoder='none;dense XYZ map -> trainable Conv3to256 patch16;NO spatial pooling',
        initialization='fresh E121 initial values:FM from E118best120k,ImageNet ViT frozen;no learned E119/E120/E121 weights;fresh optimizer',
        trainable='XYZ patch embedding,geometry norm,fusion,target-query decoder,400position embeddings,RGB/goal projections,cloud head,whole FM/state encoder',
        source_E121_weights=None,budget='23520 optimizer updates,B128,same E120/E121 budget;approximately10.217full-data epochs',
        comparison='replace four training layouts/256demos with all32training layouts/2048demos;keep400-token architecture and training hyperparameters',
        auxiliary_label_cleanup='reject isolated depth spikes >10mm from3x3median,GTmask only; no GT3Dpose filtering; input depth unchanged',
        geometry=dict(native_resolution=[240,320],depth_unit_m=.002,XYZ_frame='robot base',
            normalization='E118global train-only xyz mean/std',patch=16,letterbox=[40,40],patch_grid=[20,20],
            spatial_pooling=False,ViT_tokens=400,XYZ_tokens=400,decoder_memory_tokens=400,
            feature_projection='RGB768to256;channel concatenation512to256',
            query='single target query latent;aux cloud and FM read this latent',
            GTmask_input=False,input_pixel_sampling=False,invalid_depth='zero normalized XYZ'),
        cache='Native RGB uint8 63.66GiB plus lossless zlib uint16 depth in CPU RAM; frozen ViT online per batch;no disk feature cache',
        planned_rollout='final23520updates;original1408held-out episodes,H25/K10/Euler10,noise0,300frames,no videos',
        code_full_sha256={n:sha(Path(__file__).with_name(n)) for n in ['full_train.py','full_evaluate.py','patch_fusion.py','fusion.py','fusion_train.py','cloud_train.py','cloud_data.py','cloud_evaluate.py','xyz_map.py']})
    train(model,features,labels,mean,std,out=ROOT/'train',epochs=None,updates=UPDATES,
          artifact_root=ROOT,config_overrides=overrides)


if __name__=='__main__':main()
