"""Repeat E120 with400native spatial patches, all cached features in CPU RAM."""
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from OATFlow.localization.model import state_hash
from OATFlow.top_pick.cloud import ROOT as PREVIOUS
from OATFlow.top_pick.cloud_data import jobs_and_labels,write
from OATFlow.top_pick.cloud_train import cache,train
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.fusion_train import preflight,online_preflight
from OATFlow.top_pick.patch_fusion import ROOT,PatchFusionPolicy


def fresh():
    torch.manual_seed(102)
    return PatchFusionPolicy().cuda()


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    ROOT.mkdir(exist_ok=False);shutil.copyfile('docs/research/e121_README.md',ROOT/'README.md')
    for name in ['partitions.json','train_manifest.jsonl','val_manifest.jsonl','test_manifest.jsonl',
                 *[f'part_{k:02d}.jsonl' for k in range(1,9)]]:
        shutil.copyfile(PREVIOUS/name,ROOT/name)
    manifests={k:[json.loads(s) for s in (ROOT/f'{k}_manifest.jsonl').read_text().splitlines()]
               for k in ('train','val','test')}
    jobs,labels=jobs_and_labels(manifests)
    write(ROOT/'cache_jobs.json',[dict(path=r['path'],starts=s.tolist(),offset=o) for r,s,o in jobs])
    model=fresh().eval();features,labels,mean,std=cache(model,jobs,labels,artifact_root=ROOT,include_depth=True)
    with np.load(PREVIOUS/'labels.npz') as old,np.load(ROOT/'labels.npz') as new:
        for key in old.files:np.testing.assert_array_equal(old[key],new[key])
    write(ROOT/'label_alignment.json',dict(status='passed',E119_E120_action_cloud_labels_bit_exact=True,
        patch_tokens=400,spatial_pooling=False,features_bytes=features.nbytes,
        subset_sha256=sha(ROOT/'partitions.json'),feature_storage='CPU RAM;batch features copied to GPU'))
    features=torch.from_numpy(features)
    labels={k:(v if k=='depth' else torch.from_numpy(v)) for k,v in labels.items()}
    model.cloud_mean.copy_(torch.tensor(mean,device='cuda'));model.cloud_std.copy_(torch.tensor(std,device='cuda'))
    frozen=state_hash(model.vision);ids=np.flatnonzero(labels['is_train'].numpy())[:128]
    preflight(model,features,labels,ids,artifact_root=ROOT)
    online_preflight(model,manifests['train'],artifact_root=ROOT)
    del model;gc.collect();torch.cuda.empty_cache();model=fresh()
    assert state_hash(model.vision)==frozen
    overrides=dict(experiment='e121_patches',
        inputs='full front RGB + full depth + intrinsics/extrinsics + targetID + measured joints',
        architecture='RGB400native ViT patches + XYZ400aligned patches -> channel fusion400tokens -> two256D target-query decoder blocks ->960Dgoal + state ->FM',
        point_encoder='none;dense XYZ map -> fresh trainable Conv3to256 patch16;NO spatial pooling',
        initialization='FM from E118best120k;ImageNet ViT frozen;all E120shared-module initial values retained;fresh400position embeddings;no E120 learned weights',
        trainable='XYZ patch embedding,geometry norm,fusion,target-query decoder,400position embeddings,RGB/goal projections,cloud head,whole FM/state encoder',
        source_E120_weights=None,comparison='same E120data,seed102,80epochs,B128,LRs,loss,validation;remove20x20to8x8spatial pooling in BOTH branches',
        geometry=dict(native_resolution=[240,320],depth_unit_m=.002,XYZ_frame='robot base',
            normalization='E118global train-only xyz mean/std',patch=16,letterbox=[40,40],patch_grid=[20,20],
            spatial_pooling=False,ViT_tokens=400,XYZ_tokens=400,decoder_memory_tokens=400,
            feature_projection='RGB768to256 retained;channel concatenation512to256 retained',
            query='same single target query latent;aux cloud and FM read this latent',
            GTmask_input=False,input_pixel_sampling=False,invalid_depth='zero normalized XYZ'),
        cache='RGB FP16 native features22.65GiB plus raw uint16 depth5.66GiB in CPU RAM;transfer each batch only;no disk feature cache',
        code_patch_sha256={n:sha(Path(__file__).with_name(n)) for n in ['patch_fusion.py','patch_train.py','fusion.py','fusion_train.py','cloud_train.py','xyz_map.py']})
    train(model,features,labels,mean,std,out=ROOT/'train',epochs=80,artifact_root=ROOT,config_overrides=overrides)


if __name__=='__main__':main()
