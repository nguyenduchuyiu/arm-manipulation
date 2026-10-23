"""Canonical home observations, four task queries and RAM-only encoder cache."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import av
import numpy as np
import torch
from tqdm import tqdm

from OATFlow.task_mask.model import TaskMask
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.full_train import ROOT as PICK

ROOT=PICK.parent/'e123_mask'
MODEL=None


def read(path,frames):
    selected=[];wanted=set(frames)
    with av.open(str(path/'rgb_front.mp4')) as video:
        video.streams.video[0].thread_count=1
        for i,image in enumerate(video.decode(video=0)):
            if i in wanted:selected.append((i,image.to_ndarray(format='rgb24')))
            if i>=max(wanted):break
    assert [i for i,_ in selected]==sorted(wanted),str(path)
    with np.load(path/'masks.npz') as z:
        assert z['views'].tolist()==['front','wrist']
        instance=z['instance'][sorted(wanted),0].copy()
    return [rgb for _,rgb in selected],instance


def manifests():
    result={split:list(map(json.loads,(PICK/f'{split}_manifest.jsonl').read_text().splitlines()))
            for split in ('train','val','test')}
    assert [len(result[k]) for k in result]==[2048,256,1408]
    train_keys={(r['layout_id'],r['permutation']) for r in result['train']}
    assert all(not train_keys & {(r['layout_id'],r['permutation']) for r in result[s]} for s in ('val','test'))
    return result


def groups(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[(row['layout_id'],row['permutation'])].append(row)
    result=[]
    for key,current in sorted(grouped.items()):
        assert sorted(r['target_id'] for r in current)==[0,1,2,3]
        current.sort(key=lambda r:r['target_id'])
        result.append(dict(key=list(key),rows=current,path=current[0]['path'],frame=0))
    return result


def initialize():
    global MODEL
    torch.set_num_threads(8);torch.manual_seed(123)
    MODEL=TaskMask().eval()


def encode_group(job):
    start=time.perf_counter();path=DATA/job['path']
    rgb,instances=read(path,[job['frame']]);rgb=rgb[0];instance=instances[0]
    if job.get('check_initial',False):
        arrays=[]
        for row in job['rows']:
            with np.load(DATA/row['path']/'sim_state.npz') as z:
                arrays.append(tuple(z[k] for k in ('initial_qpos','initial_qvel','initial_ctrl')))
        for value in arrays[1:]:
            for a,b in zip(arrays[0],value):np.testing.assert_array_equal(a,b)
    with torch.inference_mode():embedding=MODEL.encode(rgb)[0].numpy().astype(np.float16)
    return embedding,rgb,instance,time.perf_counter()-start


def cache(jobs,description):
    n=len(jobs);features=np.empty((n,256,64,64),np.float16)
    images=np.empty((n,240,320,3),np.uint8);instances=np.empty((n,240,320),np.uint8)
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        futures={pool.submit(encode_group,job):i for i,job in enumerate(jobs)}
        for future in tqdm(as_completed(futures),total=n,desc=description,mininterval=10):
            i=futures[future];embedding,rgb,instance,seconds=future.result()
            features[i]=embedding;images[i]=rgb;instances[i]=instance
    return dict(features=torch.from_numpy(features),images=images,instances=torch.from_numpy(instances),
                jobs=jobs,elapsed_s=time.perf_counter()-start,GiB=(features.nbytes+images.nbytes+instances.nbytes)/2**30)


def query_index(cached):
    result=[]
    for i,job in enumerate(cached['jobs']):
        for row in job['rows']:
            result.append(dict(image=i,target=row['target_id'],path=row['path'],
                group=row['evaluation_group'],layout=row['layout_id'],permutation=row['permutation'],frame=job['frame']))
    return result


def batch(cached,index,chosen):
    images=torch.tensor([index[i]['image'] for i in chosen])
    target=torch.tensor([index[i]['target'] for i in chosen])
    features=cached['features'][images].float()
    instance=cached['instances'][images]
    truth=instance==target[:,None,None]+1
    return features,target,truth,instance


def frozen_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.sam.image_encoder.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()
