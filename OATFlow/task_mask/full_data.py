"""Full stride-one RGB/instance RAM cache; frozen encoder runs online on CUDA."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import defaultdict
from pathlib import Path
import time

import av
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from mobile_sam.utils.transforms import ResizeLongestSide

from OATFlow.task_mask.data import read
from OATFlow.top_pick.cloud import DATA

ROOT=Path('/workspace/robotics_assets/experiments/e124_mask_full')


def read_episode(row):
    directory=DATA/row['path']
    with np.load(directory/'masks.npz') as z:
        assert z['views'].tolist()==['front','wrist']
        labels=z['instance']
        assert labels.dtype==np.uint8
        instances=labels[:,0].copy()
    images=np.empty((row['frames'],240,320,3),np.uint8)
    count=0
    with av.open(str(directory/'rgb_front.mp4')) as video:
        video.streams.video[0].thread_count=1
        for count,image in enumerate(video.decode(video=0),1):
            images[count-1]=image.to_ndarray(format='rgb24')
    assert count==len(instances)==row['frames'],row['path']
    return images,instances


def native_cache(rows):
    n=sum(r['frames'] for r in rows)
    images=np.empty((n,240,320,3),np.uint8)
    instances=np.empty((n,240,320),np.uint8)
    offsets=np.concatenate(([0],np.cumsum([r['frames'] for r in rows])))
    started=time.perf_counter()
    # Bound completed episode buffers, rather than retaining every Future result.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(read_episode,rows[i]):i for i in range(min(4,len(rows)))}
        cursor=len(pending)
        with tqdm(total=len(rows),desc='Cache full RGB + instance labels',mininterval=10) as bar:
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    i=pending.pop(future);rgb,mask=future.result()
                    images[offsets[i]:offsets[i+1]]=rgb
                    instances[offsets[i]:offsets[i+1]]=mask
                    del rgb,mask
                    bar.update(1)
                    if cursor<len(rows):
                        pending[pool.submit(read_episode,rows[cursor])]=cursor;cursor+=1
    return images,instances,dict(frames=n,GiB=(images.nbytes+instances.nbytes)/2**30,
                                seconds=time.perf_counter()-started,offsets=offsets.tolist())


class Frames(Dataset):
    def __init__(self,images,instances):
        self.images=images;self.instances=instances;self.resize=ResizeLongestSide(1024)

    def __len__(self):return len(self.images)

    def __getitem__(self,i):
        # Preserve the original PIL SAM resize, including its uint8 rounding.
        pixels=self.resize.apply_image(self.images[i])
        return torch.from_numpy(pixels.copy()).permute(2,0,1),torch.from_numpy(self.instances[i]),i


def temporal_jobs(rows,frames_per_episode=8):
    jobs=[]
    for row in rows:
        for frame in np.unique(np.linspace(0,row['frames']-1,frames_per_episode).round().astype(int)):
            queries=[dict(row,target_id=t) for t in range(4)]
            jobs.append(dict(path=row['path'],frame=int(frame),rows=queries))
    return jobs


@torch.inference_mode()
def gpu_cache(model,jobs,description,batch_size=16):
    n=len(jobs);features=torch.empty((n,256,64,64),dtype=torch.float16)
    images=np.empty((n,240,320,3),np.uint8);instances=torch.empty((n,240,320),dtype=torch.uint8)
    by_path=defaultdict(list)
    for i,job in enumerate(jobs):by_path[job['path']].append((i,job['frame']))
    model.eval();started=time.perf_counter();pending=[]
    def encode_pending():
        indices=[i for i,_ in pending]
        pixels=torch.stack([torch.from_numpy(model.resize.apply_image(rgb).copy()).permute(2,0,1)
                            for _,rgb in pending]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            encoded=model.sam.image_encoder(model.sam.preprocess(pixels))
        features[indices]=encoded.cpu().to(torch.float16)
        pending.clear()
    for path,current in tqdm(by_path.items(),desc=description,mininterval=10):
        wanted=sorted({frame for _,frame in current})
        rgb,mask=read(DATA/path,wanted);lookup={f:i for i,f in enumerate(wanted)}
        for i,frame in current:
            image=rgb[lookup[frame]];images[i]=image;instances[i]=torch.from_numpy(mask[lookup[frame]])
            pending.append((i,image))
            if len(pending)==batch_size:encode_pending()
    if pending:encode_pending()
    return dict(features=features,images=images,instances=instances,jobs=jobs,
                elapsed_s=time.perf_counter()-started,
                GiB=(features.numel()*features.element_size()+images.nbytes+instances.numel())/2**30)
