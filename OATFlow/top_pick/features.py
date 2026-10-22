"""Cache only frozen top ViT tokens and aligned measured EE-delta labels."""
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import DataLoader,IterableDataset
from tqdm import tqdm

from OATFlow.simulation.pose import rotation_matrix,rotation_6d
from OATFlow.top_pick.collect import DATA,ROOT
from OATFlow.top_pick.data import SOURCE,VISION
from OATFlow.top_pick.model import TopPickPolicy


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def labels(row,starts):
    path=DATA/row['path']
    with np.load(path/'observation.npz') as z:
        state=z['joint_position'];ee=z['ee_position_base_m'];rot=rotation_matrix(z['ee_rotation_6d'])
    with np.load(path/'supervision.npz') as z:
        command=z['joint_command'];valid=z['action_valid']
    end=len(valid)-1
    assert valid[:end].all() and not valid[end]
    indices=starts[:,None]+np.arange(25)
    future=np.minimum(indices+1,end)
    action=np.empty((len(starts),25,10),np.float32)
    action[...,:3]=ee[future]-ee[starts,None]
    relative=rot[future]@np.swapaxes(rot[starts,None],-1,-2)
    action[...,3:9]=rotation_6d(relative)
    action[...,9]=command[np.minimum(indices,end-1),5]
    np.testing.assert_allclose(rotation_matrix(action[...,3:9])@rot[starts,None],rot[future],atol=2e-6,rtol=0)
    mask=indices<end
    return dict(state=state[starts].astype(np.float32),action=action,mask=mask,
                target=np.full(len(starts),row['target_id'],np.int64))


class Clips(IterableDataset):
    def __init__(self,jobs):self.jobs=jobs
    def __iter__(self):
        worker=torch.utils.data.get_worker_info()
        jobs=self.jobs[worker.id::worker.num_workers] if worker else self.jobs
        for row,starts,offset in jobs:
            selected={int(index):offset+j for j,index in enumerate(starts)}
            pixels=[];ids=[]
            with av.open(str(DATA/row['path']/'rgb_top.mp4')) as video:
                video.streams.video[0].thread_count=1
                for i,frame in enumerate(video.decode(video=0)):
                    if i in selected:
                        pixels.append(frame.to_ndarray(format='rgb24'));ids.append(selected[i])
                        if len(pixels)==64:
                            yield torch.from_numpy(np.stack(pixels)).permute(0,3,1,2),np.array(ids)
                            pixels=[];ids=[]
            if pixels:yield torch.from_numpy(np.stack(pixels)).permute(0,3,1,2),np.array(ids)


def main():
    out=ROOT/'cache';out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    jobs=[];values={k:[] for k in ('state','action','mask','target','is_train')};offset=0
    rng=np.random.default_rng(102)
    for row in tqdm(rows,desc='EE delta label alignment',mininterval=5):
        if row['evaluation_group'] not in ('train','val'):continue
        n=row['frames']-1
        starts=np.arange(n) if row['evaluation_group']=='train' else np.sort(rng.choice(n,min(8,n),replace=False))
        current=labels(row,starts)
        for key,value in current.items():values[key].append(value)
        values['is_train'].append(np.full(len(starts),row['evaluation_group']=='train',bool))
        jobs.append((row,starts,offset));offset+=len(starts)
    values={k:np.concatenate(v) for k,v in values.items()}
    np.savez(out/'labels.npz',**values)
    config=dict(status='encoding',frames=offset,train_frames=int(values['is_train'].sum()),
                val_frames=int((~values['is_train']).sum()),shape=[offset,64,768],dtype='float16',
                source=str(SOURCE),source_sha256=sha(SOURCE),vision=str(VISION),vision_sha256=sha(VISION),
                manifest_sha256=sha(DATA/'manifest.jsonl'),stride=1,horizon=25,
                endpoint='pre-step state t,measured EE state t+j+1,absolute commanded jaw t+j',
                normalization='preserved from reach checkpoint; no held-out fitted statistics',jobs=len(jobs))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'README.md').write_text('# e102_top cache\n\nFrozen ViT-B/16 top64x768 tokens, float16, '
        'all stride1 train frames plus eight deterministic validation frames/episode. '
        'Aligned H25 measured future EE-delta10D labels,terminal padding masked. '
        'No test frames or labels enter training/cache. Configuration/hashes: config.json. Status: encoding.\n')
    features=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float16,shape=(offset,64,768))
    model=TopPickPolicy().cuda().eval()
    assert not any(p.requires_grad for p in model.vision.parameters())
    loader=DataLoader(Clips(jobs),batch_size=None,num_workers=4,pin_memory=True,prefetch_factor=2)
    done=0;seen=np.zeros(offset,bool)
    with torch.inference_mode(),tqdm(total=offset,desc='Frozen top ViT frames',mininterval=5) as progress:
        for rgb,ids in loader:
            with torch.autocast('cuda',dtype=torch.bfloat16):x=model.encode(rgb.cuda(non_blocking=True))
            indices=ids.numpy()
            assert not seen[indices].any()
            features[indices]=x.float().cpu().numpy().astype(np.float16)
            seen[indices]=True
            done+=len(rgb);progress.update(len(rgb))
    assert done==offset and seen.all()
    features.flush();config['status']='complete'
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    p=out/'README.md';p.write_text(p.read_text().replace('Status: encoding','Status: complete'))


if __name__=='__main__':main()
