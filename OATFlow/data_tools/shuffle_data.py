"""Load shared-context e96 episodes without duplicating their stored prefix."""
import json
from pathlib import Path

import numpy as np


def arrays(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def episode(root,ref):
    path=Path(root)/ref
    meta=json.loads((path/'episode.json').read_text())
    result={'metadata':meta,'rgb_segments':{v:[path.parent/f'rgb_{v}.mp4',path/f'rgb_{v}.mp4']
                                        for v in ('top','left','right')}}
    for name in ('observation','supervision','labels'):
        first=arrays(path.parent/f'{name}.npz');last=arrays(path/f'{name}.npz')
        result[name]={k:np.concatenate((first[k],last[k])) if k!='entity_names' else last[k] for k in last}
    return result


def query_masks(root,ref):
    path=Path(root)/ref
    meta=json.loads((path/'episode.json').read_text())
    target=meta['target_id']
    first=arrays(path.parent/'masks.npz');last=arrays(path/'masks.npz')
    n=len(first['timestamp_s'])
    packed=np.zeros((n,3,3,320,40),np.uint8)
    packed[:,:,0]=first['amodal_packed'][:,:,target]
    for channel,key in ((1,'frontmost_cover_ids'),(2,'container_ids')):
        values=first[key][:,:,target] if channel==1 else np.broadcast_to(first[key][:,None,target],(n,3))
        for cover in (5,6):
            frames,views=np.nonzero(values==cover)
            packed[frames,views,channel]=first['amodal_packed'][frames,views,cover-1]
    semantic=first['semantic_entity'][:,target]
    owner=('RedCube','BlueCube','GreenCube','YellowCube','cover_a','cover_b').index(meta['selected_cover'])+1
    relevance=np.where(semantic[:,None,None,None]==2,(first['instance']==target+1)*2,
                       (first['instance']==owner)*(semantic[:,None,None,None]==1)).astype(np.uint8)
    return dict(tcow_packed=np.concatenate((packed,last['tcow_packed'])),
                relevance=np.concatenate((relevance,last['relevance'])),
                semantic_entity=np.concatenate((semantic,last['semantic_entity'][:,0])),
                timestamp_s=np.concatenate((first['timestamp_s'],last['timestamp_s'])))
