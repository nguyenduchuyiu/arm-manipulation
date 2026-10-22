"""Predict a selected object's base XYZ from one top-camera RGB image."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from OATFlow.localization.model import FrozenViT,ObjectXYZ


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--id',type=int,choices=range(4),required=True,help='red/blue/green/yellow=0/1/2/3')
    parser.add_argument('--checkpoint',type=Path,default=Path('/workspace/robotics_assets/models/object_localization_topview/best.pt'))
    args=parser.parse_args();torch.set_num_threads(4)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    count=saved['config']['tokens_per_view']
    encoder=FrozenViT(tokens=count).cuda().eval();head=ObjectXYZ(tokens=count).cuda().eval();head.load_state_dict(saved['model'],strict=True)
    pixels=np.asarray(Image.open(args.image).convert('RGB')).copy()
    if pixels.shape!=(320,320,3):raise ValueError('expected320x320top RGB')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        tokens=encoder(torch.from_numpy(pixels[None]).cuda()).reshape(1,1,count,768)
        xyz=head(tokens,torch.tensor([args.id],device='cuda')).float()
    xyz=xyz*torch.as_tensor(saved['std'],device='cuda')+torch.as_tensor(saved['mean'],device='cuda')
    print(json.dumps(dict(object_id=args.id,xyz_base_m=xyz[0,0].cpu().tolist(),checkpoint_step=saved['step'])))


if __name__=='__main__':main()
