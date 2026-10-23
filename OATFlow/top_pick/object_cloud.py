"""GT-visible target cloud input for the E131 point-cloud-only BC baseline."""
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion,median_filter

ROOT=Path('/workspace/robotics_assets/experiments/e131_object')
CACHE=ROOT.parent/'e122_full'
SOURCE=ROOT.parent/'e118_cloud/train/best.pt'


def extract_cloud(codes,mask,K,T,rng,unit=.002):
    """Same visible-cloud construction as the existing E122 cached GT labels."""
    selected=binary_erosion(mask,iterations=1)&(codes>0)
    if selected.any():
        median=np.median(codes[selected])*unit
        selected&=np.abs(codes.astype(np.float64)*unit-median)<=.060
        smooth=median_filter(codes,size=3,mode='nearest')
        selected&=np.abs(codes.astype(np.int32)-smooth.astype(np.int32))*unit<=.010
    v,u=np.nonzero(selected);count=len(u)
    if count<8:return np.zeros((128,3),np.float32),False,count
    chosen=rng.choice(count,128,replace=count<128);u=u[chosen];v=v[chosen]
    d=codes[v,u].astype(np.float64)*unit
    camera=np.c_[(u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d]
    return (camera@T[:3,:3].T+T[:3,3]).astype(np.float32),True,count
