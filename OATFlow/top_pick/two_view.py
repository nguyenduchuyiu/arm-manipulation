"""Fixed front/left calibrated RGB-D and target-cloud fusion for frozen E131."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.ndimage import binary_erosion,median_filter

from OATFlow.data_tools.collect_shuffle import BASE,NAMES
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.simulation.render import visual_model
from OATFlow.top_pick.cloud import DATA

VIEWS=('cam_front','cam_left')
LOOK_AT=np.array([.54,-.12,.035])
LEFT=np.array([0.,-.12,.60])
VOXEL=.003


def write_scene(path):
    tree=ET.parse(DATA/'scene.xml');world=tree.getroot().find('worldbody')
    front=world.find("camera[@name='cam_front']");left=ET.SubElement(world,'camera',dict(front.attrib))
    backward=LEFT-LOOK_AT;backward/=np.linalg.norm(backward)
    right=np.cross([0.,0.,1.],backward);right/=np.linalg.norm(right);up=np.cross(backward,right)
    left.set('name','cam_left');left.set('pos',' '.join(map(str,LEFT)))
    left.set('xyaxes',' '.join(map(str,np.r_[right,up])))
    tree.write(path)


def calibration(model,state,camera):
    cid=model.camera(camera).id;focal=120/np.tan(np.deg2rad(model.cam_fovy[cid])/2)
    K=np.array([[focal,0,159.5],[0,focal,119.5],[0,0,1]],np.float32)
    T=np.eye(4,dtype=np.float32);T[:3,:3]=state.cam_xmat[cid].reshape(3,3)@np.diag([1.,-1.,-1.]);T[:3,3]=state.cam_xpos[cid]-BASE
    return K,T


def filtered_points(codes,mask,K,T):
    assert mask.shape==codes.shape==(240,320) and mask.dtype==bool
    selected=binary_erosion(mask,iterations=1)&(codes>0)
    if selected.any():
        median=np.median(codes[selected])*.002
        selected&=np.abs(codes.astype(np.float64)*.002-median)<=.060
        smooth=median_filter(codes,size=3,mode='nearest')
        selected&=np.abs(codes.astype(np.int32)-smooth.astype(np.int32))*.002<=.010
    v,u=np.where(selected)
    if len(u)<8:return np.empty((0,3),np.float32)
    depth=codes[v,u].astype(np.float64)*.002
    local=np.c_[(u-K[0,2])*depth/K[0,0],(v-K[1,2])*depth/K[1,1],depth]
    return (local@T[:3,:3].T+T[:3,3]).astype(np.float32)


def fuse(points,rng):
    assert len(points)==2 and all(p.ndim==2 and p.shape[1]==3 for p in points)
    counts=[len(p) for p in points];xyz=np.concatenate(points)
    stats=dict(points_per_view=counts,usable_views=[n>=8 for n in counts],voxel_size_m=VOXEL)
    if not len(xyz):return np.zeros((128,3),np.float32),False,dict(stats,fused_points=0,both_view_voxels=0)
    keys=np.floor(xyz.astype(np.float64)/VOXEL).astype(np.int64)
    _,inverse=np.unique(keys,axis=0,return_inverse=True);n=np.bincount(inverse)
    merged=np.stack([np.bincount(inverse,weights=xyz[:,axis],minlength=len(n))/n for axis in range(3)],1).astype(np.float32)
    views=np.repeat([0,1],counts)
    source_counts=np.stack([np.bincount(inverse,weights=views==v,minlength=len(n)) for v in range(2)],1)
    assert int(source_counts.sum())==len(xyz)
    stats.update(fused_points=len(merged),both_view_voxels=int((source_counts>0).all(1).sum()))
    if len(merged)<8:return np.zeros((128,3),np.float32),False,stats
    ids=rng.choice(len(merged),128,replace=len(merged)<128)
    cloud=merged[ids];assert np.isfinite(cloud).all()
    return cloud,True,stats


class Rig:
    def __init__(self,path):
        self.model=visual_model(path,object_bodies=NAMES,lighting_scene=DATA/'scene.xml');clip_planes(self.model)
        self.model.vis.global_.offwidth=320;self.model.vis.global_.offheight=240
        self.state=mujoco.MjData(self.model);self.option=mujoco.MjvOption();self.option.geomgroup[3]=0
        self.renderer=mujoco.Renderer(self.model,height=240,width=320)
    def capture(self,qpos):
        self.state.qpos[:]=qpos;mujoco.mj_forward(self.model,self.state);result=[]
        for camera in VIEWS:
            self.renderer.update_scene(self.state,camera=camera,scene_option=self.option);rgb=self.renderer.render().copy()
            self.renderer.enable_depth_rendering();depth=self.renderer.render().copy();self.renderer.disable_depth_rendering()
            valid=np.isfinite(depth)&(depth>0)&(depth<=6.5535);codes=np.zeros((240,320),np.uint16)
            codes[valid]=np.rint(depth[valid]/.002).astype(np.uint16)
            K,T=calibration(self.model,self.state,camera)
            result.append(dict(rgb=rgb,depth=codes,K=K,T=T))
        return result
    def close(self):self.renderer.close()


class Observation:
    def __init__(self,rig,segment):self.rig=rig;self.segment=segment
    def __call__(self,qpos,target,rng):
        captured=self.rig.capture(qpos);masks,metadata=self.segment([v['rgb'] for v in captured],[target,target])
        points=[filtered_points(v['depth'],mask,v['K'],v['T']) for v,mask in zip(captured,masks)]
        cloud,good,stats=fuse(points,rng)
        stats['segmentation']=metadata
        return cloud,good,stats,np.stack(masks)
