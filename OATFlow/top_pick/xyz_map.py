"""Calibrated dense front-depth geometry, shared by training and online rollout."""
import numpy as np
import torch

BASE=np.array([.539369022999,.0639727628473,0.])
DEPTH_UNIT=.002


def backproject(depth,intrinsics,camera_to_base):
    """Raw uint16 depth codes -> [B,3,240,320] metric XYZ in the robot base."""
    assert depth.shape[1:]==(240,320)
    # Geometry stays FP32 even inside mixed-precision policy execution.
    with torch.autocast(device_type=depth.device.type,enabled=False):
        z=depth.float()*DEPTH_UNIT;K=intrinsics.float();T=camera_to_base.float()
        v,u=torch.meshgrid(torch.arange(240,device=z.device),torch.arange(320,device=z.device),indexing='ij')
        x=(u-K[:,0,2,None,None])*z/K[:,0,0,None,None]
        y=(v-K[:,1,2,None,None])*z/K[:,1,1,None,None]
        # Elementwise rotation avoids TF32 matmul rounding in metric geometry.
        xyz=torch.stack([T[:,i,0,None,None]*x+T[:,i,1,None,None]*y+
                         T[:,i,2,None,None]*z+T[:,i,3,None,None] for i in range(3)],1)
        valid=(depth>0)[:,None]
        return xyz.masked_fill(~valid,0),valid


def camera_calibration(model,state):
    cid=model.camera('cam_front').id
    focal=120/np.tan(np.deg2rad(model.cam_fovy[cid])/2)
    K=np.array([[focal,0,159.5],[0,focal,119.5],[0,0,1.]],np.float32)
    transform=np.eye(4,dtype=np.float32)
    transform[:3,:3]=state.cam_xmat[cid].reshape(3,3)@np.diag([1.,-1.,-1.])
    transform[:3,3]=state.cam_xpos[cid]-BASE
    return K,transform
