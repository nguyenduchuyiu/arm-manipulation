"""Fuse frozen RGB ViT features with spatially aligned dense XYZ-map patches."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.top_pick.cloud import CloudPickPolicy,SOURCE
from OATFlow.top_pick.xyz_map import backproject

ROOT=Path('/workspace/robotics_assets/experiments/e120_fusion')


class FusionPolicy(CloudPickPolicy):
    def __init__(self,initialize=True):
        super().__init__(initialize=initialize)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(120)
            self.geometry_patch=nn.Conv2d(3,256,kernel_size=16,stride=16,bias=False)
            self.geometry_norm=nn.LayerNorm(256)
            self.fusion=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256))
        self.register_buffer('scene_mean',torch.zeros(3))
        self.register_buffer('scene_std',torch.ones(3))
        if initialize:
            saved=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
            self.scene_mean.copy_(saved['model']['xyz_mean']);self.scene_std.copy_(saved['model']['xyz_std'])

    def geometry_tokens(self,scene):
        xyz,valid=backproject(scene['depth'],scene['intrinsics'],scene['camera_to_base'])
        normalized=((xyz-self.scene_mean[None,:,None,None])/self.scene_std[None,:,None,None]).masked_fill(~valid,0)
        # Identical native geometry, 40px letterbox, 16px patches and 8x8 pooling as RGB.
        square=F.pad(normalized,(0,0,40,40))
        patches=self.geometry_patch(square)
        tokens=F.adaptive_avg_pool2d(patches,(8,8)).flatten(2).transpose(1,2)
        return self.geometry_norm(tokens)

    def visual_hidden(self,features,target_id,scene):
        rgb=self.visual(features)
        geometry=self.geometry_tokens(scene)
        memory=self.fusion(torch.cat((rgb,geometry),dim=-1))+self.position
        return self.decoder(self.target(target_id)[:,None],memory)[:,0]

    def conditioning(self,features,joints,target_id,scene):
        return self.context_from_hidden(self.visual_hidden(features,target_id,scene),joints)

    @torch.no_grad()
    def sample(self,features,joints,target_id,noise,steps=10,scene=None):
        assert scene is not None
        output=self.flow.sample_actions(noise,self.conditioning(features,joints,target_id,scene),steps)
        return torch.cat((output[...,:9],-.0255*(output[...,9:]+1)/2),dim=-1)

    @torch.no_grad()
    def predict_cloud(self,features,target_id,scene):
        return self.predicted_cloud(self.visual_hidden(features,target_id,scene))*self.cloud_std+self.cloud_mean

    def loss_parts(self,features,joints,target_id,actions,mask,cloud,cloud_valid,scene):
        hidden=self.visual_hidden(features,target_id,scene)
        values=torch.cat((actions[...,:9],-1-2*actions[...,9:]/.0255),dim=-1)
        noisy,time,target=self.flow.training_path(values)
        velocity=self.flow.velocity(noisy,time,self.context_from_hidden(hidden,joints))
        error=(velocity.float()-target.float()).square()
        flow=(error*mask[...,None]).sum()/(mask.sum()*10)
        prediction=self.predicted_cloud(hidden)
        if cloud_valid.any():
            a=prediction[cloud_valid];b=(cloud[cloud_valid]-self.cloud_mean)/self.cloud_std
            distances=torch.cdist(a,b).square()
            aux=.5*(distances.amin(1).mean()+distances.amin(2).mean())
            with torch.no_grad():
                metric=torch.cdist(a*self.cloud_std+self.cloud_mean,cloud[cloud_valid])
                error_mm=.5*(metric.amin(1).mean()+metric.amin(2).mean())*1000
        else:
            aux=prediction.sum()*0;error_mm=prediction.new_zeros(())
        return flow,aux,error_mm
