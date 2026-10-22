"""E121: retain every native ViT/XYZ patch, with no spatial feature pooling."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.top_pick.fusion import FusionPolicy
from OATFlow.top_pick.xyz_map import backproject

ROOT=Path('/workspace/robotics_assets/experiments/e121_patches')


class PatchFusionPolicy(FusionPolicy):
    def __init__(self,initialize=True):
        super().__init__(initialize=initialize)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(121)
            self.position=nn.Parameter(torch.randn(1,400,256)*.02)

    @torch.no_grad()
    def encode(self,rgb):
        assert rgb.shape[1:]==(3,240,320)
        square=(self.rgb_mean*255).expand(len(rgb),3,320,320).clone()
        square[:,:,40:280]=rgb
        pixels=(square/255-self.rgb_mean)/self.rgb_std
        patches=self.vision._process_input(pixels)
        tokens=torch.cat((self.vision.class_token.expand(len(rgb),-1,-1),patches),dim=1)
        return self.vision.encoder(tokens)[:,1:]

    def geometry_tokens(self,scene):
        xyz,valid=backproject(scene['depth'],scene['intrinsics'],scene['camera_to_base'])
        normalized=((xyz-self.scene_mean[None,:,None,None])/self.scene_std[None,:,None,None]).masked_fill(~valid,0)
        patches=self.geometry_patch(F.pad(normalized,(0,0,40,40)))
        return self.geometry_norm(patches.flatten(2).transpose(1,2))
