"""Auxiliary current-object-center supervision at the 256D target decoder hidden."""
import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.top_pick.model import TopPickPolicy
from OATFlow.top_pick.collect import ROOT as BASE

ROOT=BASE.parent/'e103_center'


class CenterPolicy(TopPickPolicy):
    def __init__(self,pretrained=True,fm_pretrained=True):
        super().__init__(pretrained=pretrained,fm_pretrained=fm_pretrained)
        # Adding the head must not change e102's action-noise/time RNG sequence.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(103)
            self.center_head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,3))
        self.register_buffer('center_mean',torch.zeros(3))
        self.register_buffer('center_std',torch.ones(3))

    def loss_parts(self,features,joints,target_id,actions,mask,center):
        hidden=self.visual_hidden(features,target_id)
        values=torch.cat((actions[...,:9],-1-2*actions[...,9:]/.0255),dim=-1)
        noisy,time,target=self.flow.training_path(values)
        velocity=self.flow.velocity(noisy,time,self.context_from_hidden(hidden,joints))
        error=(velocity.float()-target.float()).square()
        flow=(error*mask[...,None]).sum()/(mask.sum()*10)
        predicted=self.center_head(hidden).float()
        auxiliary=F.mse_loss(predicted,(center-self.center_mean)/self.center_std)
        error_mm=(predicted*self.center_std+self.center_mean-center).norm(dim=-1).mean()*1000
        return flow,auxiliary,error_mm

    @torch.no_grad()
    def predict_center(self,features,target_id):
        return self.center_head(self.visual_hidden(features,target_id)).float()*self.center_std+self.center_mean


def load_policy(saved):
    model=(CenterPolicy if saved['config'].get('aux_center') else TopPickPolicy)(pretrained=False)
    model.load_state_dict(saved['model'],strict=True)
    return model
