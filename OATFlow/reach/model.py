"""Joint-conditioned10D EE-delta reach policy."""
import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.reach.flow import FlowMatchingHead


class ReachPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow=FlowMatchingHead(horizon=25)
        self.xyz_encoder=nn.Sequential(nn.Linear(3,128),nn.SiLU(),nn.Linear(128,960))
        self.register_buffer('xyz_mean',torch.zeros(3))
        self.register_buffer('xyz_std',torch.ones(3))

    def conditioning(self,goal_base,joints):
        assert goal_base.shape[1:]==(3,) and joints.shape[1:]==(6,)
        state=torch.cat(((joints[:,:5]-self.flow.state_mean[:5])/self.flow.state_std[:5],
            -1-2*joints[:,5:]/.0255),dim=-1)
        state=F.pad(state,(0,26))+self.flow.proprio(state)
        goal=self.xyz_encoder((goal_base-self.xyz_mean)/self.xyz_std)
        context=torch.stack((goal,self.flow.state_proj(state)),dim=1)
        return [projection(context) for projection in self.flow.context_projections]

    @torch.no_grad()
    def sample(self,goal_base,joints,noise,steps=10):
        prediction=self.flow.sample_actions(noise,self.conditioning(goal_base,joints),steps)
        return torch.cat((prediction[...,:9],-.0255*(prediction[...,9:]+1)/2),dim=-1)

    def loss(self, goal_base, joints, actions, mask):
        # The flow denormalizes XYZ/rotation; its jaw channel stays fixed-range normalized.
        values = torch.cat((actions[..., :9], -1-2*actions[..., 9:]/.0255), dim=-1)
        noisy, time, target = self.flow.training_path(values)
        velocity = self.flow.velocity(noisy, time, self.conditioning(goal_base, joints))
        error = (velocity.float()-target.float()).square()
        return (error * mask[..., None]).sum() / (mask.sum() * error.shape[-1])
