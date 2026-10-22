"""Frozen ViT-B/16 plus target query, with optional reach FM initialization."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vit_b_16

from OATFlow.reach.model import ReachPolicy
from OATFlow.top_pick.data import SOURCE,VISION


class TopPickPolicy(ReachPolicy):
    def __init__(self, pretrained=True, fm_pretrained=True):
        super().__init__()
        if pretrained:
            saved=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
            if fm_pretrained:
                self.load_state_dict(saved['model'],strict=True)
            else:
                # Match action/state units without transferring any learned FM parameter.
                for name,buffer in self.named_buffers():buffer.copy_(saved['model'][name])
        # Replace the old oracle goal-XYZ token with a target-conditioned visual token.
        del self.xyz_encoder
        self.visual=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,256))
        self.target=nn.Embedding(4,256)
        self.position=nn.Parameter(torch.randn(1,64,256)*.02)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(256,4,1024,dropout=0.,
                                             batch_first=True,norm_first=True),num_layers=2)
        self.goal=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,960))
        self.vision=vit_b_16(image_size=320);self.vision.heads=nn.Identity()
        if pretrained:
            saved=torch.load(VISION,map_location='cpu',weights_only=True,mmap=True)
            saved={k:v for k,v in saved.items() if not k.startswith('heads.')}
            pos=saved['encoder.pos_embedding'];grid=pos[:,1:].transpose(1,2).reshape(1,768,14,14)
            grid=F.interpolate(grid,(20,20),mode='bicubic',align_corners=True)
            saved['encoder.pos_embedding']=torch.cat((pos[:,:1],grid.flatten(2).transpose(1,2)),dim=1)
            self.vision.load_state_dict(saved,strict=True)
        self.vision.requires_grad_(False);self.vision.eval()
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def train(self,mode=True):
        super().train(mode);self.vision.eval();return self

    @torch.no_grad()
    def encode(self,rgb):
        x=(rgb.float()/255-self.rgb_mean)/self.rgb_std
        patches=self.vision._process_input(x)
        x=self.vision.encoder(torch.cat((self.vision.class_token.expand(len(x),-1,-1),patches),dim=1))[:,1:]
        return F.adaptive_avg_pool2d(x.transpose(1,2).reshape(len(x),768,20,20),(8,8)).flatten(2).transpose(1,2)

    def visual_hidden(self,features,target_id):
        memory=self.visual(features)+self.position
        query=self.target(target_id)[:,None]
        return self.decoder(query,memory)[:,0]

    def context_from_hidden(self,hidden,joints):
        goal=self.goal(hidden)
        state=torch.cat(((joints[:,:5]-self.flow.state_mean[:5])/self.flow.state_std[:5],
                         -1-2*joints[:,5:]/.0255),dim=-1)
        state=F.pad(state,(0,26))+self.flow.proprio(state)
        tokens=torch.stack((goal,self.flow.state_proj(state)),dim=1)
        return [projection(tokens) for projection in self.flow.context_projections]

    def conditioning(self,features,joints,target_id):
        return self.context_from_hidden(self.visual_hidden(features,target_id),joints)

    def loss(self,features,joints,target_id,actions,mask):
        values=torch.cat((actions[...,:9],-1-2*actions[...,9:]/.0255),dim=-1)
        noisy,time,target=self.flow.training_path(values)
        velocity=self.flow.velocity(noisy,time,self.conditioning(features,joints,target_id))
        error=(velocity.float()-target.float()).square()
        return (error*mask[...,None]).sum()/(mask.sum()*10)

    @torch.no_grad()
    def sample(self,features,joints,target_id,noise,steps=10):
        output=self.flow.sample_actions(noise,self.conditioning(features,joints,target_id),steps)
        return torch.cat((output[...,:9],-.0255*(output[...,9:]+1)/2),dim=-1)
