"""Frozen ViT-B/16 top-camera object-center localization."""
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
from torchvision.models.vision_transformer import interpolate_embeddings

WEIGHTS = Path('/workspace/robotics_assets/pretrained/vision/vit_b_16-c867db91.pth')


def state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenViT(nn.Module):
    def __init__(self, tokens=64):
        super().__init__()
        if tokens not in (64, 400):
            raise ValueError('tokens must be 64 or 400')
        self.tokens = tokens
        self.vit = vit_b_16(image_size=320)
        state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
        state = interpolate_embeddings(320,16,state,interpolation_mode='bicubic')
        self.vit.load_state_dict(state,strict=True)
        self.vit.heads = nn.Identity()
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        self.requires_grad_(False); self.eval()

    def train(self,mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self,rgb):
        return self.features(rgb)

    def features(self,rgb):
        # uint8 RGB NHWC, 320px preserved; CLS participates but is not exported.
        x = rgb.permute(0,3,1,2).float()/255
        x = self.vit._process_input((x-self.mean)/self.std)
        x = torch.cat((self.vit.class_token.expand(len(x),-1,-1),x),dim=1)
        x = self.vit.encoder(x)[:,1:]
        if self.tokens == 400:
            return x
        x = x.transpose(1,2).reshape(-1,768,20,20)
        return F.adaptive_avg_pool2d(x,(8,8)).flatten(2).transpose(1,2)


class IDAttention(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width,4,batch_first=True,dropout=0.)
        self.ff = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))

    def forward(self,q,memory):
        key = self.memory_norm(memory)
        q = q + self.attention(self.query_norm(q),key,key,need_weights=False)[0]
        return q + self.ff(q)


class ObjectXYZ(nn.Module):
    def __init__(self,width=192,tokens=64):
        super().__init__()
        if tokens not in (64, 400):
            raise ValueError('tokens must be 64 or 400')
        self.tokens = tokens
        self.projection = nn.Sequential(nn.LayerNorm(768),nn.Linear(768,width))
        self.view = nn.Parameter(torch.randn(1,3,1,width)*.02)
        self.position = nn.Parameter(torch.randn(1,1,tokens,width)*.02)
        self.identity = nn.Embedding(4,width)
        self.blocks = nn.ModuleList([IDAttention(width) for _ in range(2)])
        self.xyz = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),nn.Linear(width,3))

    def forward(self,tokens,ids=None):
        assert tokens.shape[1] == 1 and tokens.shape[2:] == (self.tokens,768)
        b = len(tokens)
        # Keep the saved view-tensor shape; top-view inference uses its first entry.
        memory = (self.projection(tokens)+self.view[:,:tokens.shape[1]]+self.position).flatten(1,2)
        if ids is None: ids = torch.arange(4,device=tokens.device)[None].expand(b,-1)
        if ids.ndim==1: ids = ids[:,None]
        q = self.identity(ids)
        for block in self.blocks: q = block(q,memory)
        return self.xyz(q)
