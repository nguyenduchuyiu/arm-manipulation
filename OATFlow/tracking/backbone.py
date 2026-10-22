"""Causal space/time backbone for point-query tracking.

The divided-attention and first-frame CLS design follows TimeSformer/TCOW.
Implemented locally using PyTorch; no upstream modules are imported.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x, causal=False):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(x.transpose(1, 2).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SpaceTimeBlock(nn.Module):
    def __init__(self, width, heads, drop_path=0.):
        super().__init__()
        self.temporal_norm1 = nn.LayerNorm(width, eps=1e-6)
        self.temporal_attn = Attention(width, heads)
        self.temporal_fc = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = MLP(width)
        self.drop_path = drop_path

    def drop(self, x):
        if not self.training or self.drop_path == 0:
            return x
        keep = 1 - self.drop_path
        mask = (torch.rand((len(x),) + (1,) * (x.ndim - 1), device=x.device) < keep).to(x.dtype)
        return x * mask / keep

    def forward(self, x, cls):
        b, v, t, n, d = x.shape
        temporal = x.permute(0, 1, 3, 2, 4).reshape(b * v * n, t, d)
        temporal = self.temporal_fc(self.drop(self.temporal_attn(self.temporal_norm1(temporal), causal=True)))
        x = x + temporal.reshape(b, v, n, t, d).permute(0, 1, 3, 2, 4)
        spatial = torch.cat((cls[:, :, None].expand(b, v, t, 1, d), x), dim=3)
        update = self.drop(self.attn(self.norm1(spatial.reshape(b * v * t, n + 1, d))))
        update = update.reshape(b, v, t, n + 1, d)
        x = x + update[:, :, :, 1:]
        # A CLS averaged over time leaks future frames through the next block.
        cls = cls + update[:, :, 0, :1]
        if self.drop_path:
            tokens = torch.cat((cls, x.reshape(b, v, t * n, d)), dim=2)
            update = self.drop(self.mlp(self.norm2(tokens)).reshape(b * v, 1 + t * n, d))
            update = update.reshape(b, v, 1 + t * n, d)
            return x + update[:, :, 1:].reshape_as(x), cls + update[:, :, :1]
        return x + self.mlp(self.norm2(x)), cls + self.mlp(self.norm2(cls))


class PatchEmbedding(nn.Module):
    def __init__(self, width, patch_size, in_channels=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, width, patch_size, stride=patch_size)

    def forward(self, rgb):
        b, v, t, c, h, w = rgb.shape
        x = self.proj(rgb.reshape(b * v * t, c, h, w)).flatten(2).transpose(1, 2)
        return x.reshape(b, v, t, -1, x.shape[-1])


class CausalBackbone(nn.Module):
    def __init__(self, image_size, patch_size, max_frames, width, depth, heads,
                 in_channels=3, drop_path_rate=0.):
        super().__init__()
        n = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.patch_embed = PatchEmbedding(width, patch_size, in_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.empty(1, n + 1, width))
        self.time_embed = nn.Parameter(torch.zeros(1, max_frames, width))
        self.blocks = nn.ModuleList([SpaceTimeBlock(width, heads, drop_path_rate * i / max(1, depth - 1))
                                     for i in range(depth)])
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def embed(self, rgb):
        x = self.patch_embed(rgb)
        b, v, t, _, d = x.shape
        x = x + self.pos_embed[:, None, None, 1:].to(x.dtype)
        # Slicing keeps prefix embeddings invariant when more frames are appended.
        x = x + self.time_embed[:, None, :t, None].to(x.dtype)
        cls = (self.cls_token + self.pos_embed[:, :1]).to(x.dtype)
        return x, cls[:, None].expand(b, v, 1, d)
