"""Current wrist image encoded with a copy of TCOW's pretrained spatial layers."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def wrist_tensor(rgb, device):
    if rgb.shape != (320, 320, 3) or rgb.dtype.name != "uint8":
        raise ValueError("expected full, uncropped wrist RGB uint8 [320,320,3]")
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device)
    return (image.float() / 255 - .45) / .225


class SpatialBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm1 = deepcopy(source.norm1)
        self.attn = deepcopy(source.attn)
        self.norm2 = deepcopy(source.norm2)
        self.mlp = deepcopy(source.mlp)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class WristEncoder(nn.Module):
    tokens = 400

    def __init__(self, backbone):
        super().__init__()
        source = backbone.timesformer.model
        patch = source.patch_embed.proj
        if patch.out_channels != 768 or patch.kernel_size != (16, 16):
            raise ValueError("wrist encoder requires TCOW ViT-B/16 spatial weights")
        self.patch = nn.Conv2d(3, 768, 16, 16, bias=patch.bias is not None)
        with torch.no_grad():
            self.patch.weight.copy_(patch.weight[:, :3])
            if patch.bias is not None:
                self.patch.bias.copy_(patch.bias)
        self.cls_token = nn.Parameter(source.cls_token.detach().clone())
        height, width = source.patch_embed.img_size
        grid = (height // 16, width // 16)
        if source.pos_embed.shape != (1, grid[0] * grid[1] + 1, 768):
            raise ValueError("TCOW positional embeddings disagree with its image grid")
        positions = source.pos_embed[:, 1:].detach().transpose(1, 2).reshape(1, 768, *grid)
        positions = F.interpolate(positions, size=(20, 20), mode="nearest").flatten(2).transpose(1, 2)
        self.pos_embed = nn.Parameter(torch.cat((source.pos_embed[:, :1].detach(), positions), dim=1).clone())
        self.blocks = nn.ModuleList([SpatialBlock(block) for block in source.blocks])
        self.norm = deepcopy(source.norm)

    def forward(self, image):
        if image.shape[1:] != (3, 320, 320):
            raise ValueError("wrist encoder requires current RGB [B,3,320,320]")
        patches = self.patch(image).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(len(image), -1, -1), patches), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and torch.is_grad_enabled() else block(x)
        # Keep all 20x20 locations. CLS participates internally but is not sent to FM.
        return self.norm(x)[:, 1:]
