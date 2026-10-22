"""Frozen RGB ViT-B/16 and a mask-conditioned two-view XYZ decoder."""
import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.localization.model import FrozenViT


class TwoViewXYZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrozenViT(tokens=400)
        width = 384
        self.projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.mask_embedding = nn.Linear(1, width)
        self.position = nn.Parameter(torch.randn(1, 2, 400, width) * .02)
        self.query = nn.Parameter(torch.randn(1, 1, width) * .02)
        layer = nn.TransformerDecoderLayer(width, 6, 1536, dropout=0., batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 4, norm=nn.LayerNorm(width))
        self.xyz = nn.Linear(width, 3)

    @torch.no_grad()
    def encode(self, rgb):
        assert rgb.shape[1:] == (2, 320, 320, 3)
        # Keep the same ViT/decoder boundary precision with and without the RAM cache.
        return self.encoder(rgb.flatten(0, 1)).unflatten(0, (len(rgb), 2)).half()

    def from_features(self, features, masks):
        assert features.shape[1:] == (2, 400, 768)
        assert masks.shape[1:] == (2, 320, 320)
        occupancy = F.avg_pool2d(masks.flatten(0, 1)[:, None].float(), 16).flatten(2).transpose(1, 2)
        occupancy = occupancy.unflatten(0, (len(features), 2))
        memory = self.projection(features) + self.mask_embedding(occupancy) + self.position
        context = self.decoder(self.query.expand(len(features), -1, -1), memory.flatten(1, 2))
        return self.xyz(context[:, 0]).float()

    def forward(self, rgb, masks):
        return self.from_features(self.encode(rgb), masks)
