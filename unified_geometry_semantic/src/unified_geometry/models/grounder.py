"""SE(3)-equivariant scalar/vector encoder and task-conditioned coordinate decoder.

Scalar features use SAM3, distances and vector norms only. Vector operations mix
channels with shared XYZ weights and scalar gates. No learned spatial-axis bias.
"""

import math
import torch
from torch import nn

SCALE = 0.1
VECTORS = 8


def neighbors(values, index):
    batch = torch.arange(len(values), device=values.device)[:, None, None]
    return values[batch, index]


def mix_vectors(layer, vectors):
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        return layer(vectors.float().transpose(-1, -2)).transpose(-1, -2)


class EquivariantBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_scalar = nn.Linear(128, 128)
        self.neighbor_scalar = nn.Linear(128, 128)
        self.radial = nn.Sequential(nn.Linear(9, 64), nn.SiLU(), nn.Linear(64, 128))
        self.edge_weight = nn.Linear(128, 1)
        self.vector_coefficient = nn.Linear(128, VECTORS)
        self.self_vector = nn.Linear(VECTORS, VECTORS, bias=False)
        self.neighbor_vector = nn.Linear(VECTORS, VECTORS, bias=False)
        self.scalar_update = nn.Sequential(
            nn.Linear(256 + VECTORS, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.scalar_norm = nn.LayerNorm(128)
        self.vector_gate = nn.Linear(128, VECTORS)
        self.register_buffer(
            "radial_scales", torch.tensor([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
        )
        nn.init.normal_(self.vector_coefficient.weight, std=0.01)
        nn.init.zeros_(self.vector_coefficient.bias)

    def forward(self, h, v, relative, index):
        distance2 = relative.square().sum(-1)
        radial = torch.cat(
            [
                torch.log1p(distance2)[..., None],
                torch.exp(-distance2[..., None] / self.radial_scales.square()),
            ],
            -1,
        )
        message = torch.nn.functional.silu(
            self.self_scalar(h)[:, :, None]
            + neighbors(self.neighbor_scalar(h), index)
            + self.radial(radial)
        )
        weight = self.edge_weight(message).softmax(dim=2)
        scalar_message = (weight * message).sum(2)
        vector_norm = v.square().sum(-1).add(1e-08).sqrt()
        new_h = self.scalar_norm(
            h + self.scalar_update(torch.cat([h, scalar_message, vector_norm], -1))
        )
        transported = neighbors(mix_vectors(self.neighbor_vector, v), index)
        generated = (
            0.1 * self.vector_coefficient(message).tanh()[..., None] * relative[:, :, :, None, :]
        )
        new_v = mix_vectors(self.self_vector, v) + (
            weight[..., None] * (transported + generated)
        ).sum(2)
        new_v = new_v * self.vector_gate(new_h).sigmoid()[..., None]
        return (new_h, new_v)


class EquivariantGrounder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_norm = nn.LayerNorm(1024)
        self.point_input = nn.Sequential(
            nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 128), nn.LayerNorm(128)
        )
        self.vector_input = nn.Linear(128, VECTORS)
        nn.init.normal_(self.vector_input.weight, std=0.01)
        nn.init.zeros_(self.vector_input.bias)
        self.encoder = nn.ModuleList([EquivariantBlock(), EquivariantBlock()])
        self.task_embedding = nn.Embedding(4, 128)
        self.point_queries = nn.Parameter(torch.randn(128, 128) * 0.02)
        layer = nn.TransformerDecoderLayer(
            128,
            4,
            384,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, 2, norm=nn.LayerNorm(128))
        self.coordinate_query = nn.Linear(128, 128, bias=False)
        self.coordinate_key = nn.Linear(128, 128, bias=False)
        self.vector_readout = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, VECTORS))
        nn.init.normal_(self.vector_readout[-1].weight, std=0.01)
        nn.init.zeros_(self.vector_readout[-1].bias)

    def encode(self, points, features, index):
        h = self.point_input(self.feature_norm(features.float()))
        centered = (points - points.mean(1, keepdim=True)) / SCALE
        v = self.vector_input(h).tanh()[..., None] * centered[:, :, None, :]
        relative = (neighbors(points, index) - points[:, :, None]) / SCALE
        for block in self.encoder:
            h, v = block(h, v, relative, index)
        return (h, v)

    def decode(self, points, h, v, task_id):
        query = self.point_queries[None] + self.task_embedding(task_id)[:, None]
        latent = self.decoder(query, h)
        logits = torch.bmm(
            self.coordinate_query(latent), self.coordinate_key(h).transpose(1, 2)
        ) / math.sqrt(128)
        coefficient = self.vector_readout(latent).tanh().float()
        with torch.autocast(device_type=points.device.type, enabled=False):
            weight = logits.float().softmax(-1)
            points = points.float()
            center = points.mean(1, keepdim=True)
            base = center + torch.bmm(weight, points - center)
            pooled = torch.bmm(weight, v.float().flatten(-2)).reshape(len(points), 128, VECTORS, 3)
            offset = (coefficient[..., None] * pooled).sum(-2) * SCALE
            return base + offset

    def forward(self, points, features, task_id, index):
        h, v = self.encode(points, features, index)
        return self.decode(points, h, v, task_id)

    def all_tasks(self, points, features, index):
        h, v = self.encode(points, features, index)
        b, n, _ = h.shape

        def expand(value):
            return value[:, None].expand(b, 4, *value.shape[1:]).reshape(b * 4, *value.shape[1:])

        task = torch.arange(4, device=points.device).repeat(b)
        return self.decode(expand(points), expand(h), expand(v), task).reshape(b, 4, 128, 3)
