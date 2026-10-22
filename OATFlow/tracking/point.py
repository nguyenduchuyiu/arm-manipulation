"""TCOW's one-view RGB+query-channel backbone with an object-center XYZ head."""
import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.tracking.backbone import CausalBackbone


class PointXYZ(nn.Module):
    def __init__(self, image_size=(240, 320), patch_size=16, max_frames=30,
                 width=768, depth=12, heads=12, drop_path_rate=.1, local_query=False):
        super().__init__()
        self.config = dict(image_size=list(image_size), patch_size=patch_size,
                           max_frames=max_frames, width=width, depth=depth, heads=heads,
                           drop_path_rate=drop_path_rate, local_query=local_query)
        self.image_size = tuple(image_size)
        self.max_frames = max_frames
        self.patch_size = patch_size
        self.local_query = local_query
        self.backbone = CausalBackbone(image_size, patch_size, max_frames, width, depth, heads,
                                      in_channels=4, drop_path_rate=drop_path_rate)
        # Original TCOW checkpoint has norm_embeddings=False.
        self.score = nn.Linear(width, 1)
        self.xyz = nn.Sequential(nn.LayerNorm(width + 2), nn.Linear(width + 2, 256),
                                 nn.GELU(), nn.Linear(256, 3))
        self.descriptor = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128, bias=False)) if local_query else None
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, image_size[0] // patch_size),
                                torch.linspace(-1, 1, image_size[1] // patch_size), indexing='ij')
        self.register_buffer('patch_xy', torch.stack((xx, yy), dim=-1).reshape(-1, 2))
        self.register_buffer('xyz_mean', torch.zeros(3))
        self.register_buffer('xyz_std', torch.ones(3))

    def features(self, rgb, query):
        if (rgb.ndim != 5 or rgb.shape[1] != 3 or tuple(rgb.shape[-2:]) != self.image_size
                or not 1 <= rgb.shape[2] <= self.max_frames):
            raise ValueError('expected RGB [B,3,T,H,W]')
        b, _, t, h, w = rgb.shape
        if (query.shape != (b, 1, t, h, w) or not query.is_floating_point()
                or not torch.isfinite(query).all() or (query < 0).any() or (query > 1).any()
                or query[:, :, 1:].count_nonzero() or not (query[:, :, 0].sum((-1, -2)) > 0).all()):
            raise ValueError('query must be [B,1,T,H,W], one frame0 prompt and zero later frames')
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255
        elif rgb.is_floating_point():
            rgb = rgb.float()
        else:
            raise ValueError('RGB must be uint8 or floating point')
        pixels = torch.cat(((rgb - .45) / .225, query), dim=1)
        x, cls = self.backbone.embed(pixels.permute(0, 2, 1, 3, 4)[:, None])
        for block in self.backbone.blocks:
            x, cls = block(x, cls)
        return x[:, 0]

    def hidden(self, rgb, query):
        features = self.features(rgb, query)
        scores = self.score(features).squeeze(-1).float()
        if self.local_query:
            reference = self.initial_reference(features, query)
            keys = F.normalize(self.descriptor(features).float(), dim=-1)
            target = F.normalize(self.descriptor(reference).float(), dim=-1)
            scores = scores + 5 * (keys * target[:, None, None]).sum(-1)
        attention = scores.softmax(-1).to(features.dtype)
        pooled = (features * attention[..., None]).sum(-2)
        xy = attention @ self.patch_xy.to(attention.dtype)
        return self.xyz[:3](torch.cat((pooled, xy), dim=-1))

    def forward(self, rgb, query):
        return self.xyz[3](self.hidden(rgb, query))

    def initial_reference(self, features, query):
        # Only frame0 determines the fixed appearance reference, even for long clips.
        patch_mass = F.avg_pool2d(query[:, :, 0], self.patch_size, self.patch_size)
        weights = torch.zeros_like(patch_mass).flatten(1)
        weights.scatter_(1, patch_mass.flatten(1).argmax(-1, keepdim=True), 1)
        weights = F.max_pool2d(weights.reshape_as(patch_mass), 3, stride=1, padding=1).flatten(1).float()
        weights = weights / weights.sum(-1, keepdim=True)
        return (features[:, 0] * weights[..., None].to(features.dtype)).sum(-2)

    def set_xyz_statistics(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.xyz_mean.dtype, device=self.xyz_mean.device)
        std = torch.as_tensor(std, dtype=self.xyz_std.dtype, device=self.xyz_std.device)
        if (mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('XYZ statistics require finite mean/std[3], std>0')
        self.xyz_mean.copy_(mean); self.xyz_std.copy_(std)

    def normalize_xyz(self, xyz):
        return (xyz - self.xyz_mean) / self.xyz_std

    @torch.no_grad()
    def predict(self, rgb, query):
        return self(rgb, query)[:, -1].float() * self.xyz_std + self.xyz_mean


def load_tcow(model, rgb_path, query_path):
    source = torch.load(rgb_path, map_location='cpu', weights_only=True, mmap=True)
    extra = torch.load(query_path, map_location='cpu', weights_only=True)
    if any(model.config[k] != v for k, v in source['config'].items()):
        raise ValueError('TCOW backbone configuration does not match')
    state = source['backbone']
    state['patch_embed.proj.weight'] = torch.cat((state['patch_embed.proj.weight'], extra['query_weight']), dim=1)
    expected = model.backbone.state_dict()
    if (state.keys() != expected.keys() or any(state[k].shape != expected[k].shape for k in state)
            or any(not value.isfinite().all() for value in state.values())):
        raise ValueError('invalid TCOW query backbone tensors')
    model.backbone.load_state_dict(state, strict=True)
