"""Small frozen-representation readouts for slot colors and current world XYZ."""
import torch
from torch import nn


class SceneProbe(nn.Module):
    """Shared color classifier on four learned spatial pools; goal query for XYZ."""
    def __init__(self, input_dim=960, width=128):
        super().__init__()
        self.input_dim = input_dim
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.SiLU())
        self.slot_weights = nn.Parameter(torch.zeros(4, 131))
        self.goal_query = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.goal_attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.color = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4))
        self.position = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))
        nn.init.zeros_(self.position[-1].weight)
        nn.init.zeros_(self.position[-1].bias)
        self.register_buffer('position_mean', torch.zeros(5, 3))
        self.register_buffer('position_std', torch.ones(5, 3))

    def set_statistics(self, mean, std):
        self.position_mean.copy_(mean)
        self.position_std.copy_(std.clamp_min(.005))

    @torch.no_grad()
    def initialize_slots(self, train_pixel_centers):
        rows, cols = torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij')
        pixels = torch.stack(((cols.flatten() + .5) * 40, (rows.flatten() + .5) * 40), -1).float().to(self.slot_weights.device)
        train_pixel_centers = train_pixel_centers.to(self.slot_weights.device)
        self.slot_weights.fill_(-10.)
        self.slot_weights[:, :64].copy_(-.5 * ((pixels[None] - train_pixel_centers[:, None]) / 32.).square().sum(-1))

    def forward(self, tokens):
        if tokens.shape[1:] != (131, self.input_dim):
            raise ValueError('expected131context tokens')
        value = self.project(tokens)
        # Slot-specific pooling is independent of goal/content. A shared color
        # classifier encourages reading local colors instead of a layout code.
        slots = torch.einsum('sl,bld->bsd', self.slot_weights.softmax(-1).to(value.dtype), value)
        query = self.goal_query.expand(len(tokens), -1, -1) + value[:, 129:130]
        goal = query + self.goal_attention(query, value, value, need_weights=False)[0]
        position = self.position(torch.cat((slots, goal), 1)).float()
        return self.color(slots).float(), position * self.position_std + self.position_mean


class ProprioProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 31))
        self.register_buffer('position_mean', torch.zeros(5, 3))
        self.register_buffer('position_std', torch.ones(5, 3))

    def set_statistics(self, mean, std):
        self.position_mean.copy_(mean)
        self.position_std.copy_(std.clamp_min(.005))

    def forward(self, proprio, target):
        value = self.net(torch.cat((proprio, nn.functional.one_hot(target, 4).to(proprio.dtype)), -1)).float()
        return value[:, :16].reshape(-1, 4, 4), value[:, 16:].reshape(-1, 5, 3) * self.position_std + self.position_mean


def fm_tokens(policy, context):
    """Exact per-layer K/V arrays consumed by the frozen FM cross-attention."""
    with torch.no_grad():
        kv = [projection(context) for projection in policy.flow.context_projections]
        return torch.cat([value.flatten(2) for pair in kv for value in pair], -1)


def probe_loss(logits, positions, colors, truth, position_std):
    color = nn.functional.cross_entropy(logits.flatten(0, 1), colors.flatten())
    error = ((positions - truth) / position_std).square()
    objects = error[:, :4].mean()
    goal = error[:, 4].mean()
    return color + objects + goal, dict(color=float(color.detach()), object_position=float(objects.detach()), target_position=float(goal.detach()))
