"""ACT conditional VAE action head over the existing observation context."""
import torch
from torch import nn
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTEncoder, ACTDecoder, create_sinusoidal_pos_embedding

from OATFlow.policy.model import FlowMatchingHead


class ACTHead(nn.Module):
    action_dim = 6
    latent_dim = 32
    kl_weight = 10.0
    set_statistics = FlowMatchingHead.set_statistics

    def __init__(self, horizon=50):
        super().__init__()
        self.horizon = horizon
        config = ACTConfig(device='cpu', chunk_size=horizon, n_action_steps=horizon)
        self.proprio = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, 256),
                                     nn.SiLU(), nn.Linear(256, 32))
        nn.init.zeros_(self.proprio[-1].weight)
        nn.init.zeros_(self.proprio[-1].bias)
        self.state_proj = nn.Linear(32, 960)
        self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
        self.cls = nn.Embedding(1, 512)
        self.posterior_state = nn.Linear(6, 512)
        self.posterior_action = nn.Linear(6, 512)
        self.posterior_params = nn.Linear(512, 2 * self.latent_dim)
        self.context_projection = nn.Linear(960, 512)
        self.latent_projection = nn.Linear(self.latent_dim, 512)
        self.decoder = ACTDecoder(config)
        self.queries = nn.Embedding(horizon, 512)
        self.output = nn.Linear(512, 6)
        self.register_buffer('posterior_positions', create_sinusoidal_pos_embedding(horizon + 2, 512)[:, None])
        self.register_buffer('memory_positions', create_sinusoidal_pos_embedding(132, 512)[:, None])
        for name in ('state_mean', 'action_mean'):
            self.register_buffer(name, torch.zeros(6))
        for name in ('state_std', 'action_std'):
            self.register_buffer(name, torch.ones(6))
        for module in (self.vae_encoder, self.decoder):
            for parameter in module.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

    def posterior(self, proprio, action, valid):
        batch = len(proprio)
        if action.shape != (batch, self.horizon, 6) or valid.shape != (batch, self.horizon) or valid.dtype != torch.bool:
            raise ValueError('ACT posterior requires Hx6 labels and boolean validity mask')
        if not valid.any(dim=1).all():
            raise ValueError('ACT training chunks need at least one valid action')
        normalized = (action - self.action_mean) / self.action_std
        normalized = normalized.masked_fill(~valid[:, :, None], 0)
        state = (proprio - self.state_mean) / self.state_std
        tokens = torch.cat((self.cls.weight[None].expand(batch, -1, -1),
                            self.posterior_state(state)[:, None], self.posterior_action(normalized)), dim=1)
        padding = torch.cat((torch.zeros(batch, 2, device=valid.device, dtype=torch.bool), ~valid), dim=1)
        encoded = self.vae_encoder(tokens.transpose(0, 1), pos_embed=self.posterior_positions,
                                   key_padding_mask=padding)[0]
        return self.posterior_params(encoded).float().chunk(2, dim=-1)

    def forward(self, context, proprio, action=None, valid=None):
        batch = len(proprio)
        if context.shape != (batch, 131, 960):
            raise ValueError('ACT requires the existing131 observation context tokens')
        if self.training:
            if action is None or valid is None:
                raise ValueError('ACT training requires action labels and validity mask')
            mu, logvar = self.posterior(proprio, action, valid)
            latent = mu + (.5 * logvar).exp() * torch.randn_like(mu)
        else:
            if action is not None or valid is not None:
                raise ValueError('ACT inference must not use future action labels')
            mu = logvar = None
            latent = torch.zeros(batch, self.latent_dim, device=context.device)
        memory = torch.cat((self.latent_projection(latent)[:, None], self.context_projection(context)), dim=1)
        queries = torch.zeros(self.horizon, batch, 512, device=context.device, dtype=memory.dtype)
        decoded = self.decoder(queries, memory.transpose(0, 1),
                               decoder_pos_embed=self.queries.weight[:, None], encoder_pos_embed=self.memory_positions)
        return self.output(decoded.transpose(0, 1)), mu, logvar

    def loss(self, prediction, action, valid, mu, logvar):
        target = (action - self.action_mean) / self.action_std
        l1 = ((prediction.float() - target).abs() * valid[:, :, None]).sum() / (valid.sum() * 6)
        kl = (-.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
        return l1 + self.kl_weight * kl, dict(l1=l1, kl=kl)

    def denormalize(self, prediction):
        return prediction.float() * self.action_std + self.action_mean
