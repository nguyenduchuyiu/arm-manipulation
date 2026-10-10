"""Current-frame, task-conditioned policy using the existing FM action expert."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vit_b_16

from OATFlow.policy.model import ContextDecoder, FlowMatchingHead


class PriorPolicy(nn.Module):
    def __init__(self, vision_weights=None, flow_weights=None, horizon=25, action_head='fm'):
        super().__init__()
        if action_head not in ('fm', 'act'):
            raise ValueError('action head must be fm or act')
        self.action_head_type = action_head
        if action_head == 'fm':
            self.flow = FlowMatchingHead(horizon=horizon)
            self.initialization = self.flow.load_pretrained(flow_weights) if flow_weights else {}
            self.visual_projection = self.flow.visual
            del self.flow.visual
        else:
            from OATFlow.prior.act import ACTHead
            self.act = ACTHead(horizon)
            self.visual_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256),
                                                  nn.SiLU(), nn.Linear(256, 960))
            self.initialization = dict(action_head='ACT CVAE; random initialization',
                                       width=512, decoder_layers=1, posterior_layers=4, latent_dim=32, kl_weight=10.)
            if flow_weights:
                from safetensors import safe_open
                with safe_open(str(flow_weights), framework='pt', device='cpu') as source:
                    state = {name: source.get_tensor('state_proj.' + name) for name in ('weight', 'bias')}
                self.act.state_proj.load_state_dict(state, strict=True)
                self.initialization.update(state_projection_source=str(flow_weights),
                                           loaded_tensors=2, loaded_parameters=sum(value.numel() for value in state.values()))
        self.vision_encoder = vit_b_16(image_size=320)
        self.vision_encoder.heads = nn.Identity()
        if vision_weights:
            state = torch.load(vision_weights, map_location="cpu", weights_only=True, mmap=True)
            state = {k: v for k, v in state.items() if not k.startswith("heads.")}
            pos = state["encoder.pos_embedding"]
            grid = pos[:, 1:].transpose(1, 2).reshape(1, 768, 14, 14)
            grid = F.interpolate(grid, (20, 20), mode="bicubic", align_corners=True)
            state["encoder.pos_embedding"] = torch.cat((pos[:, :1], grid.flatten(2).transpose(1, 2)), 1)
            self.vision_encoder.load_state_dict(state, strict=True)
        self.vision_encoder.requires_grad_(False)
        self.vision_encoder.eval()
        self.view_embedding = nn.Parameter(torch.randn(2, 1, 960) * .02)
        self.task_embedding = nn.Embedding(12, 960)
        self.target_embedding = nn.Embedding(4, 960)
        self.context_decoder = ContextDecoder(tokens=131)
        self.head.context_tokens = 131
        self.initialization.update(context_shape=[131, 960], contextualize=True,
                                   tcow=False, vision_tokens_per_view=64,
                                   task_embedding_size=12, target_embedding_size=4,
                                   action_representation="absolute_joint", horizon=horizon)
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225])[None, :, None, None])

    @property
    def head(self):
        return self.flow if self.action_head_type == 'fm' else self.act

    def train(self, mode=True):
        super().train(mode)
        self.vision_encoder.eval()
        return self

    @torch.no_grad()
    def encode_views(self, overview, wrist):
        batch = len(overview)
        if overview.shape != (batch, 3, 320, 320) or wrist.shape != overview.shape:
            raise ValueError("both views must be current RGB [B,3,320,320]")
        images = (torch.cat((overview, wrist)).float() / 255 - self.rgb_mean) / self.rgb_std
        patches = self.vision_encoder._process_input(images)
        cls = self.vision_encoder.class_token.expand(2 * batch, -1, -1)
        features = self.vision_encoder.encoder(torch.cat((cls, patches), 1))[:, 1:]
        grid = features.transpose(1, 2).reshape(2 * batch, 768, 20, 20)
        return F.adaptive_avg_pool2d(grid, (8, 8)).flatten(2).transpose(1, 2).split(batch)

    def conditioning_features(self, overview, wrist, proprio, task_id, target_id):
        batch = len(proprio)
        if overview.shape != (batch, 64, 768) or wrist.shape != overview.shape:
            raise ValueError("cached views must be frozen ViT features [B,64,768]")
        if proprio.shape != (batch, 6) or task_id.shape != (batch,):
            raise ValueError("expected proprio [B,6] and task IDs [B]")
        if task_id.dtype != torch.long or (task_id < 0).any() or (task_id >= 12).any():
            raise ValueError("task IDs must be int64 in [0,12)")
        if target_id.shape != (batch,) or target_id.dtype != torch.long or (target_id < 0).any() or (target_id >= 4).any():
            raise ValueError("target IDs must be int64 in [0,4)")
        features = torch.cat((overview, wrist))
        overview_features, wrist_features = self.visual_projection(features).split(batch)
        state = (proprio - self.head.state_mean) / self.head.state_std
        state = F.pad(state, (0, 26)) + self.head.proprio(state)
        context = torch.cat((overview_features + self.view_embedding[0],
                             wrist_features + self.view_embedding[1],
                             self.task_embedding(task_id)[:, None],
                             self.target_embedding(target_id)[:, None],
                             self.head.state_proj(state)[:, None]), 1)
        return self.context_decoder(context, self.flow.context_projections if self.action_head_type == 'fm' else None)

    def conditioning(self, overview, wrist, proprio, task_id, target_id):
        features = self.encode_views(overview, wrist)
        return self.conditioning_features(*features, proprio, task_id, target_id)

    def forward_features(self, overview, wrist, proprio, task_id, target_id, noisy, time):
        context, kv = self.conditioning_features(overview, wrist, proprio, task_id, target_id)
        return self.flow.velocity(noisy, time, kv), context

    def forward(self, overview, wrist, proprio, task_id, target_id, noisy, time):
        context, kv = self.conditioning(overview, wrist, proprio, task_id, target_id)
        return self.flow.velocity(noisy, time, kv), context

    def act_forward_features(self, overview, wrist, proprio, task_id, target_id, action=None, valid=None):
        context, _ = self.conditioning_features(overview, wrist, proprio, task_id, target_id)
        return self.act(context, proprio, action, valid), context

    @torch.no_grad()
    def sample(self, overview, wrist, proprio, task_id, target_id, noise=None, steps=None):
        context, kv = self.conditioning(overview, wrist, proprio, task_id, target_id)
        if self.action_head_type == 'act':
            if noise is not None or steps is not None:
                raise ValueError('ACT inference does not use FM noise or Euler steps')
            prediction, _, _ = self.act(context, proprio)
            return self.act.denormalize(prediction)
        return self.flow.sample_actions(noise, kv, 10 if steps is None else steps)
