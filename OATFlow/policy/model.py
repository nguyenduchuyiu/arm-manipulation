# Portions adapted from Hugging Face LeRobot SmolVLA.
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OAT-Flow policy: tracking features, wrist vision, context decoder and flow matching.

Action input/output projections use six coordinates; older padded heads require retraining.
Upstream TCOW itself lives in third_party/tcow; tracking.py loads that dependency.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

SOURCE_REPO = "lerobot/smolvla_base"
SOURCE_REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
        return self.weight * value.to(x.dtype)


def apply_rotary_embedding(x, positions):
    """LeRobot apply_rope, with [B,L,H,64] input."""
    fraction = torch.arange(32, device=x.device, dtype=torch.float32) / 32
    phase = positions[..., None].float() / (10000 ** fraction)
    cos, sin = phase.cos()[..., None, :], phase.sin()[..., None, :]
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)


class ActionAttention(nn.Module):
    def __init__(self, cross):
        super().__init__()
        self.q_proj = nn.Linear(720, 960, bias=False)
        self.k_proj = nn.Linear(320 if cross else 720, 320, bias=False)
        self.v_proj = nn.Linear(320 if cross else 720, 320, bias=False)
        self.o_proj = nn.Linear(960, 720, bias=False)


class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(720, 2048, bias=False)
        self.up_proj = nn.Linear(720, 2048, bias=False)
        self.down_proj = nn.Linear(2048, 720, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ActionTransformerBlock(nn.Module):
    def __init__(self, cross):
        super().__init__()
        self.self_attn = ActionAttention(cross)
        self.mlp = ActionMLP()
        self.input_layernorm = RMSNorm(720)
        self.post_attention_layernorm = RMSNorm(720)

    def forward(self, x, positions, context_kv=None):
        residual = x
        value = self.input_layernorm(x)
        batch, length, _ = value.shape
        attn = self.self_attn
        q = apply_rotary_embedding(attn.q_proj(value).reshape(batch, length, 15, 64), positions)
        if context_kv is None:
            k = apply_rotary_embedding(attn.k_proj(value).reshape(batch, length, 5, 64), positions)
            v = attn.v_proj(value).reshape(batch, length, 5, 64)
        else:
            prefix_k, prefix_v = context_kv
            k = attn.k_proj(prefix_k.flatten(2)).reshape(batch, -1, 5, 64)
            v = attn.v_proj(prefix_v.flatten(2)).reshape(batch, -1, 5, 64)
        # Every action reads all action tokens in SA and all visual/state tokens
        # in CA, retaining the existing policy's dense interaction.
        k = k.repeat_interleave(3, dim=2)
        v = v.repeat_interleave(3, dim=2)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, length, 960)
        x = residual + attn.o_proj(attended)
        return x + self.mlp(self.post_attention_layernorm(x))


class ContextKVProjection(nn.Module):
    def __init__(self):
        super().__init__()
        # These projections are loaded from the corresponding VLM CA layer.
        self.norm = RMSNorm(960)
        self.k_proj = nn.Linear(960, 320, bias=False)
        self.v_proj = nn.Linear(960, 320, bias=False)

    def forward(self, context):
        batch, length, _ = context.shape
        x = self.norm(context)
        positions = torch.arange(length, device=x.device)[None].expand(batch, -1)
        k = apply_rotary_embedding(self.k_proj(x).reshape(batch, length, 5, 64), positions)
        v = self.v_proj(x).reshape(batch, length, 5, 64)
        return k, v


class ContextTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(960)
        self.post_attention_layernorm = RMSNorm(960)
        self.self_attn = nn.Module()
        for name, out in (("q_proj", 960), ("k_proj", 320), ("v_proj", 320), ("o_proj", 960)):
            setattr(self.self_attn, name, nn.Linear(960, out, bias=False))
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(960, 2560, bias=False)
        self.mlp.up_proj = nn.Linear(960, 2560, bias=False)
        self.mlp.down_proj = nn.Linear(2560, 960, bias=False)

    def forward(self, x, positions):
        value = self.input_layernorm(x)
        batch, length, _ = value.shape
        q = apply_rotary_embedding(self.self_attn.q_proj(value).reshape(batch, length, 15, 64), positions)
        k = apply_rotary_embedding(self.self_attn.k_proj(value).reshape(batch, length, 5, 64), positions)
        v = self.self_attn.v_proj(value).reshape(batch, length, 5, 64)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.repeat_interleave(3, dim=2).transpose(1, 2),
            v.repeat_interleave(3, dim=2).transpose(1, 2), is_causal=False)
        x = x + self.self_attn.o_proj(attended.transpose(1, 2).reshape(batch, length, 960))
        value = self.post_attention_layernorm(x)
        return x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(value)) * self.mlp.up_proj(value))


class ContextDecoder(nn.Module):
    """Four transformer blocks contextualizing observation tokens."""

    def __init__(self, tokens=365):
        super().__init__()
        self.tokens = tokens
        self.layers = nn.ModuleList([ContextTransformerBlock() for _ in range(4)])

    def forward(self, context, projections=None):
        if context.shape[1:] != (self.tokens, 960) or (projections is not None and len(projections) != 8):
            raise ValueError(f"context decoder requires {self.tokens} tokens and eight K/V projections")
        positions = torch.arange(self.tokens, device=context.device)[None].expand(len(context), -1)
        # All tokens describe observations available now, so they can attend
        # bidirectionally. Actions and future observations never enter here.
        for layer in self.layers:
            # ACT fits in memory; recomputation is only needed by the FM path.
            context = (checkpoint(layer, context, positions, use_reentrant=False)
                       if projections is not None and self.training and torch.is_grad_enabled()
                       else layer(context, positions))
        return context, [] if projections is None else [projection(context) for projection in projections]


def preprocess_wrist_image(rgb, device):
    if rgb.shape != (320, 320, 3) or rgb.dtype.name != "uint8":
        raise ValueError("expected full, uncropped wrist RGB uint8 [320,320,3]")
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device)
    return (image.float() / 255 - .45) / .225


def vision_state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class VisionTransformerBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm1 = deepcopy(source.norm1)
        self.attn = deepcopy(source.attn)
        self.norm2 = deepcopy(source.norm2)
        self.mlp = deepcopy(source.mlp)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class WristVisionEncoder(nn.Module):
    """ViT-B/16 wrist features with optional ImageNet initialization."""

    tokens = 400

    def __init__(self, backbone):
        super().__init__()
        self.imagenet_normalization = False
        self.frozen = False
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
        self.blocks = nn.ModuleList([VisionTransformerBlock(block) for block in source.blocks])
        self.norm = deepcopy(source.norm)

    def load_vit_base(self, path):
        """Load torchvision ViT_B_16_Weights.IMAGENET1K_V1, without its classifier."""
        report = self.load_vit_state(torch.load(path, map_location="cpu", weights_only=True, mmap=True))
        return {**report, "path": str(path)}

    def load_vit_state(self, source):
        names = {"patch.weight": "conv_proj.weight", "patch.bias": "conv_proj.bias",
                 "cls_token": "class_token", "pos_embed": "encoder.pos_embedding",
                 "norm.weight": "encoder.ln.weight", "norm.bias": "encoder.ln.bias"}
        if len(self.blocks) != 12 or source["encoder.pos_embedding"].shape not in ((1, 197, 768), (1, 401, 768)):
            raise ValueError("expected torchvision ViT-B/16 at 224px or 320px")
        for i in range(12):
            for target, original in {
                "norm1": "ln_1", "norm2": "ln_2", "attn.qkv": "self_attention.in_proj",
                "attn.proj": "self_attention.out_proj", "mlp.fc1": "mlp.linear_1", "mlp.fc2": "mlp.linear_2",
            }.items():
                for suffix in ("weight", "bias"):
                    separator = "_" if original.endswith("in_proj") else "."
                    name = f"encoder.layers.encoder_layer_{i}.{original}{separator}{suffix}"
                    if original in ("mlp.linear_1", "mlp.linear_2"):
                        current = name.replace("mlp.linear_1", "mlp.0").replace("mlp.linear_2", "mlp.3")
                        name = current if current in source else name
                    names[f"blocks.{i}.{target}.{suffix}"] = name
        if set(source) - {"heads.head.weight", "heads.head.bias"} != set(names.values()):
            raise ValueError("unexpected torchvision ViT-B/16 weight keys")
        state = {target: source[original] for target, original in names.items()}
        if state["pos_embed"].shape[1] == 197:
            positions = state["pos_embed"][:, 1:].transpose(1, 2).reshape(1, 768, 14, 14)
            positions = F.interpolate(positions, size=(20, 20), mode="bicubic", align_corners=True)
            state["pos_embed"] = torch.cat((state["pos_embed"][:, :1], positions.flatten(2).transpose(1, 2)), dim=1)
        self.load_state_dict(state, strict=True)
        self.imagenet_normalization = True
        return {"source": "torchvision.ViT_B_16_Weights.IMAGENET1K_V1",
                "loaded_tensors": len(state), "excluded_tensors": ["heads.head.weight", "heads.head.bias"],
                "encoder_state_sha256": vision_state_sha256(self.state_dict()),
                "position_grid": "20x20 preserved" if source["encoder.pos_embedding"].shape[1] == 401 else "14x14 -> 20x20, bicubic, align_corners=True",
                "rgb_mean": [.485, .456, .406], "rgb_std": [.229, .224, .225]}

    def freeze(self):
        self.requires_grad_(False)
        self.frozen = True
        self.eval()

    def train(self, mode=True):
        return super().train(mode and not self.frozen)

    def forward(self, image):
        if image.shape[1:] != (3, 320, 320):
            raise ValueError("wrist encoder requires current RGB [B,3,320,320]")
        if self.imagenet_normalization:
            # Dataset/rollout tensors use TCOW normalization; convert before pretrained ViT.
            mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
            std = image.new_tensor([.229, .224, .225])[None, :, None, None]
            image = (image * .225 + .45 - mean) / std
        patches = self.patch(image).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(len(image), -1, -1), patches), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and torch.is_grad_enabled() else block(x)
        # Keep all 20x20 locations. CLS participates internally but is not sent to FM.
        return self.norm(x)[:, 1:]


class FlowMatchingHead(nn.Module):
    """Flow action expert with visual and proprioceptive conditioning."""

    flow_type = "dense_action_expert"
    horizon = 25
    action_representation = "absolute_joint"
    action_dim = 6

    def __init__(self, wrist_backbone=None, contextualize=False, horizon=25):
        super().__init__()
        if horizon < 1:
            raise ValueError("positive action horizon required")
        self.horizon = horizon
        self.visual = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256),
                                    nn.SiLU(), nn.Linear(256, 960))
        self.wrist_enabled = wrist_backbone is not None
        if contextualize and not self.wrist_enabled:
            raise ValueError("contextualized policy requires wrist vision")
        self.contextualize = contextualize
        self.wrist_tokens = 64 if contextualize else (400 if self.wrist_enabled else 0)
        self.context_tokens = 301 + self.wrist_tokens
        if contextualize:
            self.context_decoder = ContextDecoder()
        if self.wrist_enabled:
            self.wrist_encoder = WristVisionEncoder(wrist_backbone)
            self.wrist_projection = deepcopy(self.visual)
            self.overview_view_embedding = nn.Parameter(torch.randn(1, 1, 960) * .02)
            self.wrist_view_embedding = nn.Parameter(torch.randn(1, 1, 960) * .02)
        self.proprio = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, 256),
                                     nn.SiLU(), nn.Linear(256, 32))
        # Start the new proprio MLP as a residual on padded, normalized state.
        nn.init.zeros_(self.proprio[-1].weight)
        nn.init.zeros_(self.proprio[-1].bias)
        self.state_proj = nn.Linear(32, 960)
        self.action_in_proj = nn.Linear(self.action_dim, 720)
        self.action_out_proj = nn.Linear(720, self.action_dim)
        self.action_time_mlp_in = nn.Linear(1440, 720)
        self.action_time_mlp_out = nn.Linear(720, 720)
        self.layers = nn.ModuleList([ActionTransformerBlock(cross=i % 2 == 1) for i in range(16)])
        self.norm = RMSNorm(720)
        self.context_projections = nn.ModuleList([ContextKVProjection() for _ in range(8)])
        for name in ("state_mean", "action_mean"):
            self.register_buffer(name, torch.zeros(self.action_dim if name.startswith("action") else 6))
        for name in ("state_std", "action_std"):
            self.register_buffer(name, torch.ones(self.action_dim if name.startswith("action") else 6))

    @property
    def output(self):
        return self.action_out_proj

    def load_pretrained(self, path: Path):
        from safetensors.torch import load_file
        state = load_file(str(path))
        expected = {key for key in self.state_dict()
                    if not key.startswith(("visual.", "proprio.", "wrist_", "overview_", "context_decoder.", "state_mean", "state_std",
                                           "action_mean", "action_std"))}
        if state.keys() != expected:
            raise ValueError(f"action expert export keys differ: missing={sorted(expected - state.keys())}, "
                             f"extra={sorted(state.keys() - expected)}")
        if (state["action_in_proj.weight"].shape != (720, 32)
                or state["action_out_proj.weight"].shape != (32, 720)
                or state["action_out_proj.bias"].shape != (32,)):
            raise ValueError("expected SmolVLA export with 32 action coordinates")
        # Preserve pretrained weights for the six coordinates used by NexArm.
        state["action_in_proj.weight"] = state["action_in_proj.weight"][:, :self.action_dim].contiguous()
        state["action_out_proj.weight"] = state["action_out_proj.weight"][:self.action_dim].contiguous()
        state["action_out_proj.bias"] = state["action_out_proj.bias"][:self.action_dim].contiguous()
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != self.state_dict().keys() - expected:
            raise ValueError("unexpected action expert load result")
        return {"source": SOURCE_REPO, "revision": SOURCE_REVISION,
                "loaded_tensors": len(state), "loaded_parameters": sum(x.numel() for x in state.values()),
                "new_parameters": sum(v.numel() for k, v in self.named_parameters() if k not in expected),
                "blocks": 8, "expert_layers": 16, "width": 720,
                "action_dim": self.action_dim, "pretrained_action_dim": 32,
                "action_projection_initialization": "first six input columns and output rows; no action padding",
                "context_shape": [self.context_tokens, 960],
                "wrist_tokens": self.wrist_tokens, "contextualize": self.contextualize,
                "wrist_initialization": "TCOW spatial weights" if self.wrist_enabled else None,
                "interaction": "bidirectional SA + dense CA"}

    def load_prior(self, path):
        source = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if source["architecture"] != "oracle_task_target_fm" or source["config"]["action_representation"] != "absolute_joint":
            raise ValueError("absolute joint vision/FM prior required")
        if not self.wrist_enabled or not self.contextualize:
            raise ValueError("prior initialization requires wrist vision and context decoder")
        state = source["model"]
        if state["flow.action_out_proj.weight"].shape != (self.action_dim, 720):
            raise ValueError("six-coordinate prior head required; retrain the older padded prior")
        common = {key.removeprefix("flow."): value for key, value in state.items() if key.startswith("flow.")}
        expected = {key for key in self.state_dict() if not key.startswith(("visual.", "wrist_", "overview_", "context_decoder."))}
        if set(common) != expected:
            raise ValueError("prior FM parameter keys differ from main FM")
        self.load_state_dict(common, strict=False)
        visual = {key.removeprefix("visual_projection."): value for key, value in state.items() if key.startswith("visual_projection.")}
        self.visual.load_state_dict(visual, strict=True)
        self.wrist_projection.load_state_dict(visual, strict=True)
        decoder = {key.removeprefix("context_decoder."): value for key, value in state.items() if key.startswith("context_decoder.")}
        self.context_decoder.load_state_dict(decoder, strict=True)
        vision = {key.removeprefix("vision_encoder."): value for key, value in state.items() if key.startswith("vision_encoder.")}
        vision_report = self.wrist_encoder.load_vit_state(vision)
        with torch.no_grad():
            self.overview_view_embedding.copy_(state["view_embedding"][0:1])
            self.wrist_view_embedding.copy_(state["view_embedding"][1:2])
        statistics = dict(source["config"]["statistics"])
        self.set_statistics(statistics)
        return dict(source=str(path), source_step=source["step"], source_epoch=source["epoch"],
                    loaded_fm_tensors=len(common), loaded_visual_tensors=len(visual),
                    loaded_context_tensors=len(decoder), wrist_initialization=vision_report,
                    context_shape=[self.context_tokens, 960], normalization=statistics,
                    excluded=["task_embedding", "target_embedding", "overview vision backbone"],
                    overview_initialization="TCOW pretrained backbone; prior visual adapter")

    def encode_wrist(self, wrist):
        features = self.wrist_encoder(wrist)
        if self.contextualize:
            grid = features.transpose(1, 2).reshape(len(wrist), 768, 20, 20)
            features = F.adaptive_avg_pool2d(grid, (8, 8)).flatten(2).transpose(1, 2)
        return features

    def conditioning(self, latent, proprio, wrist=None, wrist_features=None):
        if latent.shape[1:] != (300, 768) or proprio.shape[1:] != (6,):
            raise ValueError("expected TCOW [B,300,768] and proprio [B,6]")
        state = (proprio - self.state_mean) / self.state_std
        padded = F.pad(state, (0, 26)) + self.proprio(state)
        visual = [self.visual(latent)]
        if self.wrist_enabled:
            visual[0] = visual[0] + self.overview_view_embedding
        if self.wrist_enabled:
            if wrist is None and wrist_features is None:
                raise ValueError("wrist-enabled policy requires current wrist RGB")
            if wrist_features is None:
                if len(wrist) != len(latent):
                    raise ValueError("wrist and TCOW batch sizes differ")
                wrist_features = self.encode_wrist(wrist)
            elif wrist is not None:
                raise ValueError("supply either wrist RGB or cached features")
            if wrist_features.shape != (len(latent), self.wrist_tokens, 768):
                raise ValueError("cached wrist feature shape differs from policy")
            visual.append(self.wrist_projection(wrist_features) + self.wrist_view_embedding)
        elif wrist is not None or wrist_features is not None:
            raise ValueError("checkpoint has no wrist encoder")
        context = torch.cat((*visual, self.state_proj(padded)[:, None]), dim=1)
        if self.contextualize:
            return self.context_decoder(context, self.context_projections)
        return context, [projection(context) for projection in self.context_projections]

    def set_statistics(self, statistics):
        if statistics.get("action_representation") != "absolute_joint":
            raise ValueError("absolute joint normalization required")
        for name in ("state_mean", "state_std", "action_mean", "action_std"):
            value = torch.as_tensor(statistics[name], dtype=torch.float32)
            expected = self.action_dim if name.startswith("action") else 6
            if value.shape != (expected,) or not torch.isfinite(value).all():
                raise ValueError(f"invalid normalization statistic: {name}")
            if name.endswith("std") and (value <= 0).any():
                raise ValueError(f"nonpositive normalization statistic: {name}")
            setattr(self, name, value.to(getattr(self, name).device).clone())

    def velocity(self, noisy_action, time, context_kv):
        if noisy_action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"action expert requires [B,{self.horizon},6] noisy actions")
        embedded = self.action_in_proj(noisy_action)
        fraction = torch.linspace(0, 1, 360, device=time.device,
                                  dtype=torch.float32 if time.device.type == "mps" else torch.float64)
        periods = .004 * (4 / .004) ** fraction
        # Public API keeps noise at time=0; pretrained SmolVLA uses noise at 1.
        phase = (1 - time.to(periods.dtype))[:, None] * (2 * math.pi / periods)[None]
        time_emb = torch.cat((phase.sin(), phase.cos()), dim=-1).to(embedded.dtype)
        x = torch.cat((embedded, time_emb[:, None].expand_as(embedded)), dim=-1)
        x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x)))
        positions = torch.arange(self.horizon, device=x.device)[None].expand(len(x), -1)
        for index, kv in enumerate(context_kv):
            x = self.layers[2 * index](x, positions)
            x = self.layers[2 * index + 1](x, positions, kv)
        return -self.action_out_proj(self.norm(x))

    def forward(self, latent, proprio, noisy_action, time, wrist=None, wrist_features=None):
        context, kv = self.conditioning(latent, proprio, wrist, wrist_features)
        return self.velocity(noisy_action, time, kv), context

    def sample_noise(self, batch, device, generator=None):
        return torch.randn((batch, self.horizon, self.action_dim), generator=generator, device=device)

    def training_path(self, action):
        if action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"expected H{self.horizon} actions with {self.action_dim} coordinates")
        normalized = (action - self.action_mean) / self.action_std
        noise = torch.randn_like(normalized)
        smol_time = torch.distributions.Beta(1.5, 1.0).sample((len(action),)).to(action.device) * .999 + .001
        time = 1 - smol_time
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * normalized
        return noisy, time, normalized - noise

    def sample(self, latent, proprio, noise, steps=10, wrist=None):
        _, kv = (self.conditioning(latent, proprio) if wrist is None else
                 self.conditioning(latent, proprio, wrist))
        return self.sample_actions(noise, kv, steps)

    @torch.no_grad()
    def sample_actions(self, noise, context_kv, steps=10):
        if steps < 1:
            raise ValueError("positive flow steps required")
        if noise.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"action noise must have shape [B,{self.horizon},6]")
        estimate = noise.float().clone()
        for index in range(steps):
            fraction = index / steps
            time = torch.full((len(noise),), fraction, device=noise.device)
            velocity = self.velocity(estimate, time, context_kv).float()
            estimate += velocity / steps
        return estimate * self.action_std + self.action_mean


class OATFlowPolicy(nn.Module):
    """Target-conditioned TCOW tracking and wrist/state-conditioned flow actions."""

    def __init__(self, seeker: nn.Module, flow: nn.Module):
        super().__init__()
        self.tcow = seeker
        self.flow = flow
        self._latent = None
        self.tcow.seeker.tracker_backbone.register_forward_hook(self._capture_backbone)

    def _capture_backbone(self, _module, _inputs, output):
        dense = output[0]
        self._latent = dense[:, :, -1].flatten(2).transpose(1, 2)

    def forward(self, rgbd, query, proprio=None, noisy_action=None, time=None, wrist=None):
        self._latent = None
        mask_logits, _flags = self.tcow(rgbd, query)
        if self._latent is None:
            raise RuntimeError("TCOW backbone hook did not capture dense features")
        latent = self._latent
        if noisy_action is None:
            return mask_logits, None, latent, None
        velocity, context = (self.flow(latent, proprio, noisy_action, time) if wrist is None else
                             self.flow(latent, proprio, noisy_action, time, wrist=wrist))
        return mask_logits, velocity, latent, context


def restore_policy(seeker, checkpoint):
    if (checkpoint["flow_type"] != "dense_action_expert" or checkpoint["action_representation"] != "absolute_joint"
            or not checkpoint["wrist_camera"] or not checkpoint["contextualize"]):
        raise ValueError("absolute joint TCOW/wrist/context-decoder checkpoint required")
    set_tracker_input(seeker)
    flow = FlowMatchingHead(wrist_backbone=seeker.seeker.tracker_backbone, contextualize=True)
    if checkpoint.get("wrist_camera", False):
        flow.wrist_encoder.imagenet_normalization = checkpoint.get("wrist_imagenet_normalization", False)
        if checkpoint.get("wrist_encoder_frozen", False):
            flow.wrist_encoder.freeze()
    model = OATFlowPolicy(seeker, flow)
    model.load_state_dict(checkpoint["model"], strict=True)
    if checkpoint.get("tcow_frozen", False):
        model.tcow.requires_grad_(False)
        model.tcow.eval()
    return model


def load_training_tracker(seeker, source):
    if "model" in source:
        state = {key.removeprefix("tcow."): value for key, value in source["model"].items()
                 if key.startswith("tcow.")}
        step = source["step"]
    else:
        state, step = dict(source["net_seeker"]), source.get("source_step", 0)
    key = "seeker.tracker_backbone.timesformer.model.patch_embed.proj.weight"
    patch = state[key]
    target_channels = seeker.seeker.tracker_backbone.timesformer.model.patch_embed.proj.in_channels
    if target_channels != 4 or patch.shape[1] not in (4, 5):
        raise ValueError("new training requires RGB+query TCOW (4 patch channels)")
    if patch.shape[1] == 5:
        # RGB and query stay bit-identical; the learned depth kernel is discarded.
        state[key] = patch[:, [0, 1, 2, 4]].contiguous()
    seeker.load_state_dict(state, strict=True)
    return step


def checkpoint_uses_depth(checkpoint):
    key = "seeker.tracker_backbone.timesformer.model.patch_embed.proj.weight"
    state = checkpoint.get("net_seeker")
    patch = state[key] if state is not None else checkpoint["model"]["tcow." + key]
    channels = patch.shape[1]
    if channels not in (4, 5):
        raise ValueError("expected RGB+query or historical RGB-D+query checkpoint")
    visual_input = "rgbd" if channels == 5 else "rgb"
    if checkpoint.get("visual_input", visual_input) != visual_input:
        raise ValueError("visual_input metadata disagrees with patch weights")
    return channels == 5


def set_tracker_input(seeker):
    backbone = seeker.seeker.tracker_backbone
    patch = backbone.timesformer.model.patch_embed
    old = patch.proj
    if old.in_channels not in (4, 5):
        raise ValueError("expected four or five TCOW patch channels")
    if old.in_channels == 5:
        new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride,
                        old.padding, old.dilation, old.groups, old.bias is not None,
                        device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(old.weight[:, [0, 1, 2, 4]])
            if old.bias is not None:
                new.bias.copy_(old.bias)
        patch.proj = new
    seeker.seeker.input_channels = 4
    backbone.Ci = 4
    return seeker
