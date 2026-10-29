"""Frozen DINOv3 normalized patch tokens, no CLS/register tokens or mask inputs."""

from unified_geometry.paths import HF_CACHE
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

WEIGHTS = (
    HF_CACHE
    / "models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51"
)
CANDIDATES = ["block6", "block12", "block18", "block24", "mean_last4"]
RESOLUTION = 1152
DIM = 1024


class Features:
    def __init__(self, K, T, selection=None):
        self.model = (
            AutoModel.from_pretrained(
                WEIGHTS,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        self.processor = AutoImageProcessor.from_pretrained(WEIGHTS, local_files_only=True)
        self.K = K
        self.T = T
        self.selection = selection
        self.max_projection_error = 0.0
        self.checked = False
        assert (
            self.model.config.hidden_size == DIM
            and self.model.config.patch_size == 16
            and (self.model.config.num_hidden_layers == 24)
        )

    @torch.inference_mode()
    def __call__(self, samples, all_candidates=False):
        images = []
        grids = []
        for d in samples:
            images.append(Image.fromarray(d["rgb"]))
            p = d["scene_points"].astype(float)
            camera = (p - self.T[:3, 3]) @ self.T[:3, :3]
            pixel = camera @ self.K.T
            uv = pixel[:, :2] / pixel[:, 2:]
            self.max_projection_error = max(
                self.max_projection_error, float(np.abs(uv - d["pixel_uv"]).max())
            )
            grids.append(np.c_[(uv[:, 0] + 0.5) / 320 * 2 - 1, (uv[:, 1] + 0.5) / 240 * 2 - 1])
        inputs = self.processor(
            images=images,
            size={"height": RESOLUTION, "width": RESOLUTION},
            return_tensors="pt",
        )["pixel_values"].to("cuda", dtype=torch.bfloat16)
        assert inputs.shape[-2:] == (RESOLUTION, RESOLUTION)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(pixel_values=inputs, output_hidden_states=True)
        hidden = output.hidden_states
        assert len(hidden) == 25
        prefix = 1 + self.model.config.num_register_tokens
        side = RESOLUTION // 16
        assert output.last_hidden_state.shape[1:] == (prefix + side * side, DIM)

        def normalized(value):
            return F.layer_norm(
                value.float(),
                (DIM,),
                self.model.norm.weight.float(),
                self.model.norm.bias.float(),
                self.model.norm.eps,
            )

        if not self.checked:
            torch.testing.assert_close(
                normalized(hidden[24]), output.last_hidden_state, rtol=0, atol=0
            )
            self.checked = True
        names = CANDIDATES if all_candidates else [self.selection]
        assert all((name in CANDIDATES for name in names))
        grid = torch.tensor(np.stack(grids), device="cuda", dtype=torch.float32)[:, :, None, :]
        results = {}
        for name in names:
            if name == "mean_last4":
                tokens = sum((normalized(hidden[i]) for i in [21, 22, 23, 24])) / 4
            else:
                tokens = normalized(hidden[int(name[5:])])
            patch = tokens[:, prefix:].transpose(1, 2).reshape(len(samples), DIM, side, side)
            sampled = (
                F.grid_sample(
                    patch,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                .squeeze(-1)
                .transpose(1, 2)
            )
            results[name] = sampled.half()
        return results if all_candidates else results[self.selection]
