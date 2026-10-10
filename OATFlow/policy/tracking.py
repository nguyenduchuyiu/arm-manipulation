"""Load untouched upstream TCOW and apply its three-mask training loss."""
from pathlib import Path
import sys

import torch
from torch.utils.checkpoint import checkpoint


TCOW = Path(__file__).resolve().parents[1] / "third_party/tcow"
sys.path[:0] = [str(TCOW / "model"), str(TCOW / "eval"), str(TCOW / "utils"), str(TCOW),
                str(TCOW / "third_party/TimeSformer")]
from seeker import Seeker as Seeker  # noqa: E402
from loss import MyLosses as MyLosses  # noqa: E402


def checkpoint_transformer_blocks(model):
    # Recompute each unchanged upstream block during backward to bound VRAM.
    blocks = model.seeker.tracker_backbone.timesformer.model.blocks
    for block in blocks:
        forward = block.forward
        block.forward = lambda x, batch, frames, width, forward=forward: checkpoint(
            forward, x, batch, frames, width, use_reentrant=False)


def original_mask_loss(losses, logits, truth, progress, frame_valid=None):
    occlusion = truth[:, 1].any(dim=(-1, -2)).float()
    fractions = torch.zeros((*occlusion.shape[:1], 1, occlusion.shape[1], 3), device=truth.device)
    fractions[:, 0, :, 0] = occlusion
    target = truth[:, 0, None]
    covered_pixels = (truth[:, 0, None] * truth[:, 1, None]).to(torch.uint8)
    # Upstream downweights the query frame only for its last batch element.
    track_weights = torch.cat([
        losses.get_mask_track_frame_weights(item[None], 0)
        for item in fractions
    ], dim=0)[..., None, None]
    track_weights = track_weights * losses.get_mask_track_pixel_weights(
        fractions, target, covered_pixels)
    if frame_valid is not None:
        track_weights = track_weights * frame_valid[:, None, :, None, None]
    total = losses.my_mask_loss(logits[:, 0, None].float(), target, track_weights,
                                progress, False) * losses.train_args.track_lw
    for channel, scale in ((1, losses.train_args.occl_mask_lw),
                           (2, losses.train_args.cont_mask_lw)):
        positive = truth[:, channel].any(dim=(-1, -2))
        weight = torch.where(positive, 1.0, losses.train_args.occl_cont_zero_weight)
        weight = weight[:, None, :, None, None].expand_as(truth[:, channel, None])
        if frame_valid is not None:
            weight = weight * frame_valid[:, None, :, None, None]
        total = total + scale * losses.my_mask_loss(
            logits[:, channel, None].float(), truth[:, channel, None], weight,
            progress, True)
    return total
