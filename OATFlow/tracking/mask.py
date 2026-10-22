"""Original causal TCOW backbone and native target-mask patch readout."""
import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.tracking.backbone import CausalBackbone
from OATFlow.tracking.point import load_tcow


class TargetMaskTCOW(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = dict(image_size=[240, 320], patch_size=16, max_frames=30,
                           width=768, depth=12, heads=12, drop_path_rate=.1,
                           output_channels=1, flag_channels=0, track_map_stride=4)
        self.backbone = CausalBackbone((240, 320), 16, 30, 768, 12, 12,
                                       in_channels=4, drop_path_rate=.1)
        self.tracker_post_linear = nn.Linear(768, 16 * 16)

    def features(self, rgb, query):
        if rgb.ndim != 5 or rgb.shape[1] != 3 or rgb.shape[-2:] != (240, 320):
            raise ValueError('expected RGB [B,3,T,240,320]')
        b, _, t, h, w = rgb.shape
        if not 1 <= t <= 30 or query.shape != (b, 1, t, h, w):
            raise ValueError('expected query [B,1,T,240,320], T<=30')
        if query[:, :, 1:].count_nonzero() or not (query[:, :, 0].sum((-1, -2)) > 0).all():
            raise ValueError('one nonempty frame0 mask prompt; zero query at later frames')
        pixels = rgb.float() / 255 if rgb.dtype == torch.uint8 else rgb.float()
        pixels = torch.cat(((pixels - .45) / .225, query.float()), dim=1)
        x, cls = self.backbone.embed(pixels.permute(0, 2, 1, 3, 4)[:, None])
        for block in self.backbone.blocks:
            x, cls = block(x, cls)
        return x[:, 0]  # [B,T,300,768], no pooling or learned context decoder.

    def masks_from_features(self, features):
        b, t, n, _ = features.shape
        if n != 300:
            raise ValueError('expected 15x20 spatial patches')
        logits = self.tracker_post_linear(features)
        logits = logits.reshape(b, t, 15, 20, 16, 16).permute(0, 1, 2, 4, 3, 5)
        logits = logits.reshape(b * t, 1, 240, 320)
        # Preserve the pretrained TCOW readout's smoothing and interpolation.
        logits = F.avg_pool2d(logits, 4, 4)
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=True)
        return logits.reshape(b, t, 240, 320)

    def forward(self, rgb, query):
        return self.masks_from_features(self.features(rgb, query))


def load_original(model, weights):
    load_tcow(model, weights / 'tracking_backbone.pt', weights / 'tracking_query.pt')
    model.tracker_post_linear.load_state_dict(torch.load(
        weights / 'tracking_target_mask.pt', map_location='cpu', weights_only=True), strict=True)


def mask_loss(logits, truth, visible, occlusion, progress):
    """TCOW target loss: 0.2 weighted BCE + 0.4 bootstrapped BCE + 0.4 soft IoU."""
    logits, truth = logits.float(), truth.float()
    bce = F.binary_cross_entropy_with_logits(logits, truth, reduction='none')
    pos_frac = truth.mean().clamp_min(.05)
    neg_frac = (1 - truth).mean().clamp_min(.05)
    ratio = pos_frac / neg_frac
    positive = torch.where(pos_frac > neg_frac, ratio.pow(-.7), ratio.pow(-.3))
    negative = torch.where(pos_frac > neg_frac, ratio.pow(.3), ratio.pow(.7))
    weights = torch.where(truth.bool(), positive, negative)
    weights = weights * torch.where(truth.bool() & ~visible.bool(), 2., 1.)
    b, t, h, w = truth.shape
    band = int((h * w) ** .5 / 12)
    band += band % 2 == 0
    # Gaussian blur > 0 has the same support as binary dilation (all taps positive).
    enlarged = F.max_pool2d(truth.reshape(b * t, 1, h, w), band, 1, band // 2).reshape_as(truth) > 0
    weights = weights * torch.where(enlarged & ~truth.bool(), 3., 1.)
    frame_weights = (occlusion.float() * 5).clamp_min(1)
    frame_weights[:, 0] *= .2
    weights = weights * frame_weights[..., None, None]
    weighted_bce = (bce * weights).mean()
    fraction = min(max(1 - progress * 8.5, .15), 1.)
    bootstrap = bce.flatten().topk(max(1, int(fraction * bce.numel())), sorted=False).values.mean()
    probability = logits.sigmoid()
    intersection = (probability * truth).sum()
    union = (probability + truth - probability * truth).sum()
    soft_iou = 1 - intersection / (union + .1)
    return .2 * weighted_bce + .4 * bootstrap + .4 * soft_iou, weighted_bce, bootstrap, soft_iou


def frame_iou(logits, truth):
    prediction, truth = logits > 0, truth.bool()
    intersection = (prediction & truth).sum((-1, -2)).float()
    union = (prediction | truth).sum((-1, -2))
    return torch.where(union > 0, intersection / union.clamp_min(1), 1.)
