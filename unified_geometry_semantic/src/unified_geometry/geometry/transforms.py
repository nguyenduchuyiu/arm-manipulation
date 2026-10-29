"""Haar SO(3) and shared coordinate transforms, retained from E137."""

import torch


def sample_se3(count, generator, device="cuda", dtype=torch.float32, d=0.5):
    q = torch.randn(count, 4, generator=generator, device=device, dtype=dtype)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(count, 3, 3)
    t = (torch.rand(count, 3, generator=generator, device=device, dtype=dtype) * 2 - 1) * d
    return (R, t)


def apply(points, R, t):
    """Column-vector p'=R p+t, implemented for row-vector arrays B×...×3."""
    shape = points.shape
    return (torch.bmm(points.reshape(shape[0], -1, 3), R.transpose(1, 2)) + t[:, None]).reshape(
        shape
    )


def inverse(points, R, t):
    shape = points.shape
    return torch.bmm(points.reshape(shape[0], -1, 3) - t[:, None], R).reshape(shape)
