"""Pure geometric label/output operations; no object pose or canonical rotation."""


def geometry(points):
    center = points.float().mean(-2)
    centered = points.float() - center[..., None, :]
    radius = centered.square().sum(-1).mean(-1).sqrt()
    return (center, radius)


def unit_shape(points):
    center, radius = geometry(points)
    assert bool((radius > 1e-08).all()), "Cannot normalize a collapsed point cloud"
    return (points.float() - center[..., None, :]) / radius[..., None, None]


def world_cloud(shape, center, radius):
    return center[..., None, :] + radius[..., None, None] * shape
