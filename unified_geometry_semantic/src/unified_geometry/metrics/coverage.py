"""Visible-surface voxel coverage and robust extents."""

import numpy as np
from scipy.spatial import cKDTree

TAUS = [1.0, 2.0, 5.0]


def distances(source, target):
    return cKDTree(target).query(source)[0]


def representatives(surface, size):
    origin = surface.min(0)
    keys = np.floor((surface - origin) / size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    result = []
    for index in range(inverse.max() + 1):
        points = surface[inverse == index]
        result.append(points[np.linalg.norm(points - points.mean(0), axis=1).argmin()])
    return np.array(result)


def extent(points):
    q = np.quantile(points, [0.05, 0.95], axis=0, method="linear")
    return dict(
        min_max_xyz_mm=np.ptp(points, axis=0).tolist(),
        q05_xyz_mm=q[0].tolist(),
        q95_xyz_mm=q[1].tolist(),
        q95_minus_q05_xyz_mm=(q[1] - q[0]).tolist(),
    )


def coverage(points, reference, surface, size):
    reps = representatives(surface, size)
    dp, dr = (distances(reps, points), distances(reps, reference))
    assert np.allclose(dp, distances(reps, np.repeat(points, 2, axis=0)), atol=1e-12, rtol=0)
    bins = np.linspace(surface[:, 2].min(), surface[:, 2].max(), 6)
    ids = np.clip(np.searchsorted(bins, reps[:, 2], side="right") - 1, 0, 4)
    rows = []
    for tau in TAUS:
        regions = []
        for index in range(5):
            mask = ids == index
            regions.append(
                dict(
                    z_lower_mm=float(bins[index]),
                    z_upper_mm=float(bins[index + 1]),
                    regions=int(mask.sum()),
                    prediction_percent=float((dp[mask] <= tau).mean() * 100)
                    if mask.any()
                    else None,
                    GT128_reference_percent=float((dr[mask] <= tau).mean() * 100)
                    if mask.any()
                    else None,
                )
            )
        rows.append(
            dict(
                tolerance_mm=tau,
                covered_regions=int((dp <= tau).sum()),
                regions=len(reps),
                prediction_percent=float((dp <= tau).mean() * 100),
                GT128_reference_percent=float((dr <= tau).mean() * 100),
                z_bins=regions,
            )
        )
    return (dict(voxel_mm=size, rows=rows), reps, dp, dr)
