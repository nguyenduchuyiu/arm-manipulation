"""CPU geometry metrics, unsquared distances in millimeters."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from unified_geometry.metrics.coverage import extent, coverage


def cd(p, g):
    d = cdist(p, g)
    return float(0.5 * (d.min(0).mean() + d.min(1).mean()) * 1000)


def score(p, g, surface):
    p = p.astype(float) * 1000
    g = g.astype(float) * 1000
    surface = surface.astype(float) * 1000
    d = cdist(p, g)
    r, c = linear_sum_assignment(d)
    full = cdist(p, surface)
    cov, _, _, _ = coverage(p, g, surface, 4.0)
    rows = {x["tolerance_mm"]: x for x in cov["rows"]}
    pc = p - p.mean(0)
    gc = g - g.mean(0)
    pr = np.sqrt(np.square(pc).sum(1).mean())
    gr = np.sqrt(np.square(gc).sum(1).mean())
    unit = cdist(pc / pr, gc / gr)
    ur, uc = linear_sum_assignment(unit)
    return dict(
        emd_mm=float(d[r, c].mean()),
        unit_shape_EMD=float(unit[ur, uc].mean()),
        unit_shape_CD=float(0.5 * (unit.min(0).mean() + unit.min(1).mean())),
        Rg_absolute_error_mm=float(abs(pr - gr)),
        Rg_prediction_mm=float(pr),
        Rg_GT_mm=float(gr),
        pred_to_GT128_mm=float(d.min(1).mean()),
        GT128_to_pred_mm=float(d.min(0).mean()),
        pred_to_visibleGT_mm=float(full.min(1).mean()),
        visibleGT_to_pred_mm=float(full.min(0).mean()),
        prediction_extent_mm=extent(p)["min_max_xyz_mm"],
        GT128_extent_mm=extent(g)["min_max_xyz_mm"],
        prediction_robust_extent_mm=extent(p)["q95_minus_q05_xyz_mm"],
        GT128_robust_extent_mm=extent(g)["q95_minus_q05_xyz_mm"],
        visibleGT_robust_extent_mm=extent(surface)["q95_minus_q05_xyz_mm"],
        coverage_2mm_percent=rows[2.0]["prediction_percent"],
        coverage_5mm_percent=rows[5.0]["prediction_percent"],
        GT128_reference_coverage_2mm_percent=rows[2.0]["GT128_reference_percent"],
        GT128_reference_coverage_5mm_percent=rows[5.0]["GT128_reference_percent"],
    )


def stack(records):
    return {key: np.asarray([r[key] for r in records]) for key in records[0]}


def summarize(records):
    arrays = stack(records)
    result = dict(
        queries=len(records),
        **{
            key: value.mean(0).tolist() if value.ndim > 1 else float(value.mean())
            for key, value in arrays.items()
        },
    )
    result["robust_extent_MAE_xyz_mm"] = (
        np.abs(arrays["prediction_robust_extent_mm"] - arrays["GT128_robust_extent_mm"])
        .mean(0)
        .tolist()
    )
    p, g = (arrays["Rg_prediction_mm"], arrays["Rg_GT_mm"])
    den = float(np.square(g - g.mean()).sum())
    result["Rg_R2"] = float(1 - np.square(p - g).sum() / den) if den > 1e-12 else None
    return result


def gain(predicted, expected):
    predicted = np.asarray(predicted)
    expected = np.asarray(expected)
    denominator = float((expected * expected).sum())
    return float((predicted * expected).sum() / max(denominator, 1e-12))
