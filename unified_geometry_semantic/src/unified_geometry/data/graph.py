"""Same observation-derived canonical UV16 graph as E175, no new graph design."""

import numpy as np
from scipy.spatial import cKDTree


def graph(pixel_uv):
    assert len(np.unique(pixel_uv, axis=0)) == len(pixel_uv)
    order = np.lexsort((pixel_uv[:, 0], pixel_uv[:, 1]))
    ordered = pixel_uv[order].astype(float)
    _, index = cKDTree(ordered).query(ordered, k=17, workers=1)
    assert np.array_equal(index[:, 0], np.arange(len(index)))
    result = np.empty((len(order), 16), np.uint16)
    result[order] = order[index[:, 1:]]
    return result
