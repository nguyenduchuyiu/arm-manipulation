"""Original E135 procedural split distributions and rejection sampler."""

import numpy as np
from scipy.spatial.transform import Rotation


def quadrant(position):
    return int(position[0] >= 0.53) + 2 * int(position[1] >= -0.25)


def parameters(rng, split):
    """Independent per-cube draws, conditional only on split, visibility and separation."""
    rows = []
    for color in range(4):
        for _ in range(1000):
            position = rng.uniform([0.36, -0.37, 0.035], [0.7, -0.13, 0.125])
            if split == "positions":
                band = int(rng.integers(6))
                axis, lo, hi = [
                    (0, 0.29, 0.34),
                    (0, 0.72, 0.77),
                    (1, -0.43, -0.39),
                    (1, -0.11, -0.07),
                    (2, 0.145, 0.165),
                    (2, 0.17, 0.19),
                ][band]
                position[axis] = rng.uniform(lo, hi)
            q = quadrant(position)
            if (q == color) != (split == "combinations"):
                continue
            side = float(rng.uniform(0.024, 0.05))
            if split == "sizes":
                side = float(
                    rng.uniform(0.014, 0.022) if rng.random() < 0.5 else rng.uniform(0.054, 0.068)
                )
            angles = rng.uniform([-10, -10, -20], [10, 10, 20])
            if split == "rotations":
                angles = rng.uniform([20, 20, 30], [35, 35, 45]) * rng.choice([-1, 1], 3)
            rotation = Rotation.from_euler("xyz", angles, degrees=True)
            half_height = np.abs(rotation.as_matrix()[2]).sum() * side / 2
            if position[2] - half_height < 0.004:
                continue
            rows.append(
                dict(
                    position=position.tolist(),
                    side_m=side,
                    euler_xyz_deg=angles.tolist(),
                    quaternion_wxyz=np.roll(rotation.as_quat(), 1).tolist(),
                    quadrant=q,
                )
            )
            break
        else:
            return None
    positions = np.array([r["position"] for r in rows])
    radii = np.array([r["side_m"] for r in rows]) * np.sqrt(3) / 2
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    if np.any((distances < radii[:, None] + radii[None] + 0.008) & ~np.eye(4, dtype=bool)):
        return None
    return rows
