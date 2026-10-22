"""Frame0 click annotations and top-camera causal clips; no training mask files."""
import json

import mujoco
import numpy as np
import torch

from OATFlow.tracking.data import load_episodes, sample_indices


def top_camera(root):
    model = mujoco.MjModel.from_xml_path(str(root / 'scene.xml'))
    state = mujoco.MjData(model); mujoco.mj_forward(model, state)
    camera = model.camera('cam_top').id
    rotation = state.cam_xmat[camera].reshape(3, 3) @ np.diag([1., -1., -1.])
    return dict(position=state.cam_xpos[camera].copy(), rotation=rotation,
                focal=160 / np.tan(np.deg2rad(model.cam_fovy[camera]) / 2),
                base=np.array(json.loads((root / 'config.json').read_text())['base_origin_world_m']))


def project(xyz_base_m, camera):
    xyz = (np.asarray(xyz_base_m) + camera['base'] - camera['position']) @ camera['rotation']
    if not (xyz[..., 2] > 0).all():
        raise ValueError('query point is behind camera')
    uv = camera['focal'] * xyz[..., :2] / xyz[..., 2:3] + 159.5
    uv[..., 1] -= 40
    return uv


def load_points(root, rows, segments=None, evaluation_labels=False, stride=3):
    episodes, segments = load_episodes(root, rows, stride=stride, segments=segments,
                                      evaluation_labels=evaluation_labels, views=('top',), center_crop=True)
    camera = top_camera(root)
    for ep in episodes:
        center = ep['context']['xyz'][0, ep['row']['target_id']]
        uv = project(center, camera).astype(np.float32)
        if not ((uv >= 0).all() and (uv < [320, 240]).all()):
            raise ValueError('frame0 query outside top image')
        ep['query_uv'] = uv
    return episodes, segments


def point_clip(episode, end, frames=30):
    if frames == 1 and end != 0:
        raise ValueError('single-frame point clip must contain frame0')
    indices = sample_indices(end, frames)
    nc = len(episode['context']['time'])
    prefix = indices < nc
    rgb = torch.empty((frames, 3, 240, 320), dtype=torch.uint8)
    for source, selected, offset in ((episode['context'], prefix, 0), (episode['expert'], ~prefix, nc)):
        if selected.any():
            rgb[torch.from_numpy(np.flatnonzero(selected))] = source['rgb'][0, torch.from_numpy(indices[selected] - offset)]
    query = torch.zeros((1, frames, 240, 320))
    x, y = np.rint(episode['query_uv']).astype(int)
    yy, xx = torch.meshgrid(torch.arange(max(0, y - 3), min(240, y + 4)),
                            torch.arange(max(0, x - 3), min(320, x + 4)), indexing='ij')
    query[0, 0, yy, xx] = ((xx - x).square() + (yy - y).square() <= 9).float()
    return (rgb.permute(1, 0, 2, 3)[None], query[None],
            torch.from_numpy(episode['xyz'][indices].copy())[None])
