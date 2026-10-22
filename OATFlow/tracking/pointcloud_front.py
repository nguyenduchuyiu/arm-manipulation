"""Front-oblique RGB-D replay and causal inference with the unchanged top-trained TCOW."""
import mujoco
import numpy as np
import torch
from tqdm import tqdm

from OATFlow.tracking.data import sample_indices
from OATFlow.tracking.mask import TargetMaskTCOW


POSITION = np.array([.54, -.66, .60])
LOOK_AT = np.array([.54, -.12, .035])
FOVY = 55.


def configure(model, state, base):
    # Reuse a fixed world camera slot; its orientation is independent of target pose.
    cid = model.camera('cam_left').id
    assert model.cam_bodyid[cid] == 0
    backward = POSITION-LOOK_AT
    backward /= np.linalg.norm(backward)
    right = np.cross([0., 0., 1.], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    rotation = np.stack((right, up, backward), axis=1)
    model.cam_pos[cid] = POSITION
    mujoco.mju_mat2Quat(model.cam_quat[cid], rotation.ravel())
    model.cam_fovy[cid] = FOVY
    mujoco.mj_forward(model, state)
    return dict(position=state.cam_xpos[cid].copy(),
                rotation=state.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.]),
                focal=160/np.tan(np.deg2rad(FOVY)/2), base=np.asarray(base))


def load_tracker(checkpoint):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert (saved['epoch'], saved['step']) == (9, 4608)
    model = TargetMaskTCOW()
    model.load_state_dict(saved['model'], strict=True)
    return model.eval().requires_grad_(False).cuda()


def render(model, state, qpos, endpoints, body, name):
    rgb = np.empty((len(qpos), 240, 320, 3), np.uint8)
    depth = np.empty((len(endpoints), 240, 320), np.float32)
    truth = np.empty((len(endpoints), 240, 320), bool)
    hidden = np.empty(len(endpoints), bool)
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    only = mujoco.MjvOption(); only.geomgroup[:] = 0; only.geomgroup[5] = 1
    geoms = np.flatnonzero((model.geom_bodyid == body) & (model.geom_rgba[:, 3] > 0))
    original_groups = model.geom_group[geoms].copy()
    k = 0
    with mujoco.Renderer(model, height=320, width=320) as renderer:
        samples = model.vis.quality.offsamples
        model.vis.quality.offsamples = 0
        segmentation = mujoco.Renderer(model, height=320, width=320)
        model.vis.quality.offsamples = samples
        with segmentation:
            segmentation.enable_segmentation_rendering()
            for frame in tqdm(range(len(qpos)), desc=f'{name}: render front RGB-D', mininterval=5):
                state.qpos[:] = qpos[frame]; mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera='cam_left', scene_option=option)
                rgb[frame] = renderer.render()[40:280]
                if k == len(endpoints) or frame != endpoints[k]:
                    continue
                renderer.enable_depth_rendering()
                depth[k] = renderer.render()[40:280]
                renderer.disable_depth_rendering()
                segmentation.update_scene(state, camera='cam_left', scene_option=option)
                seg = segmentation.render()[40:280]
                visible = np.isin(seg[..., 0], geoms) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                if frame == 0:
                    prompt = visible.copy()
                    assert prompt.sum() > 20
                # Additional GT render only for IoU/visibility evaluation, never tracker input.
                model.geom_group[geoms] = 5
                segmentation.update_scene(state, camera='cam_left', scene_option=only)
                truth[k] = segmentation.render()[40:280, :, 0] >= 0
                model.geom_group[geoms] = original_groups
                assert truth[k].any()
                hidden[k] = visible.sum() / truth[k].sum() <= .05
                k += 1
    assert k == len(endpoints)
    return rgb, depth, prompt, truth, hidden


@torch.inference_mode()
def predict(model, images, endpoints, prompt, name):
    sampled = images[endpoints]
    predictions = []
    for start in tqdm(range(0, len(endpoints), 4), desc=f'{name}: causal front TCOW', mininterval=5):
        ends = range(start, min(start+4, len(endpoints)))
        indices = np.stack([sample_indices(end, 30) for end in ends])
        assert (indices[:, -1] == np.arange(start, start+len(indices))).all()
        rgb = torch.from_numpy(sampled[indices].transpose(0, 4, 1, 2, 3).copy()).cuda()
        query = torch.zeros((len(indices), 1, 30, 240, 320), dtype=torch.bool, device='cuda')
        query[:, 0, 0] = torch.from_numpy(prompt).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(rgb, query)
        assert torch.isfinite(logits).all()
        predictions.append((logits[:, -1] > 0).cpu().numpy())
    return np.concatenate(predictions)
