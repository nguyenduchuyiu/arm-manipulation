"""Synthetic 175-degree diagonal equidistant wrist RGB camera, not measured lens calibration."""
import cv2
import mujoco
import numpy as np


DIAGONAL_FOV = 175.
FACE_SIZE = 256
FOCAL = np.hypot(320, 240)/np.deg2rad(DIAGONAL_FOV)
INTRINSICS = np.array([[FOCAL, 0., 159.5], [0., FOCAL, 119.5], [0., 0., 1.]])
REFERENCE = 'Arducam B0261 IMX291, nominal175deg diagonal; synthetic equidistant, not measured calibration'


class Fisheye:
    def __init__(self, model):
        self.model = model
        self.camera = model.camera('wrist').id
        self.rgb = mujoco.Renderer(model, height=FACE_SIZE, width=FACE_SIZE)
        samples = model.vis.quality.offsamples; model.vis.quality.offsamples = 0
        self.seg = mujoco.Renderer(model, height=FACE_SIZE, width=FACE_SIZE)
        model.vis.quality.offsamples = samples
        self.seg.enable_segmentation_rendering()
        yy, xx = np.indices((240, 320))
        xy = np.stack([xx-159.5, yy-119.5], axis=-1)/FOCAL
        theta = np.linalg.norm(xy, axis=-1)
        rays = np.concatenate([xy*np.sinc(theta[..., None]/np.pi), np.cos(theta[..., None])], axis=-1)
        # Face axes are expressed in the wrist optical frame (+X right, +Y down, +Z forward).
        rotations = [np.eye(3), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
                     np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
                     np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                     np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])]
        axis = np.argmax(np.abs(rays), axis=-1)
        face = np.where(axis == 2, 0, np.where(axis == 0,
                        np.where(rays[..., 0] > 0, 1, 2), np.where(rays[..., 1] > 0, 3, 4)))
        self.faces = []
        for i, rotation in enumerate(rotations):
            mask = face == i; local = rays[mask]@rotation
            uv = FACE_SIZE/2*local[:, :2]/local[:, 2:]+(FACE_SIZE-1)/2
            assert np.all((uv >= -.501) & (uv <= FACE_SIZE-.499))
            uv = np.clip(uv, 0, FACE_SIZE-1).astype('float32')
            maps = np.zeros((240, 320, 2), 'float32'); maps[mask] = uv
            self.faces.append((rotation, mask, maps[..., 0].copy(), maps[..., 1].copy()))

    def render(self, data, option, segmentation=False):
        renderer = self.seg if segmentation else self.rgb
        renderer.update_scene(data, camera='wrist', scene_option=option)
        world = data.cam_xmat[self.camera].reshape(3, 3)@np.diag([1., -1., -1.])
        result = np.empty((240, 320, 2 if segmentation else 3), 'int32' if segmentation else 'uint8')
        for rotation, mask, mx, my in self.faces:
            axes = world@rotation
            for camera in renderer.scene.camera:
                camera.pos[:] = data.cam_xpos[self.camera]
                camera.forward[:] = axes[:, 2]; camera.up[:] = -axes[:, 1]
                camera.frustum_near = .001; camera.frustum_far = 10.
                camera.frustum_bottom = -.001; camera.frustum_top = .001
                camera.frustum_center = 0.; camera.frustum_width = 0.
                camera.orthographic = 0
            source = renderer.render()
            if segmentation:
                result[mask] = source[np.rint(my[mask]).astype(int), np.rint(mx[mask]).astype(int)]
            else:
                result[mask] = cv2.remap(source, mx, my, cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)[mask]
        return result

    def close(self):
        self.rgb.close(); self.seg.close()
