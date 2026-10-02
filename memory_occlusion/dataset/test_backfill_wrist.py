"""Check saved-state rendering geometry and the legacy gripper reconstruction."""
import unittest

import mujoco
import numpy as np

from controllers.nexarm_mujoco_backend import JOINT_NAMES, MUJOCO_JOINTS
from memory_occlusion.dataset.backfill_wrist import boundary_error, physical_positions, restore_pose
from memory_occlusion.environment.env import SCENE_XML, TARGETS, configure_mvp_model
from memory_occlusion.task import normalized


class BackfillTest(unittest.TestCase):
    def test_saved_state_camera_and_passive_gripper(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        configure_mvp_model(model)
        original, restored = mujoco.MjData(model), mujoco.MjData(model)
        addresses = [model.jnt_qposadr[model.joint(MUJOCO_JOINTS[name]).id] for name in JOINT_NAMES]
        bodies = (*TARGETS, "cover_a", "cover_b")
        pose_addresses = [model.jnt_qposadr[model.joint(name + "_joint").id] for name in bodies]
        original.qpos[addresses] = [.15, -.3, .6, -.2, .25, -.012]
        left = model.jnt_qposadr[model.joint("left_jaw_slide_joint").id]
        pinion = model.jnt_qposadr[model.joint("gripper_pinion_joint").id]
        original.qpos[left] = .012
        original.qpos[pinion] = -.012 * 246.399423811
        mujoco.mj_forward(model, original)
        poses = np.concatenate([original.qpos[address:address + 7] for address in pose_addresses])
        state = normalized(original.qpos[addresses], model.actuator_ctrlrange)
        joints = physical_positions(state, model.actuator_ctrlrange)
        restore_pose(model, restored, joints, addresses, poses, pose_addresses, 2.0)
        np.testing.assert_allclose(restored.qpos, original.qpos, atol=1e-6, rtol=0)
        np.testing.assert_allclose(restored.geom_xpos, original.geom_xpos, atol=1e-7, rtol=0)
        np.testing.assert_allclose(restored.cam_xpos, original.cam_xpos, atol=1e-7, rtol=0)
        np.testing.assert_allclose(restored.cam_xmat, original.cam_xmat, atol=1e-7, rtol=0)
        self.assertEqual(restored.time, 2.0)
        # Full snapshots preserve contact-induced passive-joint constraint residuals.
        original.qpos[left] += .0001
        restore_pose(model, restored, joints, addresses, poses, pose_addresses, 3.0, qpos=original.qpos.copy())
        np.testing.assert_array_equal(restored.qpos, original.qpos)

    def test_gate_tolerates_pixel_edges_but_rejects_scene_drift(self):
        mask = np.zeros((30, 30), bool)
        mask[10:20, 10:20] = True
        self.assertEqual(boundary_error(np.roll(mask, 2, axis=1), mask), 0)
        self.assertGreater(boundary_error(np.roll(mask, 5, axis=1), mask), .02)


if __name__ == "__main__":
    unittest.main()
