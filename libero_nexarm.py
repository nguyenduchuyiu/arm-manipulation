from pathlib import Path

import numpy as np
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm


ASSETS = Path(__file__).resolve().parent / "assets" / "robot"


class NexArmGripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(str(ASSETS / "nexarm_gripper.xml"), idn=idn)

    def format_action(self, action):
        assert len(action) == 1
        return np.clip(action, -1.0, 1.0)

    @property
    def init_qpos(self):
        return np.zeros(2)

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["left_fingerpad"],
            "right_finger": ["right_fingerpad"],
            "left_fingerpad": ["left_fingerpad"],
            "right_fingerpad": ["right_fingerpad"],
        }


class MountedNexArm(ManipulatorModel):
    def __init__(self, idn=0):
        super().__init__(str(ASSETS / "nexarm_arm.xml"), idn=idn)

    @property
    def _eef_name(self):
        return "link_5"

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "NexArmGripper"

    @property
    def default_controller_config(self):
        return "joint_position"

    @property
    def init_qpos(self):
        return np.zeros(5)

    @property
    def base_xpos_offset(self):
        tabletop = np.array([-0.73937, 0.18603, 0.90])
        return {
            "bins": tabletop,
            "empty": np.array([-0.73937, 0.18603, 0.0]),
            "table": lambda _length: tabletop,
            "kitchen_table": lambda _length: tabletop,
            "coffee_table": lambda _length: tabletop + np.array([0.0, 0.0, -0.49]),
            "living_room_table": lambda _length: tabletop
            + np.array([0.0, 0.0, -0.48]),
            "study_table": lambda _length: tabletop,
        }

    @property
    def top_offset(self):
        return np.array([0.0, 0.0, 0.32])

    @property
    def _horizontal_radius(self):
        return 0.35

    @property
    def arm_type(self):
        return "single"


GRIPPER_MAPPING["NexArmGripper"] = NexArmGripper
ROBOT_CLASS_MAPPING["MountedNexArm"] = SingleArm
