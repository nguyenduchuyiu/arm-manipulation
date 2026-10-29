"""Joint names used by the simulated arm controller."""

MUJOCO_JOINTS = {
    "shoulder_pan": "joint_1_base_to_link_1",
    "shoulder_lift": "joint_2_link_1_to_link_2",
    "elbow_flex": "joint_3_link_2_to_link_3",
    "wrist_flex": "joint_4_link_3_to_link_4",
    "wrist_roll": "joint_5_link_4_to_link_5",
    "gripper": "right_jaw_slide_joint",
}
