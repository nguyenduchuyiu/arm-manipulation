"""Closed-loop Cartesian control using FK search (no IK solver).

Instead of lerobot inverse_kinematics, the joint command for each desired
Cartesian target is found by random-restart + hill-climb over the actuator
range, scoring each candidate with lerobot RobotKinematics.forward_kinematics.
Pure kinematic FK: no MuJoCo physics clone, no gravity settling. Sag is
compensated by the closed loop, which re-reads the true joints and re-searches
each step (warm-started from the previous command).
"""
from __future__ import annotations

import mujoco
import numpy as np

from controllers.nexarm_ik import NexArmIK
from envs.nexarm_env import NexArmEnv


FK_TOL_M = 1e-3
RANDOM_RESTARTS = 200
HILL_STEP0 = 0.1


def fk_search(
    ik: NexArmIK,
    target_xyz: np.ndarray,
    q0: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Return arm q whose lerobot-FK gripper-base position reaches target_xyz."""
    def err(q: np.ndarray) -> float:
        return float(np.linalg.norm(ik.forward_kinematics(q)[:3, 3] - target_xyz))

    q = np.clip(q0, low, high)
    best_err = err(q)
    for _ in range(RANDOM_RESTARTS):
        sample = rng.uniform(low, high)
        e = err(sample)
        if e < best_err:
            best_err, q = e, sample.copy()

    step = HILL_STEP0
    while best_err > FK_TOL_M and step > 1e-4:
        improved = False
        for j in range(5):
            for d in (-step, step):
                cand = q.copy()
                cand[j] = float(np.clip(q[j] + d, low[j], high[j]))
                if cand[j] == q[j]:
                    continue
                e = err(cand)
                if e < best_err:
                    best_err, q, improved = e, cand, True
        if not improved:
            step /= 2.0
    return q, best_err


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    env = NexArmEnv(
        scene_path="assets/robot/scene.xml",
        randomize_object=False,
        frame_skip=10,
    )

    ik = NexArmIK(
        urdf_path="assets/robot/robot.urdf",
        target_frame_name="link_6_gripper_base",
    )

    observation, _ = env.reset(seed=0)

    state = observation["observation.state"].astype(np.float64)
    current_arm_qpos = state[:5]

    # Kiểm tra URDF FK có khớp MuJoCo hay không.
    fk_pose = ik.forward_kinematics(current_arm_qpos)
    fk_xyz = fk_pose[:3, 3]

    body_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "link_6_gripper_base",
    )
    mujoco_xyz = env.data.xpos[body_id].copy()

    frame_error = np.linalg.norm(fk_xyz - mujoco_xyz)
    print("LeRobot FK XYZ:", fk_xyz)
    print("MuJoCo body XYZ:", mujoco_xyz)
    print("Frame error:", frame_error)

    if frame_error > 5e-3:
        raise RuntimeError(
            "URDF and MuJoCo frames differ by more than 5 mm. "
            "Do not run FK search until joint/frame conventions are fixed."
        )

    # Yêu cầu gripper-base đi lên 3 cm.
    target_xyz = fk_xyz + np.array([0.0, 0.0, 0.03])
    print("Target XYZ:", target_xyz)

    low = env.action_space.low[:5].astype(np.float64)
    high = env.action_space.high[:5].astype(np.float64)
    rng = np.random.default_rng(0)

    # Closed-loop Cartesian control: mỗi step đọc joint thật, FK-search lại
    # với desired_xyz nội suy, rồi gửi joint target mới (warm-start từ lệnh trước).
    start_xyz = fk_xyz.copy()
    gripper_target = float(env.data.ctrl[5])
    prev_q = current_arm_qpos.copy()
    last_cmd_q = None
    last_residual = None

    for step in range(150):
        alpha = smoothstep((step + 1) / 150)
        desired_xyz = start_xyz + alpha * (target_xyz - start_xyz)

        current_arm_qpos = observation["observation.state"][:5].astype(np.float64)

        arm_command, residual = fk_search(
            ik=ik,
            target_xyz=desired_xyz,
            q0=prev_q,
            low=low,
            high=high,
            rng=rng,
        )
        prev_q = arm_command.copy()
        last_cmd_q = arm_command
        last_residual = residual

        action = np.concatenate([arm_command, [gripper_target]])
        action = np.clip(action, env.action_space.low, env.action_space.high)
        observation, _, terminated, truncated, _ = env.step(action.astype(np.float32))

        if terminated or truncated:
            break

    final_arm_qpos = observation["observation.state"][:5].astype(np.float64)
    final_xyz = ik.forward_kinematics(final_arm_qpos)[:3, 3]
    cmd_xyz = ik.forward_kinematics(last_cmd_q)[:3, 3]

    print("FK-search command q:", last_cmd_q)
    print("FK-search residual (FK-only error):", last_residual)
    print("Final arm qpos:     ", final_arm_qpos)
    print("Joint tracking error (cmd - actual):", last_cmd_q - final_arm_qpos)
    print("Target XYZ:", target_xyz)
    print("Commanded XYZ (FK of cmd):", cmd_xyz)
    print("Final XYZ (FK of actual):", final_xyz)
    print(
        "Cartesian error (target - actual FK):",
        float(np.linalg.norm(target_xyz - final_xyz)),
    )

    env.close()


if __name__ == "__main__":
    main()