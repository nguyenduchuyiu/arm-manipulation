"""Covered-state resets and K5 transitions for the absolute-joint prior."""
import multiprocessing as mp
import os

import mujoco
import numpy as np

from OATFlow.dataset.expert import setup_scene
from OATFlow.environment.control import step_joint_waypoint
from OATFlow.environment.env import COVER_DROP_GOALS_XY, MemoryOcclusionEnv
from OATFlow.environment.success import cover_deposited, jaw_contacts
from OATFlow.task import TARGETS, TASKS, normalized, physical_joint_action, task_id_from_state


GAMMA = .999
LAMBDA = .99
EXECUTE = 5
MAX_FRAMES = 1000


class GraspEnv(MemoryOcclusionEnv):
    def __init__(self):
        super().__init__(max_episode_steps=MAX_FRAMES)
        self.segmentation_renderer.close()
        self.ee = self.model.site("grasp_site").id

    def reset(self, *, seed, task_id, target_id, slot_order=None):
        task = TASKS[task_id]
        if TARGETS[target_id] not in task["objects"]:
            raise ValueError("target must belong to the task pair")
        while True:
            setup_scene(self, task_id, seed)
            first, second = task["objects"]
            order = int(self.data.xpos[self.model.body(first).id, 0] > self.data.xpos[self.model.body(second).id, 0])
            if slot_order is None or order == slot_order:
                break
            seed += 100000
        self.query_target = TARGETS[target_id]
        self.selected_cover = self.assignment[self.query_target]
        if task_id_from_state(self, task["objects"]) != task_id:
            raise ValueError("reset task differs from physical containment")
        self.task_id, self.target_id, self.seed = task_id, target_id, seed
        self.phase = "execute"
        self.elapsed_steps = self.grasp_hold_steps = 0
        self.failure_reason = None
        self.success = self.target_grasped = False
        self.cover_paid = self.cover_lifted = False
        self.first_cover_frame = None
        self.max_lifts = np.zeros(4)
        self.initial_cover_z = self.data.xpos[self.model.body(self.selected_cover).id, 2]
        self.phi = self.potential()
        self.episode_return = 0.
        return self.packet()

    def potential(self):
        ee = self.data.site_xpos[self.ee]
        cover = self.data.xpos[self.model.body(self.selected_cover).id]
        if self.cover_paid:
            target = self.data.xpos[self.model.body(self.query_target).id]
            # Raising a securely gripped target should not undo approach progress.
            distance = np.linalg.norm(ee - (target + (0, 0, .01)))
            return 5. - 10. * distance
        if self.cover_lifted:
            # The goal uses the original physical side, not the cover's moving x.
            goal = COVER_DROP_GOALS_XY[("cover_a", "cover_b")[TASKS[self.task_id]["cover_side"]]]
            return 2. - 10. * np.linalg.norm(cover[:2] - goal)
        return -10. * np.linalg.norm(ee - (cover + (0, 0, .130)))

    def privileged(self):
        bodies = (*TARGETS, *self.cover_qadr)
        contacts = np.array([len(jaw_contacts(self.model, self.data, name)) / 2 for name in bodies])
        return np.concatenate((self.data.qpos, self.data.qvel, normalized(self.data.ctrl, self.model.actuator_ctrlrange),
                               self.data.site_xpos[self.ee], contacts, np.eye(12)[self.task_id],
                               np.eye(4)[self.target_id],
                               [self.cover_paid, self.cover_lifted, self.grasp_hold_steps / 10,
                                self.elapsed_steps / MAX_FRAMES])).astype(np.float32)

    def packet(self):
        return dict(overview=self._overview(), wrist=self.wrist_image(),
                    proprio=normalized(self.data.qpos[self.robot_qpos_addresses], self.model.actuator_ctrlrange),
                    task_id=self.task_id, target_id=self.target_id, critic=self.privileged())

    def step_chunk(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (EXECUTE, 6) or not np.isfinite(actions).all():
            raise ValueError("K5 finite absolute normalized actions required")
        rewards, details = [], []
        terminated = truncated = False
        for action in actions:
            command = physical_joint_action(np.clip(action, [-1] * 5 + [0], 1), self.model.actuator_ctrlrange)
            step_joint_waypoint(self.model, self.data, command)
            self.elapsed_steps += 1
            positions = self.target_positions()
            lifts = np.array([positions[name][2] - self.initial_positions[name][2] for name in TARGETS])
            self.max_lifts = np.maximum(self.max_lifts, lifts)
            cover_now = cover_deposited(self.model, self.data, self.selected_cover)
            reward = -.01
            if cover_now and not self.cover_paid:
                self.cover_paid = True
                self.first_cover_frame = self.elapsed_steps
                reward += 10.
            cover = self.data.xpos[self.model.body(self.selected_cover).id]
            if cover[2] - self.initial_cover_z > .04 and len(jaw_contacts(self.model, self.data, self.selected_cover)) == 2:
                self.cover_lifted = True
            good_lift = lifts[self.target_id] > .04 and len(jaw_contacts(self.model, self.data, self.query_target)) == 2
            self.grasp_hold_steps = self.grasp_hold_steps + 1 if cover_now and good_lift else 0
            self.success = self.grasp_hold_steps >= 10
            wrong = next((name for i, name in enumerate(TARGETS) if i != self.target_id and lifts[i] > .04), None)
            escaped = any(np.any(pos < (.28, -.42, -.04)) or np.any(pos > (.98, .18, .65))
                          for pos in positions.values())
            if wrong:
                self.failure_reason = "wrong_object_lifted"
            elif escaped:
                self.failure_reason = "object_out_of_workspace"
            terminated = self.success or self.failure_reason is not None
            truncated = self.elapsed_steps >= MAX_FRAMES and not terminated
            if self.success:
                reward += 100.
            elif terminated:
                reward -= 100.
            # An absorbing terminal has potential zero. Timeout is bootstrapped.
            next_phi = 0. if terminated else self.potential()
            shaping = GAMMA * next_phi - self.phi
            self.phi = next_phi
            reward += shaping
            rewards.append(reward)
            details.append(dict(reward=reward, shaping=shaping, cover_deposited=cover_now,
                                held=self.grasp_hold_steps, wrong_object=wrong))
            self.episode_return += reward
            if terminated or truncated:
                break
        m = len(rewards)
        result = dict(reward=sum(GAMMA ** i * r for i, r in enumerate(rewards)),
                      frames=m, discount=GAMMA ** m, trace_discount=(GAMMA * LAMBDA) ** m,
                      terminated=terminated, truncated=truncated,
                      episode=dict(seed=self.seed, task_id=self.task_id, target_id=self.target_id,
                                   target=self.query_target, frames=self.elapsed_steps, success=self.success,
                                   cover_removed=self.cover_paid, cover_current=cover_now,
                                   first_cover_removed_frame=self.first_cover_frame,
                                   wrong_object=wrong, reason="success" if self.success else self.failure_reason or
                                   ("timeout" if truncated else None), return_raw=self.episode_return,
                                   max_object_lift_m=dict(zip(TARGETS, self.max_lifts.tolist()))),
                      waypoint_details=details)
        return self.packet(), result

    def close(self):
        self.renderer.close()


def _worker(pipe, core):
    os.sched_setaffinity(0, {core})
    env = GraspEnv()
    try:
        while True:
            command, argument = pipe.recv()
            if command == "close":
                break
            if command == "reset":
                pipe.send(env.reset(**argument))
            elif command == "step":
                pipe.send(env.step_chunk(argument))
            else:
                raise ValueError(command)
    finally:
        env.close()
        pipe.close()


class VectorGrasp:
    """CPU simulations/renderers; one batched CUDA policy lives in the parent."""
    def __init__(self, count=8):
        context = mp.get_context("spawn")
        self.pipes, self.processes = [], []
        for core in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, core))
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)

    def send(self, index, command, argument):
        self.pipes[index].send((command, argument))

    def receive(self, index):
        return self.pipes[index].recv()

    def close(self):
        for pipe, process in zip(self.pipes, self.processes):
            if process.is_alive():
                pipe.send(("close", None))
        for pipe, process in zip(self.pipes, self.processes):
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join()
            pipe.close()


def balanced_queries(rng):
    queries = [(task["task_id"], TARGETS.index(target), order)
               for task in TASKS for target in task["objects"] for order in range(2)]
    return [queries[index] for index in rng.permutation(len(queries))]
