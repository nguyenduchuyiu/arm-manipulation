"""Counterfactual four-object scenes and contact-based pick-only expert."""

from sam3_pick.simulation.grasp import GraspFailure, pick
from sam3_pick.simulation.contacts import jaw_contacts
from sam3_pick.data.tasks import TARGETS


def execute(env, target, record):
    initial = env.target_positions()
    held, longest = (0, 0)

    def callback(stage):
        nonlocal held, longest
        for name in TARGETS:
            if name != target and env.target_positions()[name][2] - initial[name][2] > 0.04:
                raise GraspFailure("expert lifted a distractor")
        gain = env.target_positions()[target][2] - initial[target][2]
        held = (
            held + 1 if gain > 0.04 and len(jaw_contacts(env.model, env.data, target)) == 2 else 0
        )
        longest = max(longest, held)
        record(target + "_" + stage)

    point = env.data.xpos[env.model.body(target).id].copy() + (0, 0, 0.01)
    pick(env.model, env.data, target, point, on_step=callback)
    if longest < 10 or env.target_positions()[target][2] - initial[target][2] <= 0.04:
        raise GraspFailure("missing stable bilateral target lift")
    return dict(max_bilateral_hold_frames=longest)
