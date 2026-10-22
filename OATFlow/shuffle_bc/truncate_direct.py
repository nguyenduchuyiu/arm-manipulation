"""Re-score a committed rollout at a shorter horizon using its saved prefix."""
from itertools import groupby

import mujoco
import numpy as np

from OATFlow.data_tools.collect_shuffle import NAMES
from OATFlow.environment.success import cover_deposited


def truncate(path, result, row, budget, model):
    n = min(result['frames'], budget)
    initial = {'initial_qpos','initial_qvel','initial_ctrl','initial_mask'}
    states = {'qpos','qvel','ctrl'}
    steps = {'commands','objects','contacts'}
    plans = {'plans','anchors','anchor_rotations','mask_packed','history_times','replan_frames'}
    with np.load(path) as z:
        assert set(z.files) == initial|states|steps|plans
        selected = z['replan_frames'] < n
        arrays = {k:z[k].copy() for k in initial}
        arrays.update({k:z[k][:n+1].copy() for k in states})
        arrays.update({k:z[k][:n].copy() for k in steps})
        arrays.update({k:z[k][selected].copy() for k in plans})
    assert len(arrays['commands'])==n and len(arrays['qpos'])==n+1
    data = mujoco.MjData(model)
    for k in states: getattr(data,k)[:] = arrays['initial_'+k]
    mujoco.mj_forward(model,data)
    initial_objects=np.stack([data.xpos[model.body(name).id].copy() for name in NAMES])
    selected_cover=row['selected_cover'];other='cover_b' if selected_cover=='cover_a' else 'cover_a'
    other_start=data.xpos[model.body(other).id,:2].copy()
    for k in states:getattr(data,k)[:] = arrays[k][-1]
    mujoco.mj_forward(model,data)
    target=row['target_id'];objects=arrays['objects'];contacts=arrays['contacts']
    lifted=(objects[:,target,2]-initial_objects[target,2]>.04)&(contacts[:,target]==2)
    longest=max((sum(1 for _ in group) for value,group in groupby(lifted) if value),default=0)
    # The original loop stops on the first qualifying success or wrong lift.
    # A recorded stop beyond this prefix therefore becomes a timeout here.
    success=bool(result['success'] and result['frames']<=budget)
    reason=result['reason'] if result['frames']<=budget else 'timeout'
    metrics=dict(cover_deposited=cover_deposited(model,data,selected_cover),
        other_cover_shift_m=float(np.linalg.norm(data.xpos[model.body(other).id,:2]-other_start)),
        lifted_bilateral_streak=longest,
        max_target_lift_m=float((objects[:,target,2]-initial_objects[target,2]).max()),
        max_cover_lift_m=(objects[:,4:,2]-initial_objects[4:,2]).max(0).tolist())
    if result['frames']<=budget:
        for key in metrics:np.testing.assert_allclose(metrics[key],result[key],rtol=0,atol=1e-12,err_msg=key)
    updated=dict(result,**metrics)
    updated.update(frames=n,success=success,reason=reason,video_frames_verified=0,frame_budget=budget,
        source_frames=result['frames'],source_reason=result['reason'])
    return updated,arrays
