"""25Hz absolute joint waypoints interpolated into 50Hz simulator commands."""
import mujoco
import numpy as np
import time


def step_joint_waypoint(model, data, waypoint, *, start_time=None, wall_times=None):
    """Hold q_k to20ms, send midpoint, then send q_(k+1) at40ms.

    The endpoint remains in data.ctrl for the next call. Only the midpoint
    and new endpoint are sent, so adjacent calls do not duplicate boundaries.
    With start_time (perf_counter), pace commands on the wall clock. Late
    commands shift the schedule forward instead of bursting to catch up.
    """
    waypoint = np.asarray(waypoint, dtype=np.float64)
    if waypoint.shape != (model.nu,) or not np.isfinite(waypoint).all():
        raise ValueError("expected a finite absolute actuator waypoint")
    bounded = np.clip(waypoint, *model.actuator_ctrlrange.T)
    if not np.allclose(waypoint, bounded, rtol=0, atol=1e-7):
        raise ValueError("waypoint outside actuator limits")
    waypoint = bounded
    substeps = round(.02 / model.opt.timestep)
    if substeps < 1 or not np.isclose(substeps * model.opt.timestep, .02):
        raise ValueError("50Hz servo period must be a multiple of the physics timestep")
    previous = data.ctrl.copy()
    midpoint = (previous + waypoint) / 2
    mujoco.mj_step(model, data, substeps)
    midpoint_time = float(data.time)
    if start_time is not None:
        time.sleep(max(0., start_time + .02 - time.perf_counter()))
    data.ctrl[:] = midpoint
    midpoint_wall = time.perf_counter()
    if wall_times is not None:
        wall_times.append(midpoint_wall)
    mujoco.mj_step(model, data, substeps)
    endpoint_time = float(data.time)
    if start_time is not None:
        deadline = max(start_time + .04, midpoint_wall + .02)
        time.sleep(max(0., deadline - time.perf_counter()))
    data.ctrl[:] = waypoint
    if wall_times is not None:
        wall_times.append(time.perf_counter())
    return np.array((midpoint_time, endpoint_time)), np.stack((midpoint, waypoint))
