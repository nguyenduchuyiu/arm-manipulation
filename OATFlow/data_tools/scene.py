"""Exact retained user arm/table scene and generation checks."""
from pathlib import Path
import mujoco
import numpy as np

REFERENCE = Path('/workspace/robotics_assets/robot/nexarm/physics_reference.xml')


def robot_table_xml():
    return REFERENCE.read_text()


def assert_user_arm_configuration(model):
    reference=mujoco.MjModel.from_xml_path(str(REFERENCE))
    for name in ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
                 'jnt_pos','jnt_axis','jnt_range','dof_damping','dof_frictionloss','dof_armature',
                 'actuator_ctrlrange','actuator_forcerange','actuator_gainprm','actuator_biasprm','eq_data'):
        np.testing.assert_allclose(getattr(model,name),getattr(reference,name),rtol=0,atol=1e-12)
    for name in ('cone','solver','impratio','integrator','timestep'):
        assert getattr(model.opt,name)==getattr(reference.opt,name),name
    return dict(status='passed',configuration_source=str(REFERENCE),shoulder_force_range_nm=model.actuator_forcerange[1].tolist(),
                cone=int(model.opt.cone),solver=int(model.opt.solver),impratio=float(model.opt.impratio),timestep_s=float(model.opt.timestep))

ROBOT_BODIES = {"world", "base_link", *(f"link_{i}" for i in range(1, 6)),
                "link_6_gripper_base", "link_6_pinion_gear", "link_6_left_jaw",
                "link_6_right_jaw", "cam_mount"}
ROBOT_COLLIDERS = {"base_link_collision_0", *(f"link_{i}_collision_0" for i in range(1, 6)),
                   "link_6_gripper_base_collision_0", "link_6_left_jaw_collision_0",
                   "link_6_right_jaw_collision_0"}
TABLE_COLLIDERS = {"table_top", *(f"table_leg_{i}" for i in range(4))}


def assert_robot_table_only(model):
    bodies = {model.body(i).name for i in range(model.nbody)}
    if bodies != ROBOT_BODIES:
        raise ValueError(f"Unexpected scene bodies: {sorted(bodies ^ ROBOT_BODIES)}")
    allowed_world = TABLE_COLLIDERS
    world_geoms = {model.geom(i).name for i in range(model.ngeom) if model.geom_bodyid[i] == 0}
    if not world_geoms <= allowed_world:
        raise ValueError(f"Unexpected world geometry: {sorted(world_geoms - allowed_world)}")
    active = {model.geom(i).name for i in range(model.ngeom)
              if model.geom_contype[i] or model.geom_conaffinity[i]}
    if active != ROBOT_COLLIDERS | TABLE_COLLIDERS:
        raise ValueError(f"Unexpected collision geometry: {sorted(active ^ (ROBOT_COLLIDERS | TABLE_COLLIDERS))}")
    return {"status": "passed", "body_names": sorted(bodies),
            "world_geometries": sorted(world_geoms), "active_colliders": sorted(active),
            "nq": model.nq, "nv": model.nv, "nu": model.nu}
