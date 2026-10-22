"""Physical completion checks for cover drop and target lift."""
import numpy as np


def jaw_contacts(model, data, body_name):
    body = model.body(body_name).id
    jaws = {model.geom(f"link_6_{side}_jaw_collision_0").id for side in ("left", "right")}
    return {jaw for c in data.contact for jaw in jaws
            if ((c.geom1 == jaw and model.geom_bodyid[c.geom2] == body)
                or (c.geom2 == jaw and model.geom_bodyid[c.geom1] == body))}


def cover_deposited(model, data, cover):
    body = model.body(cover).id
    zone = model.geom("cover_drop_zone").id
    if not np.all(np.abs(data.xpos[body, :2] - data.geom_xpos[zone, :2]) < model.geom_size[zone, :2]):
        return False
    table = model.geom("table_top").id
    supported = any((c.geom1 == table and model.geom_bodyid[c.geom2] == body)
                    or (c.geom2 == table and model.geom_bodyid[c.geom1] == body) for c in data.contact)
    dof = model.jnt_dofadr[model.body_jntadr[body]]
    return bool(supported and not jaw_contacts(model, data, cover)
                and np.linalg.norm(data.qvel[dof:dof + 3]) < .05
                and np.linalg.norm(data.qvel[dof + 3:dof + 6]) < .5)


def target_lifted(model, data, target, initial_z):
    body = model.body(target).id
    return bool(data.xpos[body, 2] - initial_z > .04
                and len(jaw_contacts(model, data, target)) == 2)
