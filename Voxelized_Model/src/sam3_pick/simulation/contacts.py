"""Physical completion checks for cover drop and target lift."""


def jaw_contacts(model, data, body_name):
    body = model.body(body_name).id
    jaws = {model.geom(f"link_6_{side}_jaw_collision_0").id for side in ("left", "right")}
    return {
        jaw
        for c in data.contact
        for jaw in jaws
        if c.geom1 == jaw
        and model.geom_bodyid[c.geom2] == body
        or (c.geom2 == jaw and model.geom_bodyid[c.geom1] == body)
    }
