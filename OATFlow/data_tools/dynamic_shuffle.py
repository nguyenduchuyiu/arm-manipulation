"""Physics-driven cover shuffle: free cubes, externally driven covers, no object teleport."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from controllers.oracle_pick import GraspFailure
from OATFlow.data_tools.collect_shuffle import NAMES, snapshot
from OATFlow.tracking.pointcloud_front import POSITION, LOOK_AT, FOVY


def capture(scene, store, phase, valid, target=None):
    snapshot(scene, store, phase, valid)
    data = scene.data
    for key in ('mocap_pos', 'mocap_quat', 'eq_active'):
        store.setdefault(key, []).append(getattr(data, key).copy())
    store.setdefault('mocap_target', []).append(data.mocap_pos.copy() if target is None else target.copy())
    state = np.empty(mujoco.mj_stateSize(scene.model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(scene.model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    store.setdefault('integration_state', []).append(state)


def scene_xml(source, destination):
    tree = ET.parse(source); root = tree.getroot(); world = root.find('worldbody')
    for camera in list(world.findall('camera')):
        world.remove(camera)
    backward = POSITION-LOOK_AT; backward /= np.linalg.norm(backward)
    right = np.cross([0., 0., 1.], backward); right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    ET.SubElement(world, 'camera', name='cam_front', pos=' '.join(map(str, POSITION)),
                  xyaxes=' '.join(map(str, np.r_[right, up])), fovy=str(FOVY))
    equality = root.find('equality')
    for k, name in enumerate(NAMES[4:]):
        ET.SubElement(world, 'body', name=name+'_driver', mocap='true',
                      pos=f'{-2 if k == 0 else 2} 0 .15')
        ET.SubElement(equality, 'weld', name=name+'_drive', body1=name, body2=name+'_driver',
                      relpose='0 0 0 1 0 0 0', active='false', solref='.004 1',
                      solimp='.99 .999 .0001', torquescale='.1')
    tree.write(destination)


def containment(scene):
    data, model = scene.data, scene.model
    for target, owner in scene.assignment.items():
        body, cover = model.body(target).id, model.body(owner).id
        local = data.xmat[cover].reshape(3, 3).T @ (data.xpos[body]-data.xpos[cover])
        if abs(local[0]) > .074 or abs(local[1]) > .059 or not -.005 < local[2] < .055:
            raise GraspFailure(f'{target}: escaped {owner}: {local}')


def context(scene, plan):
    model, data = scene.model, scene.data
    centers = np.asarray(plan['cover_centers'])
    drivers = [model.body(n+'_driver').mocapid[0] for n in NAMES[4:]]
    welds = [model.equality(n+'_drive').id for n in NAMES[4:]]
    for k, name in enumerate(NAMES[4:]):
        a = scene.cover_qadr[name]
        data.mocap_pos[drivers[k]] = data.qpos[a:a+3]
        data.mocap_quat[drivers[k]] = data.qpos[a+3:a+7]
        data.eq_active[welds[k]] = 1
    store = {}; diagnostics = []; contacts = np.zeros(4, int)
    object_bodies = [model.body(n).id for n in NAMES[:4]]
    cover_bodies = [model.body(n).id for n in NAMES[4:]]

    def frame(phase, begin, end):
        target = data.mocap_pos.copy(); target[drivers] = end
        capture(scene, store, phase, False, target)
        for step in range(20):
            t = (step+1)/20
            data.mocap_pos[drivers] = begin*(1-t)+end*t
            mujoco.mj_step(model, data)
            for contact in data.contact:
                a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
                if a in object_bodies and b in cover_bodies: contacts[object_bodies.index(a)] += 1
                if b in object_bodies and a in cover_bodies: contacts[object_bodies.index(b)] += 1
        if phase in ('shuffle', 'shuffle_pause', 'hold'):
            containment(scene)
        diagnostics.append([data.xpos[b].copy() for b in object_bodies+cover_bodies])

    positions = data.mocap_pos[drivers].copy()
    for _ in range(plan['reveal_frames']): frame('reveal', positions, positions)
    for i in range(plan['occlude_frames']):
        t = (i+1)/plan['occlude_frames']; travel = min(t/.6, 1); lower = max((t-.6)/.4, 0)
        travel = travel**2*(3-2*travel); lower = lower**2*(3-2*lower)
        target = np.array([[(-2 if j == 0 else 2)+travel*(centers[j, 0]-(-2 if j == 0 else 2)),
                            centers[j, 1], .22+lower*(.0005-.22)] for j in range(2)])
        frame('occlude', positions, target); positions = target
    relative_initial = None
    for k, path in enumerate(plan['swaps']):
        starts = positions.copy()
        if relative_initial is None:
            relative_initial = np.array([data.xpos[model.body(n).id]-data.xpos[model.body(scene.assignment[n]).id] for n in NAMES[:4]])
        for i in range(path['frames']):
            u = ((i+1)/path['frames'])**path['time_gamma']; crossing = np.clip((u-.2)/.6, 0., 1.)
            travel = crossing**2*(3-2*crossing); lane = min(u/.2, (1-u)/.2, 1.)
            lane = lane**2*(3-2*lane); target = []
            for j in range(2):
                end = centers[(1-j) if k%2 == 0 else j]
                xy = starts[j, :2]+travel*(end-starts[j, :2])
                xy[1] += np.sign(starts[j, 0]-starts[:, 0].mean())*path['route_flip']*path['arc_m']*lane
                target.append(np.r_[xy, .0005])
            target = np.asarray(target)
            frame('shuffle', positions, target); positions = target
        for _ in range(path['pause_frames']): frame('shuffle_pause', positions, positions)
    # Release external drive; both covers and cubes settle freely before expert starts.
    data.eq_active[welds] = 0
    for _ in range(max(25, plan['hold_frames'])): frame('hold', positions, positions)
    selected = np.isin(store['phase'], ['shuffle', 'shuffle_pause', 'hold'])
    trajectory = np.asarray(diagnostics)[selected]
    rel = np.stack([trajectory[:, i]-trajectory[:, 4+NAMES[4:].index(scene.assignment[n])] for i, n in enumerate(NAMES[:4])], axis=1)
    stats = dict(cover_object_contact_substeps=contacts.tolist(),
                 max_relative_translation_m=np.linalg.norm(rel-relative_initial, axis=-1).max(0).tolist(),
                 final_relative_change_m=np.linalg.norm(rel[-1]-relative_initial, axis=-1).tolist())
    if not all(contacts > 0): raise GraspFailure('shuffle did not physically contact all cubes')
    return store, stats
