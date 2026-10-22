"""Collect dynamic shuffle contexts and contact-expert continuations for front/wrist RGB-D."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from tqdm import tqdm

from controllers.oracle_pick import GraspFailure, IKFailure
from OATFlow.data_tools.collect_shuffle import (BASE, NAMES, PERMS, SEEN, UNSEEN, Scene,
    layout_spec, setup, shuffle_plan, split, save_segment, write_json)
from OATFlow.data_tools.dynamic_shuffle import capture, context, scene_xml
from OATFlow.dataset.expert import execute
from OATFlow.tracking.mask_video import digest


SOURCE = Path('/workspace/robotics_assets/datasets/shuffle_pick_multiview/scene.xml')
VIEWS = ('front', 'wrist')
WRISTS = [(-.8, 0.), (-.4, np.pi/2), (.4, np.pi/2), (.8, 0.),
          (-.4, -np.pi/2), (.4, -np.pi/2), (0., np.pi/2), (.8, np.pi/2),
          (-.8, np.pi/2), (-.8, -np.pi/2), (-1.2, 0.), (-1.2, np.pi/2)]
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def full_state(scene):
    result = np.empty(mujoco.mj_stateSize(scene.model, STATE))
    mujoco.mj_getState(scene.model, scene.data, result, STATE)
    return result


def save(directory, store, offset=0, initial=None, terminal=None):
    n = save_segment(directory, store, offset=offset, initial=initial)
    with np.load(directory/'sim_state.npz') as z: state = {k:z[k] for k in z.files}
    for key in ('mocap_pos', 'mocap_quat', 'eq_active', 'mocap_target', 'integration_state'):
        state[key] = np.asarray(store[key])
    state['integration_state_spec'] = int(STATE)
    if terminal is not None: state['terminal_integration_state'] = terminal
    np.savez_compressed(directory/'sim_state.npz', **state)
    return n


def collect_one(model, spec, group, permutation, directory):
    rejections = []
    for attempt in range(20):
        scene = Scene(model); setup(scene, spec, permutation)
        plan = shuffle_plan(group, permutation, spec)
        if attempt:
            rng = np.random.default_rng([116000, group, permutation, attempt])
            for path in plan['swaps']:
                path.update(frames=int(rng.integers(40, 111)), arc_m=float(rng.uniform(.095, .125)),
                            time_gamma=float(rng.uniform(.8, 1.3)), route_flip=int(rng.choice([-1, 1])))
        try:
            states, dynamics = context(scene, plan)
            terminal = full_state(scene); n = len(states['phase'])
            initial = (scene.data.qpos.copy(), scene.data.qvel.copy(), scene.data.ctrl.copy())
            queries = []
            for target_id, target in enumerate(NAMES[:4]):
                slot = PERMS[permutation].index(target_id)
                wrist_errors = []
                for wrist in WRISTS:
                    scene.data = mujoco.MjData(model)
                    mujoco.mj_setState(model, scene.data, terminal, STATE); mujoco.mj_forward(model, scene.data)
                    query_store = {}; decision = {'t_occ': n}
                    def record(phase):
                        if phase == 't_obj': decision['t_obj'] = n+len(query_store.get('phase', []))
                        else: capture(scene, query_store, phase, True)
                    try:
                        expert = execute(scene, target, np.random.default_rng([spec['seed'], slot, len(plan['swaps'])%2]),
                                         record, target_wrist=wrist)
                        capture(scene, query_store, 'done', False)
                        queries.append((target_id, query_store, decision, expert, wrist, wrist_errors))
                        break
                    except (GraspFailure, IKFailure) as error:
                        wrist_errors.append(dict(wrist=list(wrist), reason=str(error)))
                else:
                    raise GraspFailure(f'{target}: all grasp orientations failed: {wrist_errors[-1]}')
            break
        except (GraspFailure, IKFailure) as error:
            rejections.append(dict(attempt=attempt, reason=str(error)))
    else:
        raise GraspFailure(f'context failed20 attempts: {rejections[-1]}')
    save(directory, states, terminal=terminal)
    write_json(directory/'context.json', dict(layout_id=group, layout_seed=spec['seed'],
        initial_permutation=permutation, assignment=scene.assignment, frames=n, fps=25,
        views=list(VIEWS), resolution=[320, 240], context_plan=plan,
        dynamics=dynamics, rejected_plans=rejections, frame_semantics='pre-step, next expert starts after last context step'))
    rows = []
    for target_id, store, decision, expert, wrist, errors in queries:
        path = directory/f't{target_id}'; ne = save(path, store, n, initial=initial)
        with np.load(path/'sim_state.npz') as z: payload = {k:z[k] for k in z.files}
        payload['initial_integration_state'] = terminal
        np.savez_compressed(path/'sim_state.npz', **payload)
        dataset_split, evaluation_group = split(group, permutation)
        meta = dict(target_id=target_id, target_object_id=NAMES[target_id], layout_id=group,
            layout_seed=spec['seed'], permutation=permutation, target_slot=PERMS[permutation].index(target_id),
            context_path=f'g{group:03d}/p{permutation:02d}', context_frames=n, expert_frames=ne, frames=n+ne,
            split=dataset_split, evaluation_group=evaluation_group, decision_frames=decision,
            fps=25, resolution=[320, 240], views=list(VIEWS), action_representation='absolute_joint_command_physical_rad_m',
            state_perturbations=[], objective='cover_drop_then_target_lift', success=True,
            target_wrist=list(wrist), rejected_grasp_orientations=errors, **expert)
        write_json(path/'episode.json', meta)
        rows.append(dict(path=f'g{group:03d}/p{permutation:02d}/t{target_id}', **meta))
    return rows


def group_job(job):
    root, group, pilot = job; root = Path(root)
    if (root/f'g{group:03d}/rows.json').exists():
        return json.loads((root/f'g{group:03d}/rows.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(root/'scene.xml'))
    ids = [0, 1] if pilot else (SEEN if 32 <= group < 36 else range(24))
    rejected = []
    started = time.perf_counter()
    for attempt in range(20):
        staging = root/f'.g{group:03d}_partial'
        if staging.exists(): shutil.rmtree(staging)
        seed = 960000+group+1000*attempt
        try:
            spec = layout_spec(seed)
            rows = []
            for permutation in tqdm(ids, desc=f'Dynamic g{group:03d}', mininterval=5, leave=False):
                rows += collect_one(model, spec, group, permutation, staging/f'p{permutation:02d}')
                write_json(root/f'progress_g{group:03d}.json', dict(layout=group, contexts=len(rows)//4,
                    total_contexts=len(ids), layout_attempt=attempt, elapsed_s=time.perf_counter()-started))
        except (GraspFailure, IKFailure) as error:
            rejected.append(dict(seed=seed, reason=str(error)))
            continue
        write_json(staging/'layout.json', dict(**spec, rejected_seeds=rejected))
        write_json(staging/'rows.json', rows)
        staging.rename(root/f'g{group:03d}')
        return rows
    raise RuntimeError(f'layout{group} failed: {rejected[-1]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); out = args.output
    if not args.resume:
        out.mkdir(exist_ok=False, parents=True)
        shutil.copyfile('docs/research/e116_README.md', out/'README.md')
        scene_xml(SOURCE, out/'scene.xml')
        # Native240px vertical view equals the previously previewed320px center crop.
        import xml.etree.ElementTree as ET
        tree = ET.parse(out/'scene.xml')
        for camera in tree.getroot().iter('camera'):
            fovy = np.deg2rad(float(camera.get('fovy')))
            camera.set('fovy', str(np.rad2deg(2*np.arctan(.75*np.tan(fovy/2)))))
        tree.write(out/'scene.xml')
        code = [Path(__file__), Path(__file__).with_name('dynamic_shuffle.py'),
                Path('OATFlow/dataset/expert.py'), Path('controllers/oracle_pick.py'), out/'scene.xml']
        config = dict(status='collecting_physics', pilot=args.pilot, views=list(VIEWS), resolution=[320, 240],
            fps=25, physics_hz=500, depth_unit_m=.002, depth_max_m=6.5535, depth_views=list(VIEWS),
            base_origin_world_m=BASE.tolist(), expected_contexts=4 if args.pilot else 928,
            expected_episodes=16 if args.pilot else 3712, train_permutations=SEEN, unseen_permutations=UNSEEN,
            train_layouts=list(range(32)), val_layouts=list(range(32, 36)), test_layouts=list(range(36, 40)),
            physics='externally mocap-weld-driven covers, free cubes; no object qpos writes after setup',
            expert='original contact expert with recorded successful wrist orientation; shared context across4queries',
            final_permutation='not discretized; free physical motion changes within-cover slots. Split is by INITIAL permutation.',
            code_sha256={str(p):digest(p) for p in code})
        write_json(out/'config.json', config)
    config = json.loads((out/'config.json').read_text())
    assert config['pilot'] == args.pilot
    if args.resume:
        for path, expected in config['code_sha256'].items(): assert digest(Path(path)) == expected, path
    groups = [0, 36] if args.pilot else list(range(40))
    start = time.perf_counter(); rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(groups)), mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(group_job, (str(out), g, args.pilot)) for g in groups]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Dynamic physics groups', mininterval=5):
            rows += f.result()
            write_json(out/'progress.json', dict(stage='physics', episodes=len(rows), elapsed_s=time.perf_counter()-start))
    rows.sort(key=lambda r:(r['layout_id'], r['permutation'], r['target_id']))
    assert len(rows) == config['expected_episodes']
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config.update(status='physics_complete', physics_elapsed_s=time.perf_counter()-start)
    write_json(out/'config.json', config)


if __name__ == '__main__': main()
