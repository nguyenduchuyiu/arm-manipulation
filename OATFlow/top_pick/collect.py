"""Home-start four visible cubes; disjoint permutation and spatial holdouts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil

import mujoco
import numpy as np
from tqdm import tqdm

from controllers.oracle_pick import GraspFailure, IKFailure
from OATFlow.data_tools.collect_shuffle import (Scene, NAMES, PERMS, SEEN, UNSEEN,
                                               snapshot, save_segment, split, write_json)
from OATFlow.pick.scene import scene_spec, execute

SOURCE = Path('/workspace/robotics_assets/datasets/shuffle_pick_multiview')
ROOT = Path('/workspace/robotics_assets/experiments/e102_top')
DATA = ROOT/'data'
HOME = Path('/workspace/robotics_assets/datasets/reach_voxel_grid/trajectories/v002043.npz')


def setup(scene, spec, permutation):
    model,data=scene.model,scene.data
    mujoco.mj_resetData(model,data)
    with np.load(HOME) as home:
        data.qpos[:8]=home['state_full_qpos'][0]
        data.qvel[:8]=home['state_full_qvel'][0]
        data.ctrl[:]=home['initial_ctrl']
    for slot,color in enumerate(PERMS[permutation]):
        adr=scene.target_qadr[NAMES[color]];yaw=spec['yaw'][slot]
        data.qpos[adr:adr+7]=(*spec['positions'][slot],.018,np.cos(yaw/2),0,0,np.sin(yaw/2))
    for k,adr in enumerate(scene.cover_qadr.values()):
        data.qpos[adr:adr+7]=(-2 if k==0 else 2,0,.15,1,0,0,0)
    mujoco.mj_forward(model,data);mujoco.mj_step(model,data,250)
    data.time=0.;mujoco.mj_forward(model,data)
    for name in NAMES[:4]:
        assert abs(data.xpos[model.body(name).id,2]-.016)<.003
    return data.qpos.copy(),data.qvel.copy(),data.ctrl.copy()


def group(job, sampler=scene_spec, seed_base=1020000):
    root,g,pilot=job;root=Path(root)
    scene=Scene(mujoco.MjModel.from_xml_path(str(root/'scene.xml')))
    permutations=([0,1] if pilot else (SEEN if 32<=g<36 else range(24)))
    rejected=[]
    for attempt in range(100):
        seed=seed_base+g+attempt*1000
        staging=root/f'.g{g:03d}_partial';staging.mkdir()
        try:
            spec=sampler(seed)
            # Geometry/yaw depend only on layout, never color or query.
            spec.pop('initial_arm')
            result=[]
            for p in tqdm(permutations,desc=f'Expert g{g:03d}',mininterval=10,leave=False):
                for t,target in enumerate(NAMES[:4]):
                    initial=setup(scene,spec,p)
                    store={}
                    expert=execute(scene,target,lambda phase:snapshot(scene,store,phase,True))
                    snapshot(scene,store,'done',False)
                    path=staging/f'p{p:02d}'/f't{t}'
                    frames=save_segment(path,store,initial=initial)
                    dataset_split,evaluation_group=split(g,p)
                    row=dict(path=f'g{g:03d}/p{p:02d}/t{t}',layout_id=g,layout_seed=seed,
                             permutation=p,final_permutation=p,target_id=t,target_object_id=target,
                             target_slot=PERMS[p].index(t),slot_object_ids=PERMS[p],
                             split=dataset_split,evaluation_group=evaluation_group,
                             frames=frames,fps=25,resolution=[320,320],views=['top'],
                             objective='target_lift',success=True,initial_robot='settled common home',
                             context_frames=0,decision_frames=dict(t_obj=0),
                             action_representation='absolute_joint_command_physical_rad_m',**expert)
                    write_json(path/'episode.json',row)
                    result.append(row)
            write_json(staging/'layout.json',dict(**spec,rejected=rejected))
            staging.rename(root/f'g{g:03d}')
            return result,spec,rejected
        except (GraspFailure,IKFailure,ValueError) as error:
            rejected.append(dict(seed=seed,error=str(error)))
            shutil.rmtree(staging)
    raise RuntimeError(f'No feasible complete layout g{g}: {rejected}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DATA)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=False);shutil.copyfile(SOURCE/'scene.xml',out/'scene.xml')
    config=dict(status='collecting',seed=1020000,layouts=40,train_layouts=list(range(32)),
                val_layouts=list(range(32,36)),test_layouts=list(range(36,40)),
                train_permutations=SEEN,unseen_permutations=UNSEEN,permutation_order=PERMS,
                cameras=['top'],resolution=[320,320],fps=25,home=str(HOME),
                spatial_range='slot nominal XY + uniform +/-18mm X,+/-45mm Y and +/-6mm XY offsets',
                geometry_fixed_across_permutations=True,scene_source=str(SOURCE/'scene.xml'),
                all_four_objects_visible=True,covers='parked outside workspace',
                sample_hz=25,expert_control='25Hz command zero-order hold,500Hz physics',
                pilot=args.pilot)
    write_json(out/'config.json',config)
    readme=('# e102_top data\n\nHome-start pick-only data, four visible cubes, top RGB320px/25Hz. '
            'Random independent layout/yaw, geometry fixed across identity permutations. '
            '32 train/4 val/4 test layouts,16 seen/8 unseen permutations; four paired queries. '
            'Reject whole layout on any expert failure. Physical contact expert lifts correct target >4cm '
            'with two jaws for ten frames. No cover-removal, shuffle or placement. '
            'Training/test not run. Actual collection: config.json,manifest.jsonl,audit.json. Status: collecting.\n')
    (out/'README.md').write_text(readme)
    groups=[0,36] if args.pilot else range(40)
    rows=[];rejected=[]
    with ProcessPoolExecutor(8,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(group,(str(out),g,args.pilot)) for g in groups]
        for future in tqdm(as_completed(jobs),total=len(jobs),desc='Home pick layouts',mininterval=5):
            current,spec,failures=future.result();rows+=current;rejected+=failures
            print(json.dumps(dict(layout=current[0]['layout_id'],episodes=len(rows),rejected=len(rejected))),flush=True)
    rows.sort(key=lambda r:r['path'])
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    train=[r for r in rows if r['evaluation_group']=='train']
    train_perms={r['permutation'] for r in train}
    assert train_perms.isdisjoint({r['permutation'] for r in rows if r['evaluation_group'] in ('composition','combined')})
    assert {r['layout_id'] for r in train}.isdisjoint({r['layout_id'] for r in rows if r['evaluation_group'] in ('spatial','combined','val')})
    if not args.pilot:
        assert len(rows)==3712
        assert Counter(r['evaluation_group'] for r in rows)==dict(train=2048,val=256,composition=1024,spatial=256,combined=128)
    audit=dict(status='complete',episodes=len(rows),by_group=Counter(r['evaluation_group'] for r in rows),
               rejected=rejected,permutation_overlap=0,spatial_layout_overlap=0,all_experts_succeeded=True)
    write_json(out/'audit.json',audit);config['status']='complete';write_json(out/'config.json',config)
    (out/'README.md').write_text(readme.replace('Status: collecting','Status: physics complete; RGB pending'))


if __name__=='__main__':main()
