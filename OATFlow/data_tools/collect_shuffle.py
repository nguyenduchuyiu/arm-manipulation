"""Generate e36-style context + expert with balanced unseen permutations/layouts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations
import json
import hashlib
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

import mujoco
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from OATFlow.dataset.expert import execute
from controllers.oracle_pick import GraspFailure, IKFailure
from OATFlow.simulation.pose import ARM_JOINTS, rotation_6d

NAMES = ('RedCube','BlueCube','GreenCube','YellowCube','cover_a','cover_b')
BASE = np.array([.539369022999,.0639727628473,0.])
PERMS = tuple(permutations(range(4)))
SOURCE_SCENE = Path('/workspace/robotics_assets/datasets/shuffle_pick_multiview/scene.xml')
OUTPUT = Path('/workspace/robotics_assets/datasets/shuffle_pick_multiview')
PILOT = None


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def permutation_split():
    remaining=set(PERMS)
    orbits=[]
    while remaining:
        p=min(remaining)
        orbit={p[k:]+p[:k] for k in range(4)}
        orbits.append(orbit)
        remaining-=orbit
    held=set.union(orbits[1],orbits[4])
    unseen=[i for i,p in enumerate(PERMS) if p in held]
    seen=[i for i in range(24) if i not in unseen]
    for ids in (seen,unseen):
        counts=Counter((color,slot) for i in ids for slot,color in enumerate(PERMS[i]))
        assert len(counts)==16 and set(counts.values())=={len(ids)//4}
        assert all(PERMS.index(PERMS[i][2:]+PERMS[i][:2]) in ids for i in ids)
    return seen,unseen


SEEN,UNSEEN=permutation_split()


def split(group,permutation):
    if group<32:
        return ('train','train') if permutation in SEEN else ('test','composition')
    if group<36:
        assert permutation in SEEN
        return 'val','val'
    return 'test','spatial' if permutation in SEEN else 'combined'


class Scene:
    def __init__(self,model):
        self.model=model
        self.data=mujoco.MjData(model)
        self.target_qadr={n:int(model.joint(n+'_joint').qposadr[0]) for n in NAMES[:4]}
        self.cover_qadr={n:int(model.joint(n+'_joint').qposadr[0]) for n in NAMES[4:]}
        self.qadr=np.array([model.jnt_qposadr[model.joint(n).id] for n in (*ARM_JOINTS,'right_jaw_slide_joint')])
        self.dofs=np.array([model.jnt_dofadr[model.joint(n).id] for n in (*ARM_JOINTS,'right_jaw_slide_joint')])
        self.site=model.site('grasp_site').id

    def target_positions(self):
        return {n:self.data.xpos[self.model.body(n).id].copy() for n in NAMES[:4]}

    def query(self,target):
        self.query_target=target

    def choose_cover(self,cover):
        self.selected_cover=cover
        assert cover==self.assignment[self.query_target]


def layout_spec(seed):
    rng=np.random.default_rng(seed)
    centers=np.array([[.45,-.24],[.65,-.24]])+rng.uniform(-.025,.025,(2,2))
    if centers[1,0]-centers[0,0]<.17:
        raise GraspFailure('covers overlap at layout endpoints')
    offsets=np.array([[-.04,0],[.04,0],[-.04,0],[.04,0]])
    offsets+=rng.uniform([-.006,-.014],[.006,.014],(4,2))
    return dict(seed=int(seed),cover_centers=centers.tolist(),slot_offsets=offsets.tolist(),
                slot_yaws=rng.uniform(-np.pi,np.pi,4).tolist())


def shuffle_plan(group,permutation,spec):
    rng=np.random.default_rng(np.random.SeedSequence([960000,group,permutation]))
    swaps=int(rng.integers(1,4))
    p=PERMS[permutation]
    paired=min(permutation,PERMS.index(p[2:]+p[:2]))
    if paired!=permutation:
        swaps=int(np.random.default_rng(np.random.SeedSequence([960000,group,paired])).integers(1,4))
    paths=[]
    for k in range(swaps):
        paths.append(dict(frames=int(rng.integers(30,101)),arc_m=float(rng.uniform(.095,.125)),
                          time_gamma=float(rng.uniform(.7,1.5)),route_flip=int(rng.choice([-1,1])),
                          pause_frames=int(rng.integers(3,16))))
    return dict(seed=[960000,group,permutation],swap_count_seed=[960000,group,paired],reveal_frames=int(rng.integers(30,46)),
                occlude_frames=int(rng.integers(65,91)),hold_frames=int(rng.integers(20,36)),
                swaps=paths,cover_centers=spec['cover_centers'])


def setup(scene,spec,permutation):
    model,data=scene.model,scene.data
    mujoco.mj_resetData(model,data)
    home=np.load('/workspace/robotics_assets/datasets/reach_voxel_grid/trajectories/v002043.npz')
    data.qpos[:8],data.qvel[:8],data.ctrl[:]=home['state_full_qpos'][0],home['state_full_qvel'][0],home['initial_ctrl']
    centers=np.array(spec['cover_centers'])
    scene.assignment={}
    for slot,color in enumerate(PERMS[permutation]):
        name=NAMES[color];owner=NAMES[4+slot//2]
        scene.assignment[name]=owner
        yaw=spec['slot_yaws'][slot];adr=scene.target_qadr[name]
        xy=centers[slot//2]+spec['slot_offsets'][slot]
        data.qpos[adr:adr+7]=(*xy,.018,np.cos(yaw/2),0,0,np.sin(yaw/2))
    for k,adr in enumerate(scene.cover_qadr.values()):
        data.qpos[adr:adr+7]=(-2 if k==0 else 2,0,.15,1,0,0,0)
    mujoco.mj_forward(model,data)
    mujoco.mj_step(model,data,250)
    # Scripted scene setup keeps covers parked and the robot at common home.
    data.qpos[:8],data.qvel[:8],data.ctrl[:]=home['state_full_qpos'][0],home['state_full_qvel'][0],home['initial_ctrl']
    for k,adr in enumerate(scene.cover_qadr.values()):
        data.qpos[adr:adr+7]=(-2 if k==0 else 2,0,.15,1,0,0,0)
    data.qvel[:]=0
    data.time=0
    mujoco.mj_forward(model,data)
    assert all(abs(p[2]-.016)<.001 for p in scene.target_positions().values())


def snapshot(scene,store,phase,valid):
    model,data=scene.model,scene.data
    mujoco.mj_forward(model,data)
    for key,value in (('qpos',data.qpos),('qvel',data.qvel),('ctrl',data.ctrl),
        ('joint_position',data.qpos[scene.qadr]),('joint_velocity',data.qvel[scene.dofs]),
        ('ee_position_base',data.site_xpos[scene.site]-BASE),
        ('ee_rotation_6d',rotation_6d(data.site_xmat[scene.site].reshape(3,3))),
        ('entity_xyz_world',np.stack([data.xpos[model.body(n).id] for n in NAMES])),
        ('entity_rotation_world',np.stack([data.xmat[model.body(n).id].reshape(3,3) for n in NAMES]))):
        store.setdefault(key,[]).append(value.copy())
    store.setdefault('phase',[]).append(phase)
    store.setdefault('action_valid',[]).append(valid)


def context(scene,plan):
    data=scene.data;centers=np.array(plan['cover_centers'])
    store={}
    def record(phase):
        snapshot(scene,store,phase,False)
        data.time+=.04
    for _ in range(plan['reveal_frames']):record('reveal')
    for frame in range(plan['occlude_frames']):
        t=(frame+1)/plan['occlude_frames']
        travel=min(t/.6,1);lower=max((t-.6)/.4,0)
        travel=travel**2*(3-2*travel);lower=lower**2*(3-2*lower)
        for k,adr in enumerate(scene.cover_qadr.values()):
            start=-.15 if k==0 else 1.25
            data.qpos[adr:adr+3]=(start+travel*(centers[k,0]-start),centers[k,1],.22+lower*(.002-.22))
        record('occlude')
    for k,path in enumerate(plan['swaps']):
        starts=np.stack([data.qpos[a:a+3].copy() for a in scene.cover_qadr.values()])
        objects={n:data.qpos[a:a+3].copy() for n,a in scene.target_qadr.items()}
        for frame in range(path['frames']):
            u=((frame+1)/path['frames'])**path['time_gamma']
            crossing=np.clip((u-.2)/.6,0.,1.)
            travel=crossing*crossing*(3-2*crossing)
            lane=min(u/.2,(1-u)/.2,1.)
            lane=lane*lane*(3-2*lane)
            cover_xy=[]
            for j,(name,adr) in enumerate(scene.cover_qadr.items()):
                end=centers[(1-j) if k%2==0 else j]
                xy=starts[j,:2]+travel*(end-starts[j,:2])
                side=np.sign(starts[j,0]-starts[:,0].mean())*path['route_flip']
                xy[1]+=side*path['arc_m']*lane
                position=np.r_[xy,.002]
                data.qpos[adr:adr+3]=position
                cover_xy.append(xy)
                for target,owner in scene.assignment.items():
                    if owner==name:
                        a=scene.target_qadr[target]
                        data.qpos[a:a+3]=objects[target]+position-starts[j]
            distance=np.abs(cover_xy[0]-cover_xy[1])
            if distance[0]<.160 and distance[1]<.130:
                raise GraspFailure('shuffle covers intersect')
            record('shuffle')
        for _ in range(path['pause_frames']):record('shuffle_pause')
    for _ in range(plan['hold_frames']):record('hold')
    data.qvel[:]=0
    mujoco.mj_forward(scene.model,data)
    return store


def save_segment(directory,store,offset=0,initial=None):
    directory.mkdir(parents=True,exist_ok=True)
    values={k:np.array(v) for k,v in store.items()}
    n=len(values['phase']);times=(offset+np.arange(n))/25
    state=dict(qpos=values['qpos'],qvel=values['qvel'],ctrl=values['ctrl'],timestamp_s=times)
    if initial is not None:
        state.update(initial_qpos=initial[0],initial_qvel=initial[1],initial_ctrl=initial[2])
    np.savez_compressed(directory/'sim_state.npz',**state)
    np.savez_compressed(directory/'observation.npz',joint_position=values['joint_position'],
        joint_velocity=values['joint_velocity'],ee_position_base_m=values['ee_position_base'],
        ee_rotation_6d=values['ee_rotation_6d'],timestamp_s=times)
    np.savez_compressed(directory/'supervision.npz',joint_command=values['ctrl'],phase=values['phase'],
        action_valid=values['action_valid'],timestamp_s=times)
    np.savez_compressed(directory/'labels.npz',entity_xyz_world_m=values['entity_xyz_world'],
        entity_xyz_base_m=values['entity_xyz_world']-BASE,entity_rotation_world=values['entity_rotation_world'],
        entity_names=np.array(NAMES),timestamp_s=times)
    return n


def collect_context(model,spec,group,permutation,directory):
    scene=Scene(model)
    setup(scene,spec,permutation)
    plan=shuffle_plan(group,permutation,spec)
    states=context(scene,plan)
    initial=(scene.data.qpos.copy(),scene.data.qvel.copy(),scene.data.ctrl.copy())
    frames=save_segment(directory,states)
    final=PERMS[permutation][2:]+PERMS[permutation][:2] if len(plan['swaps'])%2 else PERMS[permutation]
    final_permutation=PERMS.index(final)
    assert (permutation in SEEN)==(final_permutation in SEEN)
    meta=dict(layout_id=group,layout_seed=spec['seed'],initial_permutation=permutation,
        final_permutation=final_permutation,slot_object_ids=list(PERMS[permutation]),
        assignment=scene.assignment,frames=frames,fps=25,views=['top','left','right'],
        resolution=[320,320],context_plan=plan,initial_robot='e86 common settled home')
    write_json(directory/'context.json',meta)
    rows=[]
    for target_id,target in enumerate(NAMES[:4]):
        scene.data=mujoco.MjData(model)
        scene.data.qpos[:],scene.data.qvel[:],scene.data.ctrl[:]=initial
        scene.data.time=frames/25
        mujoco.mj_forward(model,scene.data)
        query_store={};decision={'t_occ':frames}
        def record(phase):
            if phase=='t_obj':
                decision['t_obj']=frames+len(query_store.get('phase',[]))
            else:
                snapshot(scene,query_store,phase,True)
        # Physical grasp jitter is tied to the slot, independent of color identity.
        slot=PERMS[permutation].index(target_id)
        rng=np.random.default_rng(np.random.SeedSequence([spec['seed'],slot,len(plan['swaps'])%2]))
        expert=execute(scene,target,rng,record)
        snapshot(scene,query_store,'done',False)
        path=directory/f't{target_id}'
        n=save_segment(path,query_store,frames,initial)
        dataset_split,evaluation_group=split(group,permutation)
        assert expert['selected_cover']==scene.assignment[target]
        query=dict(target_id=target_id,target_object_id=target,
            layout_id=group,layout_seed=spec['seed'],permutation=permutation,final_permutation=final_permutation,
            target_slot=slot,final_target_slot=final.index(target_id),context_path='..',context_frames=frames,expert_frames=n,frames=frames+n,
            split=dataset_split,evaluation_group=evaluation_group,decision_frames=decision,
            fps=25,resolution=[320,320],views=['top','left','right'],
            action_representation='absolute_joint_command_physical_rad_m',state_perturbations=[],
            objective='cover_drop_then_target_lift',success=True,**expert)
        write_json(path/'episode.json',query)
        rows.append(dict(path=f'g{group:03d}/p{permutation:02d}/t{target_id}',**query))
        rows[-1]['context_path']=f'g{group:03d}/p{permutation:02d}'
    return rows


def collect_group(job):
    root,group,pilot=job;root=Path(root)
    model=mujoco.MjModel.from_xml_path(str(root/'scene.xml'))
    ids=[SEEN[0]] if pilot else (SEEN if 32<=group<36 else range(24))
    rejections=[]
    for attempt in range(100):
        seed=960000+group+1000*attempt
        staging=root/f'.g{group:03d}_partial'
        if staging.exists():shutil.rmtree(staging)
        try:
            spec=layout_spec(seed)
            rows=[]
            for permutation in tqdm(ids,desc=f'Physics g{group:03d}',mininterval=10,leave=False):
                rows+=collect_context(model,spec,group,permutation,staging/f'p{permutation:02d}')
        except (GraspFailure,IKFailure) as error:
            rejections.append(dict(seed=seed,reason=str(error)))
            if staging.exists():shutil.rmtree(staging)
            continue
        staging.rename(root/f'g{group:03d}')
        write_json(root/f'g{group:03d}'/'layout.json',dict(**spec,rejected_seeds=rejections))
        return rows,spec,rejections
    raise RuntimeError(f'layout g{group:03d} failed100attempts: {rejections[-3:]}')


def initialize_output(output,pilot):
    output.mkdir(parents=True,exist_ok=False)
    (output/'README.md').write_text('# Shuffle-pick dataset\n\nStatus: collecting. No training; RGB3/masks/centers/states. Actual scene,splits,seed,control25Hz and no expert perturbation:config.json.\n')
    shutil.copyfile(SOURCE_SCENE,output/'scene.xml')
    sources=[Path(__file__),Path(__file__).with_name('render_shuffle.py'),
             Path(__file__).with_name('audit_shuffle.py'),Path(__file__).with_name('shuffle_data.py'),
             ROOT/'controllers/oracle_pick.py',ROOT/'OATFlow/dataset/expert.py',output/'scene.xml']
    config=dict(status='collecting_physics',pilot=pilot,layouts=40,train_layouts=list(range(32)),
        val_layouts=list(range(32,36)),test_layouts=list(range(36,40)),
        train_permutations=SEEN,unseen_permutations=UNSEEN,permutation_order=[list(p) for p in PERMS],
        views=['top','left','right'],resolution=[320,320],fps=25,base_origin_world_m=BASE.tolist(),
        expected_episodes=4 if pilot else 3712,expected_contexts=1 if pilot else 928,
        mid_rollout_perturbation=False,robot_start='e86 common settled home',
        expert='original contact oracle: open/deposit cover, grasp/lift target',
        expert_control='25Hz command zero-order hold;500Hz physics; pre-step observation/command',
        normalized=False,action_units=['rad']*5+['m'],
        mask_storage='lossless uint8 instance IDs + little-endian bitpacked binary masks',
        cameras='same as e91; camera parameters in scene.xml; no calibration export',
        layout_seed_base=960000,geometry_range='cover centers +/-25mmXY; slot offsets +/-6mmX,+/-14mmY',
        context_storage='one shared context per(layout,permutation),four expert continuations')
    config['code_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    config['software']={'mujoco':mujoco.__version__,'numpy':np.__version__}
    write_json(output/'config.json',config)
    with (output/'README.md').open('a') as handle:
        handle.write('\nActual status: collecting physics. '+
                     ('Pilot:1context,4query rollouts.\n' if pilot else 'Full:40layouts,928contexts,3712query rollouts.\n'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    output=args.output
    initialize_output(output,args.pilot)
    start=time.perf_counter();rows=[]
    groups=[0] if args.pilot else range(40)
    with ProcessPoolExecutor(max_workers=min(args.workers,len(groups)),mp_context=mp.get_context('spawn')) as pool:
        pending=[pool.submit(collect_group,(str(output),g,args.pilot)) for g in groups]
        for future in tqdm(as_completed(pending),total=len(pending),desc='Complete physical layout groups',mininterval=5):
            part,spec,rejections=future.result();rows+=part
            print(json.dumps(dict(layout_seed=spec['seed'],episodes=len(part),rejected=len(rejections))),flush=True)
            write_json(output/'progress.json',dict(stage='physics',episodes=len(rows),elapsed_s=time.perf_counter()-start))
    rows.sort(key=lambda r:(r['layout_id'],r['permutation'],r['target_id']))
    expected=4 if args.pilot else 3712
    assert len(rows)==expected
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config=json.loads((output/'config.json').read_text());config['status']='physics_complete'
    config['physics_elapsed_s']=time.perf_counter()-start
    config['episodes_by_group']=dict(Counter(r['evaluation_group'] for r in rows))
    write_json(output/'config.json',config)
    print(json.dumps(config),flush=True)


if __name__=='__main__':main()
