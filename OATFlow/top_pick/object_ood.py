"""Broader expert-feasible spatial OOD distributions for unchanged E131."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mujoco
import numpy as np
import torch
from tqdm import tqdm

from controllers.oracle_pick import GraspFailure,IKFailure
from OATFlow.data_tools.collect_shuffle import Scene,NAMES,PERMS
from OATFlow.reach.pointcloud import PointCloudReach
from OATFlow.pick.scene import execute
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.cloud_evaluate import summarize
from OATFlow.top_pick.collect import setup
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.object_cloud import ROOT,CACHE,extract_cloud
from OATFlow.top_pick.object_evaluate import renderers,rollout
from OATFlow.top_pick.wide import LOW,HIGH
from OATFlow.top_pick.xyz_map import camera_calibration

OUT=ROOT/'test_ood'
NOTE=Path('docs/research/e131_ood_README.md')
REGIONS=[('x_low',[-1,0]),('x_high',[1,0]),('y_low',[0,-1]),('y_high',[0,1]),
         ('x_low_y_low',[-1,-1]),('x_low_y_high',[-1,1]),('x_high_y_low',[1,-1]),('x_high_y_high',[1,1])]
BANDS=[(.02,.04),(.04,.06),(.06,.08)]
QUOTA=4
ATTEMPTS=48


def prepare_cell(job):
    index,region,sign,band_index=job
    torch.set_num_threads(1)
    rng=np.random.default_rng(np.random.SeedSequence([131091,index]))
    physics,visual,depth_renderer,seg,option=renderers();scene=Scene(physics);display=mujoco.MjData(visual)
    rows=[];rejections=[];accepted_by_target={};sign=np.array(sign)
    with depth_renderer,seg:
        for target in range(4):
            accepted=0;attempts=0
            for attempt in range(ATTEMPTS):
                if accepted==QUOTA:break
                attempts+=1;positions=rng.uniform(LOW+.045,HIGH-.045,(4,2));yaw=rng.uniform(-np.pi,np.pi,4)
                tangent=rng.uniform(LOW+.025,HIGH-.025)
                delta=rng.uniform(*BANDS[band_index],size=2);candidate=[];stage='geometry';group=None
                try:
                    for group,side in [('outside',1),('inside',-1)]:
                        xy=positions.copy();xy[target]=tangent
                        for axis in np.flatnonzero(sign):
                            boundary=LOW[axis] if sign[axis]<0 else HIGH[axis]
                            xy[target,axis]=boundary+sign[axis]*side*delta[axis]
                        if min(np.linalg.norm(a-b) for i,a in enumerate(xy) for b in xy[i+1:])<.055:
                            raise ValueError('object separation below55mm')
                        stage='setup';initial=setup(scene,dict(positions=xy,yaw=yaw),0)
                        actual=np.stack([scene.data.xpos[physics.body(name).id].copy() for name in NAMES[:4]])
                        target_xy=actual[target,:2];outside=np.maximum(np.maximum(LOW-target_xy,target_xy-HIGH),0)
                        if group=='outside':
                            if not BANDS[band_index][0]-.001<=outside.max()<=BANDS[band_index][1]+.001:
                                raise ValueError('target drifted outside intended OOD band')
                            for axis in np.flatnonzero(sign):
                                boundary=LOW[axis] if sign[axis]<0 else HIGH[axis]
                                if sign[axis]*(target_xy[axis]-boundary)<BANDS[band_index][0]-.001:
                                    raise ValueError('target drifted outside intended region')
                        elif not ((target_xy>=LOW)&(target_xy<=HIGH)).all():
                            raise ValueError('control drifted outside training support')
                        stage='front_visibility';display.qpos[:]=initial[0];mujoco.mj_forward(visual,display)
                        depth_renderer.update_scene(display,camera='cam_front',scene_option=option);depth=depth_renderer.render().copy()
                        valid=np.isfinite(depth)&(depth>0)&(depth<=6.5535);codes=np.zeros(depth.shape,np.uint16);codes[valid]=np.rint(depth[valid]/.002).astype(np.uint16)
                        seg.update_scene(display,camera='cam_front',scene_option=option);labels=seg.render()
                        geoms=np.flatnonzero(visual.geom_bodyid==visual.body(NAMES[target]).id)
                        mask=np.isin(labels[...,0],geoms)&(labels[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
                        K,T=camera_calibration(visual,display);_,good,count=extract_cloud(codes,mask,K,T,np.random.default_rng(0))
                        if mask.sum()<16 or not good:raise ValueError(f'insufficient front target cloud:raw{mask.sum()},filtered{count}')
                        stage='physical_expert';expert=execute(scene,NAMES[target],lambda phase:None)
                        pair=index*100000+target*1000+accepted
                        row=dict(path=f'{region}/b{band_index}/t{target}_r{accepted}/{group}',evaluation_group=group,
                            layout_id=pair,permutation=0,target_id=target,target_slot=PERMS[0].index(target),target_object_id=NAMES[target],
                            pair=pair,cell=index,edge=region,band_index=band_index,offset_band_m=list(BANDS[band_index]),
                            attempt=attempt,seed=[131091,index],target_xy=target_xy.tolist(),all_object_xyz=actual.tolist(),
                            outside_distance_m=float(outside.max()),initial_target_filtered_pixels=count,expert=expert)
                        candidate.append((row,initial))
                except (GraspFailure,IKFailure,ValueError) as error:
                    rejections.append(dict(target=target,attempt=attempt,group=group,stage=stage,error=str(error)))
                    continue
                assert len(candidate)==2
                for row,initial in candidate:
                    path=OUT/'states'/row['path'];path.mkdir(parents=True,exist_ok=False)
                    np.savez_compressed(path/'sim_state.npz',**dict(zip(('initial_qpos','initial_qvel','initial_ctrl'),initial)))
                    rows.append(row)
                accepted+=1
            accepted_by_target[str(target)]=dict(pairs=accepted,attempts=attempts)
    summary=dict(cell=index,region=region,band_index=band_index,offset_band_m=list(BANDS[band_index]),
        accepted_pairs=len(rows)//2,episodes=len(rows),targets=accepted_by_target,
        rejections_by_stage=dict(Counter(r['stage'] for r in rejections)),status='available' if rows else 'no_accepted_pairs')
    write(OUT/f'preparation_cell_{index:02d}.json',dict(summary,rows=rows,rejections=rejections))
    return rows,summary


def group_summary(results,rows):
    mapping={r['path']:r for r in rows};groups={}
    for key in ['band_index','edge']:
        groups[key]={}
        for value in sorted({row[key] for row in rows},key=str):
            groups[key][str(value)]={}
            for group in ['inside','outside']:
                selected=[r for r in results if mapping[r['path']][key]==value and r['group']==group]
                if selected:groups[key][str(value)][group]=summarize(selected,0)
    groups['cells']=[]
    for cell in sorted({row['cell'] for row in rows}):
        entry=dict(cell=cell,edge=next(r['edge'] for r in rows if r['cell']==cell),band_index=next(r['band_index'] for r in rows if r['cell']==cell))
        for group in ['inside','outside']:
            selected=[r for r in results if mapping[r['path']]['cell']==cell and r['group']==group]
            entry[group]=summarize(selected,0)
        groups['cells'].append(entry)
    pairs={}
    for r in results:pairs.setdefault(mapping[r['path']]['pair'],{})[r['group']]=r['success']
    assert all(set(v)=={'inside','outside'} for v in pairs.values())
    groups['paired']=dict(Counter('both_success' if v['inside'] and v['outside'] else 'inside_only' if v['inside'] else 'outside_only' if v['outside'] else 'both_failure' for v in pairs.values()))
    return groups


def plot(rows,results,coverage):
    by_path={r['path']:r for r in results}
    training=np.load(CACHE/'test_xy/training_xy.npy')
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    ax.scatter(*training.T,s=12,c='#b2bdc6',label='Actual train centers')
    ax.add_patch(Rectangle(LOW,*(HIGH-LOW),fill=False,ls='--',color='#576876',label='Pick-training center support'))
    for group,color in [('inside','#158a72'),('outside','#da752d')]:
        for success,marker in [(True,'o'),(False,'x')]:
            points=np.array([r['target_xy'] for r in rows if r['evaluation_group']==group and by_path[r['path']]['success']==success])
            if len(points):ax.scatter(*points.T,c=color,marker=marker,s=35,alpha=.7,label=f'{group}: {"success" if success else "failure"}')
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',title='E131 spatial OOD: expert-feasible / front-visible accepted pairs')
    ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=8);fig.savefig(OUT/'xy_results.png',dpi=180);plt.close(fig)
    matrix=np.zeros((len(REGIONS),len(BANDS)),int)
    for c in coverage:matrix[c['cell']//3,c['band_index']]=c['accepted_pairs']
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained');im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=16)
    for i in range(8):
        for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
    ax.set(xticks=range(3),xticklabels=['2-4 cm','4-6 cm','6-8 cm'],yticks=range(8),yticklabels=[r[0] for r in REGIONS],title='Accepted physical pairs / 16 desired per cell (0 = unavailable)')
    fig.colorbar(im,ax=ax,label='Accepted pairs');fig.savefig(OUT/'coverage.png',dpi=180);plt.close(fig)


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=False);shutil.copyfile(NOTE,OUT/'README.md');torch.set_num_threads(4)
    checkpoint=ROOT/'train/final.pt';checkpoint_hash=sha(checkpoint)
    protocol=json.loads((ROOT/'test_k15/config.json').read_text());assert checkpoint_hash==protocol['checkpoint_sha256']
    config=dict(status='preparing',checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,step=23520,
        training=protocol['training'],training_support_world_m=[LOW.tolist(),HIGH.tolist()],regions=REGIONS,offset_bands_m=BANDS,
        desired_pairs_per_color=QUOTA,maximum_candidates_per_color=ATTEMPTS,workers=4,seed=131091,
        selection='expert success in both paired scenes + front-visible target,all fixed before learned-model loading',
        H=25,K=15,flow_steps=10,noise_seed=0,frame_budget=300,policy_hz=25,servo_hz=50,physics_hz=500,
        video=False,training_run=False,source_sha256=sha(Path(__file__)),rollout_source_sha256=sha(Path('OATFlow/top_pick/object_evaluate.py')))
    write(OUT/'config.json',config);started=time.perf_counter();rows=[];coverage=[]
    jobs=[(i*3+j,name,sign,j) for i,(name,sign) in enumerate(REGIONS) for j in range(3)]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(prepare_cell,job) for job in jobs]
        for future in tqdm(as_completed(futures),total=len(jobs),desc='Expert-feasible OOD distributions',mininterval=5):
            accepted,summary=future.result();rows.extend(accepted);coverage.append(summary)
            write(OUT/'coverage.json',sorted(coverage,key=lambda c:c['cell']))
            print(json.dumps(summary),flush=True)
    rows.sort(key=lambda r:(r['cell'],r['target_id'],r['pair'],r['evaluation_group']=='inside'));coverage.sort(key=lambda c:c['cell'])
    assert rows and len({r['path'] for r in rows})==len(rows)
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(OUT/'preparation.json',dict(status='complete',elapsed_s=time.perf_counter()-started,episodes=len(rows),pairs=len(rows)//2,
        coverage=coverage,total_candidates=sum(v['attempts'] for c in coverage for v in c['targets'].values())))
    state_hashes={r['path']:sha(OUT/'states'/r['path']/'sim_state.npz') for r in rows}
    config.update(status='evaluating',episodes=len(rows),preparation_s=time.perf_counter()-started);write(OUT/'config.json',config)
    torch.backends.cuda.matmul.allow_tf32=True
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);assert saved['step']==23520
    policy=PointCloudReach().cuda();policy.load_state_dict(saved['model'],strict=True);policy.eval().requires_grad_(False);del saved
    physics,visual,renderer,seg,option=renderers();results=[];start_eval=time.perf_counter()
    with renderer,seg:
        for start in tqdm(range(0,len(rows),32),desc='E131 extended spatial OOD episodes',mininterval=5):
            current=rollout(rows[start:start+32],policy,physics,visual,renderer,seg,option,OUT,data_root=OUT/'states')
            with (OUT/'results.jsonl').open('a') as f:
                for r in current:f.write(json.dumps(r)+'\n')
            results.extend(current);progress=summarize(results,time.perf_counter()-start_eval);progress.update(status='evaluating',complete=len(results),total=len(rows))
            write(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    summary=summarize(results,time.perf_counter()-start_eval);summary.update(status='complete',preparation_s=config['preparation_s'],
        total_elapsed_s=time.perf_counter()-started,distributions=group_summary(results,rows),coverage=coverage)
    write(OUT/'summary.json',summary);plot(rows,results,coverage)
    assert checkpoint_hash==sha(checkpoint) and {r['path'] for r in results}=={r['path'] for r in rows}
    assert all(sha(OUT/'states'/r['path']/'sim_state.npz')==state_hashes[r['path']] for r in rows)
    assert all(r['expert']['max_bilateral_hold_frames']>=10 for r in rows)
    config['status']='complete';write(OUT/'config.json',config);write(OUT/'progress.json',summary)
    write(OUT/'verification.json',dict(status='passed',checkpoint_unchanged=True,all_accepted_pairs_evaluated=True,
        all_initial_states_unchanged=True,all_expert_validations_passed=True,training_run=False,video=False))
    text=NOTE.read_text().replace('Status: PLANNED.','Status: COMPLETE.')
    text+='\n\n## Results\n\n| Offset band | Inside controls | Outside targets |\n|---|---:|---:|\n'
    for j,(lo,hi) in enumerate(BANDS):
        groups=summary['distributions']['band_index'].get(str(j),{})
        cells=[]
        for group in ['inside','outside']:
            r=groups.get(group);cells.append(f'{r["successes"]}/{r["cases"]} ({100*r["success_rate"]:.2f}%)' if r else 'Unavailable')
        text+=f'| {lo*100:.0f}-{hi*100:.0f} cm | {cells[0]} | {cells[1]} |\n'
    text+='\nExact per-cell counts and rejected candidate reasons are in coverage.json and preparation_cell_*.json; summary.json retains all groups.\n'
    NOTE.write_text(text);(OUT/'README.md').write_text(text)
    with Path('docs/research/REPORT.md').open('a') as report:
        report.write('\nE131 extended OOD completed:'+json.dumps(summary['by_group'])+'. Details:e131_ood_README.md.\n')
    print(json.dumps(dict(status='complete',episodes=len(results),groups=summary['by_group'],bands=summary['distributions']['band_index'])),flush=True)


if __name__=='__main__':main()
