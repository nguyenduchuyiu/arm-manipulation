"""Closed-loop learned top pick on paired unseen-permutation/spatial cases."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import Scene,NAMES
from OATFlow.simulation.contacts import jaw_contacts
from OATFlow.simulation.control import step as servo_step
from OATFlow.simulation.io import encoder
from OATFlow.simulation.pose import PoseIK,rotation_matrix
from OATFlow.simulation.render import visual_model
from OATFlow.top_pick.collect import DATA,ROOT,SEEN,UNSEEN
from OATFlow.top_pick.center import load_policy


def temporal_target(plans,frame,coefficient):
    """Average world poses for one endpoint, oldest prediction weighted first."""
    candidates=[(p[frame-start],r[frame-start],j[frame-start])
                for start,p,r,j in plans if start<=frame<start+len(p)]
    weights=np.exp(-coefficient*np.arange(len(candidates)))
    position=np.average([v[0] for v in candidates],axis=0,weights=weights)
    # Project the weighted matrix mean back onto SO(3), including det=+1.
    mean=np.average([v[1] for v in candidates],axis=0,weights=weights)
    u,_,vh=np.linalg.svd(mean)
    rotation=u@np.diag([1.,1.,np.linalg.det(u@vh)])@vh
    jaw=float(np.average([v[2] for v in candidates],weights=weights))
    return position,rotation,jaw,len(candidates)


def cases(data,pilot=False):
    rows=[json.loads(s) for s in (data/'manifest.jsonl').read_text().splitlines()]
    if pilot:return [r for r in rows if r['layout_id']==0 and r['permutation']==0]
    result=[]
    for group,layouts,perms in [('train',(0,31),SEEN[::4]),('composition',(0,31),UNSEEN[::2]),
                               ('spatial',(36,39),SEEN[::4]),('combined',(36,39),UNSEEN[::2])]:
        result += [r for r in rows if r['evaluation_group']==group and r['layout_id'] in layouts and r['permutation'] in perms]
    return result


def rollout(rows,data,model,checkpoint,out,oracle=False,execute_chunk=10,temporal_ensemble_coeff=None):
    physics=mujoco.MjModel.from_xml_path(str(data/'scene.xml'))
    visual=visual_model(data/'scene.xml',object_bodies=NAMES)
    visual.vis.global_.offwidth=visual.vis.global_.offheight=320
    display=mujoco.MjData(visual)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    base=np.array([.539369022999,.0639727628473,0.])
    scenes=[]
    for row in rows:
        scene=Scene(physics)
        with np.load(data/row['path']/'sim_state.npz') as z:
            for key in ('qpos','qvel','ctrl'):getattr(scene.data,key)[:]=z['initial_'+key]
        mujoco.mj_forward(physics,scene.data)
        value=dict(model=physics,data=scene.data,scene=scene,row=row,held=scene.data.ctrl.copy(),
                   ik=PoseIK((physics,scene.qadr,scene.dofs[:5],scene.site)),
                   frames=[],commands=[],objects=[],contacts=[],qpos=[],qvel=[],ctrl=[],
                   success=False,done=False,reason='timeout',streak=0,longest=0)
        if temporal_ensemble_coeff is not None:
            value.update(temporal_plans=[],ensemble_counts=[],target_jaws=[])
        value['initial_qpos']=scene.data.qpos.copy();value['initial_qvel']=scene.data.qvel.copy();value['initial_ctrl']=scene.data.ctrl.copy()
        value['initial_objects']=np.stack([scene.data.xpos[physics.body(n).id] for n in NAMES[:4]])
        if oracle:
            with np.load(data/row['path']/'observation.npz') as z:
                value['oracle_ee']=z['ee_position_base_m']+base
                value['oracle_R']=rotation_matrix(z['ee_rotation_6d'])
            with np.load(data/row['path']/'supervision.npz') as z:value['oracle_jaw']=z['joint_command'][:,5]
        scenes.append(value)
    generator=torch.Generator(device='cuda').manual_seed(0)
    with mujoco.Renderer(visual,height=320,width=320) as renderer:
        for frame in tqdm(range(300),desc=f'Rollout {rows[0]["evaluation_group"]}',mininterval=10,leave=False):
            active=[s for s in scenes if not s['done']]
            if not active:break
            if frame%execute_chunk==0 and not oracle:
                rgbs=[];joints=[]
                for s in active:
                    display.qpos[:]=s['data'].qpos;mujoco.mj_forward(visual,display)
                    renderer.update_scene(display,camera='cam_top',scene_option=option)
                    rgbs.append(renderer.render().copy());joints.append(s['data'].qpos[s['scene'].qadr])
                rgb=torch.from_numpy(np.stack(rgbs)).permute(0,3,1,2).cuda()
                joint=torch.tensor(np.stack(joints),dtype=torch.float32,device='cuda')
                targets=torch.tensor([s['row']['target_id'] for s in active],device='cuda')
                # Identical query noise within every paired scene at each replan.
                noise=model.flow.sample_noise(1,'cuda',generator).expand(len(active),-1,-1)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    plans=model.sample(model.encode(rgb),joint,targets,noise,10).float().cpu().numpy()
                for s,plan in zip(active,plans):
                    s['plan']=plan;s['anchor']=s['data'].site_xpos[s['scene'].site].copy()
                    s['anchor_R']=s['data'].site_xmat[s['scene'].site].reshape(3,3).copy()
                    if temporal_ensemble_coeff is not None:
                        # Each chunk has its own measured anchor. Combine absolute poses.
                        s['temporal_plans']=[p for p in s['temporal_plans'] if frame<p[0]+len(p[1])]
                        s['temporal_plans'].append((frame,s['anchor']+plan[:,:3],
                            rotation_matrix(plan[:,3:9])@s['anchor_R'],plan[:,9].copy()))
            for s in active:
                state=s['data'];site=s['scene'].site
                if oracle:
                    i=min(frame+1,len(s['oracle_ee'])-1)
                    target=s['oracle_ee'][i];r=s['oracle_R'][i];jaw=s['oracle_jaw'][min(frame,len(s['oracle_jaw'])-1)]
                elif temporal_ensemble_coeff is not None:
                    target,r,jaw,count=temporal_target(s['temporal_plans'],frame,temporal_ensemble_coeff)
                    s['ensemble_counts'].append(count);s['target_jaws'].append(jaw)
                else:
                    action=s['plan'][frame%execute_chunk]
                    target=s['anchor']+action[:3];r=rotation_matrix(action[3:9])@s['anchor_R'];jaw=action[9]
                command=s['ik'].command(s,target,jaw,r)
                command=np.clip(command,*physics.actuator_ctrlrange.T)
                servo_step(physics,state,s['held'],command);s['held']=command.copy()
                objects=np.stack([state.xpos[physics.body(n).id] for n in NAMES[:4]])
                contacts=np.array([len(jaw_contacts(physics,state,n)) for n in NAMES[:4]])
                gains=objects[:,2]-s['initial_objects'][:,2];t=s['row']['target_id']
                s['streak']=s['streak']+1 if gains[t]>.04 and contacts[t]==2 else 0
                s['longest']=max(s['longest'],s['streak'])
                wrong=[i for i in range(4) if i!=t and gains[i]>.04]
                if wrong:s.update(done=True,reason='wrong_object',wrong_object=NAMES[wrong[0]])
                elif s['streak']>=10:s.update(done=True,success=True,reason='success')
                assert np.isfinite(state.qpos).all()
                s['commands'].append(command);s['objects'].append(objects);s['contacts'].append(contacts)
                for key in ('qpos','qvel','ctrl'):s[key].append(getattr(state,key).copy())
                if frame%3==0 or s['done']:
                    display.qpos[:]=state.qpos;mujoco.mj_forward(visual,display)
                    renderer.update_scene(display,camera='cam_top',scene_option=option)
                    s['frames'].append(renderer.render().copy())
    results=[]
    for s in scenes:
        row=s['row'];name=row['evaluation_group']+'_'+row['path'].replace('/','_')
        result=dict(path=row['path'],group=row['evaluation_group'],target=NAMES[row['target_id']],
                    slot=row['target_slot'],success=s['success'],reason=s['reason'],frames=len(s['commands']),
                    bilateral_lift_streak=s['longest'],wrong_object=s.get('wrong_object'),
                    max_target_lift_mm=float((np.array(s['objects'])[:,row['target_id'],2]-s['initial_objects'][row['target_id'],2]).max()*1000),
                    max_ik_error_mm=float(max(s['ik_errors'])*1000))
        trace_keys=['qpos','qvel','ctrl','commands','objects','contacts']
        if temporal_ensemble_coeff is not None:
            trace_keys+=['ik_targets','ik_rotation_targets','target_jaws','ensemble_counts']
        np.savez_compressed(out/f'{name}.npz',**{key:np.stack(s[key]) for key in trace_keys},
                            initial_qpos=s['initial_qpos'],initial_qvel=s['initial_qvel'],initial_ctrl=s['initial_ctrl'])
        if row['target_id']==0:
            writer=encoder(out/f'{name}.mp4',320,320)
            for image in s['frames']:
                for _ in range(3):writer.stdin.write(image.tobytes())
            writer.stdin.close();assert writer.wait()==0
            Image.fromarray(s['frames'][-1]).save(out/f'{name}_final.png')
        results.append(result)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle',action='store_true')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'train/best.pt')
    p.add_argument('--output',type=Path)
    p.add_argument('--execute-chunk',type=int,default=10)
    p.add_argument('--temporal-ensemble-coeff',type=float)
    args=p.parse_args()
    if not 1<=args.execute_chunk<=25:p.error('execute chunk must be between 1 and 25')
    if args.temporal_ensemble_coeff is not None:
        if args.oracle or args.execute_chunk!=1 or not np.isfinite(args.temporal_ensemble_coeff) or args.temporal_ensemble_coeff<0:
            p.error('temporal ensemble requires learned K1 and a finite nonnegative coefficient')
    suffix=f'k{args.execute_chunk}'+('_te' if args.temporal_ensemble_coeff is not None else '')
    out=args.output or ROOT/(('oracle_' if args.oracle else 'test_')+suffix);out.mkdir(exist_ok=False)
    data=ROOT/'pilot' if args.oracle else DATA
    checkpoint=args.checkpoint;model=None;saved=None
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    if not args.oracle:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        model=load_policy(saved).cuda().eval()
    rows=cases(data,args.oracle)
    config=dict(status='running',data=str(data),checkpoint=str(checkpoint) if saved else 'recorded measured EE oracle',
                checkpoint_step=saved['step'] if saved else None,cases=len(rows),H=25,K=args.execute_chunk,flow_steps=10,
                noise_seed=0,frame_budget=300,policy_hz=25,servo_hz=50,
                temporal_ensemble=args.temporal_ensemble_coeff is not None,
                temporal_ensemble_coeff=args.temporal_ensemble_coeff,
                ensemble_space='absolute world XYZ,SO3 projected matrix mean,physical jaw' if args.temporal_ensemble_coeff is not None else None,
                ensemble_weighting='exp(-coefficient*i),i=0 oldest valid prediction' if args.temporal_ensemble_coeff is not None else None,
                replan_hz=25/args.execute_chunk,
                selection='two layouts x four permutations x four target queries per group',
                initial_state='recorded common home, four visible objects, covers parked',
                success='correct target lift>4cm, bilateral contacts,ten consecutive frames;wrong lift fails',
                oracle=args.oracle,training=saved['config'] if saved else None)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    readme=('# '+out.parent.name+' '+out.name+'\n\nStatus: running. '+json.dumps(config)+
            '\n\nActual training configuration and auxiliary-head settings are included in config.json. '
            'This test uses held-out states only for initialization and success scoring. '
            'Only top RGB,target color and joints enter learned policy. Oracle mode, if enabled, is a controller ceiling test.\n')
    (out/'README.md').write_text(readme)
    results=[];started=time.perf_counter()
    for offset in tqdm(range(0,len(rows),8),desc='Paired home-start pick batches',mininterval=5):
        current=rollout(rows[offset:offset+8],data,model,checkpoint,out,args.oracle,
                        args.execute_chunk,args.temporal_ensemble_coeff);results+=current
        with (out/'results.jsonl').open('a') as stream:
            for r in current:stream.write(json.dumps(r)+'\n')
        print(json.dumps(dict(cases=len(results),successes=sum(r['success'] for r in results))),flush=True)
    summary=dict(cases=len(results),successes=sum(r['success'] for r in results),
                 success_rate=sum(r['success'] for r in results)/len(results),elapsed_s=time.perf_counter()-started,
                 by_group={group:dict(cases=sum(r['group']==group for r in results),successes=sum(r['group']==group and r['success'] for r in results)) for group in sorted({r['group'] for r in results})},
                 reasons=Counter(r['reason'] for r in results))
    for group,score in summary['by_group'].items():score['success_rate']=score['successes']/score['cases']
    summary['by_target']={target:dict(cases=sum(r['target']==target for r in results),successes=sum(r['target']==target and r['success'] for r in results)) for target in NAMES[:4]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');config['status']='complete'
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'README.md').write_text(readme.replace('Status: running','Status: complete')+'\n'+json.dumps(summary)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
