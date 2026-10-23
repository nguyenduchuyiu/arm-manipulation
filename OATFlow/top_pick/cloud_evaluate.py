"""Evaluate final E119 on every held-out episode, front RGB only; no video rendering."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import Scene, NAMES
from OATFlow.simulation.contacts import jaw_contacts
from OATFlow.simulation.control import step as servo_step
from OATFlow.simulation.pose import PoseIK, rotation_matrix
from OATFlow.simulation.render import visual_model
from OATFlow.top_pick.cloud import CloudPickPolicy, ROOT, DATA
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.xyz_map import camera_calibration


def rollout(rows, model, physics, visual, renderer, option, out, frame_budget=300,k=10,data_root=DATA):
    assert 1<=k<=25
    device=next(model.parameters()).device
    display = mujoco.MjData(visual); scenes = []
    for row in rows:
        scene = Scene(physics)
        with np.load(data_root/row['path']/'sim_state.npz') as z:
            for field in ('qpos','qvel','ctrl'):getattr(scene.data,field)[:] = z['initial_'+field]
        mujoco.mj_forward(physics,scene.data)
        state = scene.data
        scenes.append(dict(data=state,row=row,scene=scene,held=state.ctrl.copy(),
            ik=PoseIK((physics,scene.qadr,scene.dofs[:5],scene.site)),
            initial_qpos=state.qpos.copy(),initial_qvel=state.qvel.copy(),initial_ctrl=state.ctrl.copy(),
            initial_objects=np.stack([state.xpos[physics.body(n).id] for n in NAMES[:4]]),
            qpos=[],commands=[],objects=[],contacts=[],success=False,done=False,reason='timeout',
            streak=0,longest=0))
    generator=torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for frame in tqdm(range(frame_budget),desc=f'Front camera pick H25/K{k}',mininterval=10,leave=False):
            active=[s for s in scenes if not s['done']]
            if not active:break
            if frame%k==0:
                pixels=[];joints=[];scene_inputs=[]
                for s in active:
                    display.qpos[:]=s['data'].qpos;mujoco.mj_forward(visual,display)
                    renderer.update_scene(display,camera='cam_front',scene_option=option)
                    pixels.append(renderer.render().copy());joints.append(s['data'].qpos[s['scene'].qadr])
                    if hasattr(model,'geometry_patch'):
                        renderer.enable_depth_rendering();depth=renderer.render();renderer.disable_depth_rendering()
                        valid=np.isfinite(depth)&(depth>0)&(depth<=6.5535)
                        codes=np.zeros(depth.shape,np.uint16);codes[valid]=np.rint(depth[valid]/.002).astype(np.uint16)
                        K,transform=camera_calibration(visual,display)
                        scene_inputs.append((codes,K,transform))
                rgb=torch.from_numpy(np.stack(pixels)).permute(0,3,1,2).to(device)
                joint=torch.tensor(np.stack(joints),dtype=torch.float32,device=device)
                target=torch.tensor([s['row']['target_id'] for s in active],device=device)
                noise=model.flow.sample_noise(1,device,generator).expand(len(active),-1,-1)
                with torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext():
                    extra={}
                    if scene_inputs:
                        geometry={key:torch.as_tensor(np.stack([s[i] for s in scene_inputs]),device=device,dtype=torch.float32)
                                  for i,key in enumerate(('depth','intrinsics','camera_to_base'))}
                        extra=dict(scene=geometry)
                    plans=model.sample(model.encode(rgb),joint,target,noise,steps=10,**extra).float().cpu().numpy()
                for s,plan in zip(active,plans):
                    assert np.isfinite(plan).all()
                    s['plan']=plan;s['anchor']=s['data'].site_xpos[s['scene'].site].copy()
                    s['rotation']=s['data'].site_xmat[s['scene'].site].reshape(3,3).copy()
            for s in active:
                state=s['data'];action=s['plan'][frame%k]
                command=s['ik'].command(s,s['anchor']+action[:3],action[9],rotation_matrix(action[3:9])@s['rotation'])
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
                s['qpos'].append(state.qpos.copy());s['commands'].append(command.copy())
                s['objects'].append(objects);s['contacts'].append(contacts)
    results=[]
    for s in scenes:
        row=s['row'];name=row['path'].replace('/','_')
        record=dict(path=row['path'],group=row['evaluation_group'],layout=row['layout_id'],
            permutation=row['permutation'],target=NAMES[row['target_id']],slot=row['target_slot'],
            success=s['success'],reason=s['reason'],frames=len(s['commands']),
            bilateral_lift_streak=s['longest'],wrong_object=s.get('wrong_object'),
            max_target_lift_mm=float((np.array(s['objects'])[:,row['target_id'],2]-s['initial_objects'][row['target_id'],2]).max()*1000),
            max_ik_error_mm=float(max(s['ik_errors'])*1000))
        np.savez_compressed(out/f'{name}.npz',qpos=np.asarray(s['qpos'],np.float32),
            commands=np.asarray(s['commands'],np.float32),objects=np.asarray(s['objects'],np.float32),
            contacts=np.asarray(s['contacts'],np.uint8),initial_qpos=s['initial_qpos'],
            initial_qvel=s['initial_qvel'],initial_ctrl=s['initial_ctrl'])
        results.append(record)
    return results


def summarize(results, elapsed):
    summary=dict(cases=len(results),successes=sum(r['success'] for r in results),
        success_rate=sum(r['success'] for r in results)/len(results),elapsed_s=elapsed,
        reasons=dict(Counter(r['reason'] for r in results)))
    for key in ('group','target'):
        summary['by_'+key]={}
        for value in sorted({r[key] for r in results}):
            current=[r for r in results if r[key]==value]
            successes=sum(r['success'] for r in current)
            summary['by_'+key][value]=dict(cases=len(current),successes=successes,success_rate=successes/len(current))
    return summary


def main(resume,root=ROOT,policy_class=CloudPickPolicy,readme='docs/research/e119_test_README.md',expected_cases=3200,k=10):
    assert 1<=k<=25
    out=root/f'test_k{k}';out.mkdir(exist_ok=resume)
    checkpoint=out/'checkpoint.pt'
    if not resume:
        os.link(root/'train/final.pt',checkpoint)
        shutil.copyfile(readme,out/'README.md')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==saved['config']['total_steps']
    if saved['config'].get('requested_updates') is None:
        assert saved['config']['effective_epochs']==saved['config']['epochs']
        selection=f'final exact{saved["config"]["epochs"]}epochs,not selected on test'
    else:
        assert saved['step']==saved['config']['requested_updates']
        selection=f'final exact{saved["step"]}updates,not selected on test'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    model=policy_class(initialize=False).cuda();model.load_state_dict(saved['model'],strict=True);model.eval()
    rows=[json.loads(s) for s in (root/'test_manifest.jsonl').read_text().splitlines()]
    assert len(rows)==expected_cases
    config=dict(status='running',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        checkpoint_step=saved['step'],training=saved['config'],test_manifest_sha256=sha(root/'test_manifest.jsonl'),
        cases=len(rows),H=25,K=k,flow_steps=10,noise_seed=0,frame_budget=300,policy_hz=25,servo_hz=50,physics_hz=500,
        inputs='native front RGB + dense depth + calibration + targetID + measured joints; no target mask' if hasattr(model,'geometry_patch')
            else 'native front RGB + targetID + measured joints; same letterbox/frozen ViT as train',
        success='correct target lift>4cm,two jaw contacts,10consecutive frames; any wrong-object lift fails',
        checkpoint_selection=selection,video=False,
        source_sha256=sha(Path(__file__)))
    write(out/'config.json',config);del saved
    physics=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
    visual=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml')
    if hasattr(model,'geometry_patch'):
        from OATFlow.data_tools.render_dynamic import clip_planes
        clip_planes(visual)
    visual.vis.global_.offwidth=320;visual.vis.global_.offheight=240
    option=mujoco.MjvOption();option.geomgroup[3]=0
    results=[]
    if resume and (out/'results.jsonl').exists():
        results=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    completed={r['path'] for r in results};assert len(completed)==len(results)
    remaining=[r for r in rows if r['path'] not in completed]
    started=time.perf_counter()
    with mujoco.Renderer(visual,height=240,width=320) as renderer:
        for offset in tqdm(range(0,len(remaining),32),desc=f'{root.name} all{expected_cases}held-out pick episodes',mininterval=5):
            current=rollout(remaining[offset:offset+32],model,physics,visual,renderer,option,out,k=k)
            with (out/'results.jsonl').open('a') as f:
                for record in current:f.write(json.dumps(record)+'\n')
            results+=current
            progress=summarize(results,time.perf_counter()-started)
            progress['status']='running';write(out/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    assert len(results)==expected_cases
    summary=summarize(results,time.perf_counter()-started);summary['status']='complete'
    write(out/'summary.json',summary);write(out/'progress.json',summary)
    config['status']='complete';write(out/'config.json',config)
    with (out/'README.md').open('a') as f:f.write('\nCOMPLETE. Exact summary:\n'+json.dumps(summary)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    main(parser.parse_args().resume)
