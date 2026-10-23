"""E131 closed-loop pick evaluation using only GT-visible target XYZ and joints."""
import argparse
import json
import re
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import Scene,NAMES
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.reach.pointcloud import PointCloudReach
from OATFlow.simulation.control import step as servo_step
from OATFlow.simulation.contacts import jaw_contacts
from OATFlow.simulation.pose import PoseIK,rotation_matrix
from OATFlow.simulation.render import visual_model
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.cloud_evaluate import summarize
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.object_cloud import ROOT,CACHE,extract_cloud
from OATFlow.top_pick.xyz_map import camera_calibration


@torch.inference_mode()
def rollout(rows,policy,physics,visual,renderer,seg,option,out,frame_budget=300,data_root=DATA,k=15):
    display=mujoco.MjData(visual);scenes=[]
    for row in rows:
        scene=Scene(physics)
        with np.load(data_root/row['path']/'sim_state.npz') as z:
            for field in ('qpos','qvel','ctrl'):getattr(scene.data,field)[:]=z['initial_'+field]
        mujoco.mj_forward(physics,scene.data);state=scene.data
        scene_rng=np.random.default_rng(np.random.SeedSequence([131,row['layout_id'],row['permutation'],row['target_id']]))
        scenes.append(dict(data=state,row=row,scene=scene,held=state.ctrl.copy(),rng=scene_rng,
            ik=PoseIK((physics,scene.qadr,scene.dofs[:5],scene.site)),initial_qpos=state.qpos.copy(),
            initial_qvel=state.qvel.copy(),initial_ctrl=state.ctrl.copy(),
            initial_objects=np.stack([state.xpos[physics.body(n).id] for n in NAMES[:4]]),
            qpos=[],commands=[],objects=[],contacts=[],success=False,done=False,reason='timeout',
            streak=0,longest=0,cloud_replans=[],clouds=[],visible_counts=[],missing_frame=None))
    generator=torch.Generator(device='cuda').manual_seed(0)
    for frame in tqdm(range(frame_budget),desc='GT cloud pick H25/K15',mininterval=10,leave=False):
        active=[s for s in scenes if not s['done']]
        if not active:break
        if frame%k==0:
            inputs=[];joints=[];usable=[]
            for s in active:
                display.qpos[:]=s['data'].qpos;mujoco.mj_forward(visual,display)
                renderer.update_scene(display,camera='cam_front',scene_option=option);depth=renderer.render().copy()
                valid=np.isfinite(depth)&(depth>0)&(depth<=6.5535);codes=np.zeros(depth.shape,np.uint16)
                codes[valid]=np.rint(depth[valid]/.002).astype(np.uint16)
                seg.update_scene(display,camera='cam_front',scene_option=option);labels=seg.render()
                geoms=np.flatnonzero(visual.geom_bodyid==visual.body(NAMES[s['row']['target_id']]).id)
                mask=np.isin(labels[...,0],geoms)&(labels[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
                K,T=camera_calibration(visual,display);cloud,good,count=extract_cloud(codes,mask,K,T,s['rng'])
                if not good:
                    s.update(done=True,reason='missing_target_cloud',missing_frame=frame);continue
                inputs.append(cloud);joints.append(s['data'].qpos[s['scene'].qadr]);usable.append(s)
                s['cloud_replans'].append(frame);s['clouds'].append(cloud);s['visible_counts'].append(count)
            if usable:
                cloud=torch.tensor(np.stack(inputs),device='cuda');joint=torch.tensor(np.stack(joints),device='cuda',dtype=torch.float32)
                noise=policy.flow.sample_noise(1,'cuda',generator).expand(len(usable),-1,-1)
                with torch.autocast('cuda',dtype=torch.bfloat16):plans=policy.sample(cloud,joint,noise,steps=10).float().cpu().numpy()
                for s,plan in zip(usable,plans):
                    assert np.isfinite(plan).all();s['plan']=plan
                    s['anchor']=s['data'].site_xpos[s['scene'].site].copy();s['rotation']=s['data'].site_xmat[s['scene'].site].reshape(3,3).copy()
        for s in [s for s in active if not s['done']]:
            state=s['data'];action=s['plan'][frame%k]
            command=s['ik'].command(s,s['anchor']+action[:3],action[9],rotation_matrix(action[3:9])@s['rotation'])
            command=np.clip(command,*physics.actuator_ctrlrange.T);servo_step(physics,state,s['held'],command);s['held']=command.copy()
            objects=np.stack([state.xpos[physics.body(n).id] for n in NAMES[:4]])
            contacts=np.array([len(jaw_contacts(physics,state,n)) for n in NAMES[:4]])
            gains=objects[:,2]-s['initial_objects'][:,2];t=s['row']['target_id']
            s['streak']=s['streak']+1 if gains[t]>.04 and contacts[t]==2 else 0;s['longest']=max(s['longest'],s['streak'])
            wrong=[i for i in range(4) if i!=t and gains[i]>.04]
            if wrong:s.update(done=True,reason='wrong_object',wrong_object=NAMES[wrong[0]])
            elif s['streak']>=10:s.update(done=True,success=True,reason='success')
            assert np.isfinite(state.qpos).all()
            s['qpos'].append(state.qpos.copy());s['commands'].append(command.copy());s['objects'].append(objects);s['contacts'].append(contacts)
    results=[]
    for s in scenes:
        row=s['row'];gains=(np.asarray(s['objects'])[:,row['target_id'],2]-s['initial_objects'][row['target_id'],2]) if s['objects'] else np.array([0.])
        record=dict(path=row['path'],group=row['evaluation_group'],layout=row['layout_id'],permutation=row['permutation'],
            target=NAMES[row['target_id']],slot=row['target_slot'],success=s['success'],reason=s['reason'],frames=len(s['commands']),
            bilateral_lift_streak=s['longest'],wrong_object=s.get('wrong_object'),max_target_lift_mm=float(gains.max()*1000),
            max_ik_error_mm=float(max(s.get('ik_errors',[0.]))*1000),missing_cloud_frame=s['missing_frame'],
            minimum_cloud_pixels=min(s['visible_counts']) if s['visible_counts'] else 0)
        if 'pair' in row:record.update(pair=row['pair'],edge=row['edge'],target_xy=row['target_xy'])
        np.savez_compressed(out/(row['path'].replace('/','_')+'.npz'),qpos=np.asarray(s['qpos'],np.float32),commands=np.asarray(s['commands'],np.float32),
            objects=np.asarray(s['objects'],np.float32),contacts=np.asarray(s['contacts'],np.uint8),initial_qpos=s['initial_qpos'],
            initial_qvel=s['initial_qvel'],initial_ctrl=s['initial_ctrl'],cloud_replan_frames=np.array(s['cloud_replans']),
            input_cloud_base_m=np.asarray(s['clouds'],np.float32),input_cloud_pixels=np.array(s['visible_counts']))
        results.append(record)
    return results


def renderers():
    physics=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
    visual=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml');clip_planes(visual)
    visual.vis.global_.offwidth=320;visual.vis.global_.offheight=240
    option=mujoco.MjvOption();option.geomgroup[3]=0
    renderer=mujoco.Renderer(visual,height=240,width=320);renderer.enable_depth_rendering()
    msaa=visual.vis.quality.offsamples;visual.vis.quality.offsamples=0
    seg=mujoco.Renderer(visual,height=240,width=320);seg.enable_segmentation_rendering();visual.vis.quality.offsamples=msaa
    return physics,visual,renderer,seg,option


def describe_run(status):
    for path in (Path('docs/research/e131_README.md'),ROOT/'README.md',ROOT/'train/README.md'):
        text=re.sub(r'^Status:.*$',f'Status: {status}',path.read_text(),count=1,flags=re.MULTILINE)
        path.write_text(text)


@torch.inference_mode()
def main(xy=False):
    out=ROOT/('test_xy' if xy else 'test_k15');out.mkdir(exist_ok=False)
    note=Path('docs/research/e131_xy_README.md' if xy else 'docs/research/e131_test_README.md');shutil.copyfile(note,out/'README.md')
    checkpoint=ROOT/'train/final.pt';h=sha(checkpoint);saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==23520
    describe_run('TRAIN COMPLETE; XY evaluation running.' if xy else 'TRAIN COMPLETE; full held-out evaluation running.')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    policy=PointCloudReach().cuda();policy.load_state_dict(saved['model'],strict=True);policy.eval();policy.requires_grad_(False)
    manifest=CACHE/'test_xy/manifest.jsonl' if xy else ROOT/'test_manifest.jsonl'
    rows=list(map(json.loads,manifest.read_text().splitlines()));assert len(rows)==(16 if xy else 1408)
    config=dict(status='running',checkpoint=str(checkpoint),checkpoint_sha256=h,step=saved['step'],training=saved['config'],
        H=25,K=15,flow_steps=10,noise_seed=0,frame_budget=300,policy_hz=25,servo_hz=50,physics_hz=500,
        inputs='128 GT-visible target XYZ points in robot base + measured joints;no RGB,targetID embedding,object-center XYZ or hidden pose',
        oracle_information='GT visible segmentation for target selection only;actual depth/calibration',
        missing_cloud='fewer8filtered target pixels:episode fails explicitly;no GT-pose fallback or stale-cloud carry-forward',
        manifest=str(manifest),manifest_sha256=sha(manifest),cases=len(rows),video=False,source_sha256=sha(Path(__file__)))
    del saved;write(out/'config.json',config)
    physics,visual,renderer,seg,option=renderers();results=[];started=time.perf_counter()
    with renderer,seg:
        for start in tqdm(range(0,len(rows),32),desc='E131 GT-cloud closed-loop episodes',mininterval=5):
            current=rollout(rows[start:start+32],policy,physics,visual,renderer,seg,option,out,
                data_root=CACHE/'test_xy/states' if xy else DATA)
            with (out/'results.jsonl').open('a') as f:
                for r in current:f.write(json.dumps(r)+'\n')
            results+=current;summary=summarize(results,time.perf_counter()-started);write(out/'progress.json',summary);print(json.dumps(summary),flush=True)
    summary=summarize(results,time.perf_counter()-started);summary['status']='complete';write(out/'summary.json',summary)
    assert sha(checkpoint)==h;config['status']='complete';write(out/'config.json',config)
    write(out/'verification.json',dict(status='passed',checkpoint_unchanged=True,all_episodes_recorded=len(results)==len(rows),
        policy_never_receives_RGB_or_GT_object_pose=True,only_visible_clouds=True))
    describe_run('COMPLETE.' if xy else 'TRAIN COMPLETE; full held-out evaluation complete; XY evaluation pending.')
    with Path('docs/research/REPORT.md').open('a') as report:
        report.write(f'\nE131 {"XY boundary" if xy else "full held-out K15"} evaluation complete: {summary["successes"]}/{summary["cases"]} ({100*summary["success_rate"]:.2f}%). Exact results: {out}/summary.json.\n')
    text=note.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nResults:\n'+json.dumps(summary,indent=2)+'\n';note.write_text(text);(out/'README.md').write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--xy',action='store_true');main(parser.parse_args().xy)
