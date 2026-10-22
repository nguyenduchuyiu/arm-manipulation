"""Feed real object depth clouds directly through the unchanged E118 reach policy."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import av
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import Scene,NAMES,BASE
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.reach.pointcloud import PointCloudReach
from OATFlow.simulation.contacts import jaw_contacts
from OATFlow.simulation.control import step as servo_step
from OATFlow.simulation.pose import PoseIK,rotation_matrix
from OATFlow.simulation.render import visual_model,add_marker,path_overlay
from OATFlow.top_pick.cloud import DATA,SOURCE
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.cloud_video import font,project
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.xyz_map import backproject

OUT=SOURCE.parent.parent/'test_pick_k10'
MANIFEST=OUT.parent.parent/'e120_fusion/test_manifest.jsonl'


def target_cloud(row):
    directory=DATA/row['path']
    with av.open(str(directory/'depth_front.mkv')) as video:
        codes=next(video.decode(video=0)).to_ndarray(format='gray16le')
    with np.load(directory/'masks.npz') as z:mask=z['instance'][0,0]==row['target_id']+1
    with np.load(directory/'calibration.npz') as z:
        K=z['intrinsics'][0].astype(np.float32);T=z['camera_to_base'][0,0].astype(np.float32)
        assert np.isclose(z['depth_unit_m'],.002)
    mask=binary_erosion(mask,iterations=1)&(codes>0)
    median=np.median(codes[mask])*.002
    mask&=np.abs(codes.astype(np.float64)*.002-median)<=.060
    assert mask.sum()>=8
    xyz,_=backproject(torch.tensor(codes.astype(np.float32))[None],torch.tensor(K)[None],torch.tensor(T)[None])
    return xyz[0].permute(1,2,0).numpy()[mask]


@torch.no_grad()
def rollout(policy,rows,clouds):
    physics=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));cases=[]
    for row in rows:
        scene=Scene(physics);state=scene.data
        with np.load(DATA/row['path']/'sim_state.npz') as z:
            for key in ('qpos','qvel','ctrl'):getattr(state,key)[:]=z['initial_'+key]
        mujoco.mj_forward(physics,state)
        initial=state.xpos[physics.body(NAMES[row['target_id']]).id].copy()
        cases.append(dict(row=row,scene=scene,data=state,held=state.ctrl.copy(),target=initial,
            ik=PoseIK((physics,scene.qadr,scene.dofs[:5],scene.site)),
            initial_qpos=state.qpos.copy(),initial_qvel=state.qvel.copy(),initial_ctrl=state.ctrl.copy(),
            initial_objects=np.stack([state.xpos[physics.body(n).id] for n in NAMES[:4]]),
            qpos=[],commands=[],ee=[],objects=[],contacts=[],speed=[],robot_table_penetration=[]))
    maximum=max(map(len,clouds))
    padded=np.stack([np.concatenate((c,np.repeat(c[:1],maximum-len(c),axis=0))) for c in clouds])
    inputs=torch.tensor(padded,device='cuda')
    joints=torch.tensor(np.stack([c['data'].qpos[c['scene'].qadr] for c in cases]),dtype=torch.float32,device='cuda')
    tf32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    batched=policy.conditioning(inputs,joints)
    for i,cloud in enumerate(clouds):
        single=policy.conditioning(torch.tensor(cloud,device='cuda')[None],joints[i:i+1])
        for batch_kv,single_kv in zip(batched,single):
            for a,b in zip(batch_kv,single_kv):torch.testing.assert_close(a[i:i+1],b,rtol=2e-5,atol=2e-5)
    torch.backends.cuda.matmul.allow_tf32=tf32
    write(OUT/'input_verification.json',dict(status='passed',points_per_object=list(map(len,clouds)),
        batch_points=maximum,duplicate_padding_preserves_context=True,all_input_points_retained=True,
        policy='strict original E118 PointCloudReach;no RGB/decoder/aux head/E120 weights',source_step=120000))
    noise_rng=torch.Generator(device='cuda').manual_seed(0)
    table=physics.geom('table_top').id;object_bodies={physics.body(n).id for n in NAMES}
    for frame in tqdm(range(300),desc='E118 direct object cloud,8episodes H25/K10',mininterval=5):
        if frame%10==0:
            joint=torch.tensor(np.stack([c['data'].qpos[c['scene'].qadr] for c in cases]),dtype=torch.float32,device='cuda')
            noise=policy.flow.sample_noise(len(cases),'cuda',noise_rng)
            with torch.autocast('cuda',dtype=torch.bfloat16):plans=policy.sample(inputs,joint,noise,steps=10).float().cpu().numpy()
            for case in cases:
                case['anchor']=case['data'].site_xpos[case['scene'].site].copy()
                case['rotation']=case['data'].site_xmat[case['scene'].site].reshape(3,3).copy()
        for i,case in enumerate(cases):
            state=case['data'];scene=case['scene'];action=plans[i,frame%10]
            assert np.isfinite(action).all()
            command=case['ik'].command(case,case['anchor']+action[:3],action[9],rotation_matrix(action[3:9])@case['rotation'])
            command=np.clip(command,*physics.actuator_ctrlrange.T)
            previous=state.site_xpos[scene.site].copy()
            servo_step(physics,state,case['held'],command);case['held']=command.copy()
            assert np.isfinite(state.qpos).all()
            case['qpos'].append(state.qpos.copy());case['commands'].append(command.copy())
            case['ee'].append(state.site_xpos[scene.site].copy())
            case['speed'].append(float(np.linalg.norm(state.site_xpos[scene.site]-previous)*25))
            case['objects'].append(np.stack([state.xpos[physics.body(n).id] for n in NAMES[:4]]))
            case['contacts'].append([len(jaw_contacts(physics,state,n)) for n in NAMES[:4]])
            penetration=0.
            for contact in state.contact:
                g1,g2=int(contact.geom1),int(contact.geom2)
                other=g2 if g1==table else (g1 if g2==table else -1)
                if other>=0 and physics.geom_bodyid[other]!=0 and physics.geom_bodyid[other] not in object_bodies:
                    penetration=max(penetration,max(0.,-float(contact.dist)))
            case['robot_table_penetration'].append(penetration)
    results=[]
    for case,cloud in zip(cases,clouds):
        row=case['row'];target=row['target_id'];ee=np.array(case['ee']);objects=np.array(case['objects']);contacts=np.array(case['contacts'])
        errors=np.linalg.norm(ee-case['target'],axis=1)*1000
        gains=objects[:,:,2]-case['initial_objects'][None,:,2];bilateral=contacts[:,target]==2
        streak=longest=0
        for good in (gains[:,target]>.04)&bilateral:
            streak=streak+1 if good else 0;longest=max(longest,streak)
        wrong_lift=bool(np.any(np.delete(gains,target,axis=1)>.04))
        record=dict(path=row['path'],layout=row['layout_id'],group=row['evaluation_group'],target=NAMES[target],slot=row['target_slot'],
            frames=300,input_points=len(cloud),final_EE_to_initial_target_mm=float(errors[-1]),minimum_EE_to_initial_target_mm=float(errors.min()),
            final_EE_to_current_target_mm=float(np.linalg.norm(ee[-1]-objects[-1,target])*1000),
            position_only_final_within_10mm=bool(errors[-1]<10),position_only_last25_within_10mm=bool((errors[-25:]<10).all()),
            final_jaw_m=float(case['qpos'][-1][case['scene'].qadr[-1]]),bilateral_contact_frames=int(bilateral.sum()),
            max_target_lift_mm=float(gains[:,target].max()*1000),bilateral_lift_streak=longest,
            pick_success=longest>=10 and not wrong_lift,wrong_object_lift=wrong_lift,
            robot_table_penetration_frames=int((np.array(case['robot_table_penetration'])>.0002).sum()),
            max_ik_error_mm=float(max(case['ik_errors'])*1000))
        filename=row['path'].replace('/','_')
        np.savez_compressed(OUT/(filename+'.npz'),qpos=np.array(case['qpos'],np.float32),commands=np.array(case['commands'],np.float32),
            initial_qpos=case['initial_qpos'],initial_qvel=case['initial_qvel'],initial_ctrl=case['initial_ctrl'],
            objects=objects.astype(np.float32),contacts=contacts.astype(np.uint8),ee=ee.astype(np.float32),
            input_cloud_base_m=cloud,goal_center_world_m=case['target'])
        results.append(record)
    return results


def render(record,model):
    name=record['path'].replace('/','_');trace=OUT/(name+'.npz');trace_hash=sha(trace)
    with np.load(trace) as z:
        qpos=np.vstack((z['initial_qpos'][None],z['qpos']));cloud=z['input_cloud_base_m'];goal=z['goal_center_world_m'];contacts=z['contacts']
    option=mujoco.MjvOption();option.geomgroup[3]=0;state=mujoco.MjData(model)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.54,-.20,.14];camera.distance=1.04;camera.azimuth=135;camera.elevation=-32
    body=model.body(record['target']).id;site=model.site('grasp_site').id;target=NAMES.index(record['target'])
    writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1344x760','-r','12.5','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast',
        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/(name+'.mp4'))],stdin=subprocess.PIPE)
    ee=[];indices=range(0,len(qpos),2)
    with mujoco.Renderer(model,height=240,width=320) as front,mujoco.Renderer(model,height=480,width=640,max_geom=2000) as overview:
        try:
            for frame in tqdm(indices,desc=f'Render direct cloud {name}',mininterval=5,leave=False):
                state.qpos[:]=qpos[frame];mujoco.mj_forward(model,state);ee.append(state.site_xpos[site].copy())
                front.update_scene(state,camera='cam_front',scene_option=option)
                left=Image.fromarray(front.render()).resize((640,480),Image.Resampling.BICUBIC);draw=ImageDraw.Draw(left)
                uv,positive=project(cloud+BASE,state,model)
                for x,y in uv[positive]*2:
                    if 0<=x<640 and 0<=y<480:draw.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill='#0cc4de')
                overview.update_scene(state,camera=camera,scene_option=option)
                for p in cloud:add_marker(overview.scene,p+BASE,[.02,.78,.9,.9],radius=.0011)
                add_marker(overview.scene,goal,[1.,.55,.1,1.],radius=.002)
                path_overlay(overview.scene,ee,(40,205,130),.0013)
                image=Image.new('RGB',(1344,760),'white');image.paste(left,(20,112));image.paste(Image.fromarray(overview.render()),(684,112));draw=ImageDraw.Draw(image)
                draw.text((20,15),f'E118 original reach | direct object cloud -> point encoder -> FM | {record["target"]}',font=font(23,True),fill='#17324d')
                draw.text((20,55),f'{record["path"]} | {record["group"]} | source120,000 | H25/K10 | all{len(cloud)}points retained',font=font(18),fill='#607587')
                draw.text((20,88),'FRONT RGB / CYAN: ACTUAL FIXED GOAL CLOUD',font=font(16,True),fill='#17324d')
                draw.text((684,88),'OBLIQUE REPLAY / ORIGINAL REACH POLICY',font=font(16,True),fill='#17324d')
                error=float(np.linalg.norm(state.site_xpos[site]-goal)*1000);current=float(np.linalg.norm(state.site_xpos[site]-state.xpos[body])*1000)
                draw.text((20,613),f't={frame/25:.2f}s | EE to initial target={error:.1f} mm | EE to current object={current:.1f} mm',font=font(19,True),fill='#17324d')
                draw.text((20,650),f'Jaw contacts={int(contacts[frame-1,target]) if frame else 0}/2 | max lift={record["max_target_lift_mm"]:.1f} mm | no scripted grasp/lift',font=font(18),fill='#607587')
                draw.text((20,687),'Input: front depth + GT object mask + calibration -> base-frame XYZ; held fixed from frame0.',font=font(17),fill='#17324d')
                draw.text((20,724),'Cyan: actual FM input cloud | Orange: initial target center (diagnostic only) | Green: measured EE path',font=font(16),fill='#607587')
                if frame==0:image.save(OUT/(name+'_start.png'))
                if frame==300:image.save(OUT/(name+'_poster.png'))
                writer.stdin.write(np.asarray(image).tobytes())
        finally:
            writer.stdin.close();assert writer.wait()==0
    with av.open(str(OUT/(name+'.mp4'))) as video:decoded=sum(1 for _ in video.decode(video=0))
    assert decoded==151 and sha(trace)==trace_hash
    return dict(video=name+'.mp4',decoded_frames=decoded,trace_sha256=trace_hash,trace_unchanged=True)


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=False);shutil.copyfile('docs/research/e118_object_README.md',OUT/'README.md')
    os.link(SOURCE,OUT/'checkpoint.pt');saved=torch.load(OUT/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==120000
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    policy=PointCloudReach().cuda();policy.load_state_dict(saved['model'],strict=True);policy.eval();policy.requires_grad_(False);del saved
    rows=[json.loads(s) for s in MANIFEST.read_text().splitlines()]
    selected=[r for r in rows if (r['layout_id'],r['permutation']) in ((0,1),(1,0))]
    assert len(selected)==8 and [r['target_id'] for r in selected]==[0,1,2,3]*2
    config=dict(status='running',checkpoint_source=str(SOURCE),checkpoint_step=120000,checkpoint_sha256=sha(SOURCE),
        source_architecture='unchanged PointCloudReach:global XYZ normalize->point MLP->maxpool->goal projection960->FM with measured joint/state token',
        weights='strict entire E118best;no E120 tensors;no finetuning',selection='predefined two scenes g000/p01 and g001/p00,all four target colors',
        selected=selected,H=25,K=10,flow_steps=10,noise_seed=0,frame_budget=300,policy_hz=25,servo_hz=50,physics_hz=500,
        input='all eroded frame0 target depth-mask points in base frame;static goal;GT mask selects object',
        sampling='all points;duplicate pad only for variable-size batching;maxpool invariant',
        comparison_limit='32mm actual cubes/table versus20mm isolated voxel during reach training;new front view;no lift skill',
        metrics='position-only reach errors plus bilateral contact/lift;not original air-reach collision-aware success',
        video_fps=12.5,source_sha256=sha(Path(__file__)))
    write(OUT/'config.json',config)
    clouds=[target_cloud(row) for row in tqdm(selected,desc='Decode eight target clouds',mininterval=5)]
    started=time.perf_counter();results=rollout(policy,selected,clouds)
    summary=dict(cases=8,final_position_within10mm=sum(r['position_only_final_within_10mm'] for r in results),
        median_final_EE_error_mm=float(np.median([r['final_EE_to_initial_target_mm'] for r in results])),
        any_bilateral_contact_cases=sum(r['bilateral_contact_frames']>0 for r in results),
        pick_successes=sum(r['pick_success'] for r in results),rollout_elapsed_s=time.perf_counter()-started)
    write(OUT/'results.json',dict(summary=summary,records=results));print(json.dumps(summary),flush=True)
    del policy;torch.cuda.empty_cache()
    visual=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml');clip_planes(visual)
    visual.vis.global_.offwidth=640;visual.vis.global_.offheight=480
    checks=[render(record,visual) for record in tqdm(results,desc='Render all8direct-cloud episodes',mininterval=5)]
    write(OUT/'video_verification.json',dict(status='passed',clips=checks,source_checkpoint_unchanged=sha(SOURCE)==config['checkpoint_sha256']))
    config.update(status='complete',summary=summary);write(OUT/'config.json',config)
    with (OUT/'README.md').open('a') as f:f.write('\nComplete. Results:\n'+json.dumps(summary)+'\n')
    print(json.dumps(dict(status='complete',output=str(OUT),summary=summary)),flush=True)


if __name__=='__main__':main()
