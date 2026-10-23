"""Replay the single E131 non-target-lift failure with its exact input clouds."""
import json
from pathlib import Path
import shutil
import subprocess

import av
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import BASE,NAMES
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.simulation.render import visual_model,path_overlay
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.object_cloud import ROOT
from OATFlow.top_pick.xyz_map import camera_calibration
from OATFlow.tracking.mask_video import font

TEST=ROOT/'test_k15'
OUT=TEST/'wrong_1'
NOTE=Path('docs/research/e131_wrong_README.md')


def project(points,K,T):
    local=(points-T[:3,3])@T[:3,:3];pixels=local@K.T
    return pixels[:,:2]/pixels[:,2:],local[:,2]>0


def surface_distance(points,center,rotation,half):
    q=np.abs((points-center)@rotation)-half
    return np.abs(np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0))


def main():
    OUT.mkdir(exist_ok=False);shutil.copyfile(NOTE,OUT/'README.md')
    records=list(map(json.loads,(TEST/'results.jsonl').read_text().splitlines()))
    selected=[r for r in records if r['reason']=='wrong_object'];assert len(selected)==1;record=selected[0]
    protocol=json.loads((TEST/'config.json').read_text());assert protocol['step']==23520 and protocol['K']==15
    trace=TEST/(record['path'].replace('/','_')+'.npz')
    inputs=[trace,TEST/'results.jsonl',Path(protocol['checkpoint'])]
    hashes={str(p):sha(p) for p in inputs};assert hashes[protocol['checkpoint']]==protocol['checkpoint_sha256']
    with np.load(trace) as z:
        states=np.concatenate([z['initial_qpos'][None],z['qpos']],0);objects=z['objects'].copy();contacts=z['contacts'].copy()
        cloud_frames=z['cloud_replan_frames'].copy();clouds=z['input_cloud_base_m'].copy();counts=z['input_cloud_pixels'].copy()
    assert len(states)==record['frames']+1==144
    np.testing.assert_array_equal(cloud_frames,np.arange(0,record['frames'],15))
    target=NAMES.index(record['target']);wrong=NAMES.index(record['wrong_object'])
    model=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml');clip_planes(model)
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    state=mujoco.MjData(model);site=model.site('grasp_site').id
    bodies=[model.body(name).id for name in NAMES[:4]];half=model.geom(record['target']+'_visual').size.copy()
    option=mujoco.MjvOption();option.geomgroup[3]=0
    camera=mujoco.MjvCamera();camera.lookat[:]=[.57,-.16,.11];camera.distance=1.0;camera.azimuth=-60;camera.elevation=-38
    writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x830','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'wrong_object.mp4')],stdin=subprocess.PIPE)
    config=dict(status='rendering',case=record,checkpoint=protocol['checkpoint'],checkpoint_step=23520,training=protocol['training'],
        source_hashes=hashes,source_sha256=sha(Path(__file__)),H=25,K=15,flow_steps=10,noise_seed=0,frame_budget=300,
        replay_frames=len(states),source_state_hz=25,video_fps=10,speed=.4,resolution=[1280,830],
        cloud='exact recorded128point GT-visible target input at last K15replan;not a prediction',
        new_physics_steps=False,model_inference=False,action_resampling=False)
    write(OUT/'config.json',config)
    metrics=[];path=[];images=[];pose_error=0.;cloud_errors=[];cloud_wrong_distances=[]
    first_contact=int(np.flatnonzero(contacts[:,wrong]>0)[0])+1
    sample_frames={0:first_contact*0,first_contact:first_contact,len(states)-1:len(states)-1}
    with mujoco.Renderer(model,height=240,width=320) as front,mujoco.Renderer(model,height=480,width=640,max_geom=2000) as overview:
        for frame in tqdm(range(len(states)),desc='Replay non-target-lift case',mininterval=5):
            state.qpos[:]=states[frame];mujoco.mj_forward(model,state)
            centers=state.xpos[bodies].copy();ee=state.site_xpos[site].copy();path.append(ee)
            if frame==0:initial=centers.copy()
            else:pose_error=max(pose_error,float(abs(centers-objects[frame-1]).max()))
            gain=(centers[:,2]-initial[:,2])*1000
            ci=int(np.searchsorted(cloud_frames,frame,side='right')-1);cloud=clouds[ci];K,T=camera_calibration(model,state)
            if frame in cloud_frames:
                distances=surface_distance(cloud.astype(float),centers[target]-BASE,state.xmat[bodies[target]].reshape(3,3),half)
                cloud_errors.append(float(distances.max()));cloud_wrong_distances.append(float(np.linalg.norm(cloud.mean(0)+BASE-centers[wrong])*1000))
                assert distances.max()<.004
            front.update_scene(state,camera='cam_front',scene_option=option);left=Image.fromarray(front.render()).resize((640,480),Image.Resampling.BICUBIC)
            draw=ImageDraw.Draw(left);uv,valid=project(cloud,K,T)
            for (u,v),good in zip(uv,valid):
                if good and 0<=u<320 and 0<=v<240:
                    x,y=u*2,v*2;draw.ellipse((x-1.4,y-1.4,x+1.4,y+1.4),fill='#00e4ff')
            pixels,_=project(centers[[target,wrong]]-BASE,K,T)
            for j,(label,color,offset) in enumerate((('TARGET: YELLOW','#ffe45b',(-155,30)),('NON-TARGET: BLUE','#ff63c3',(12,-35)))):
                x,y=pixels[j]*2;draw.ellipse((x-18,y-18,x+18,y+18),outline=color,width=2)
                draw.text((x+offset[0],y+offset[1]),label,font=font(16,True),fill=color,stroke_width=1,stroke_fill='#15263a')
            overview.update_scene(state,camera=camera,scene_option=option);path_overlay(overview.scene,path[::2],(30,210,160),.001)
            right=Image.fromarray(overview.render())
            image=Image.new('RGB',(1280,830),'#f4f6f9');draw=ImageDraw.Draw(image)
            draw.text((14,9),f'E131 | target Yellow / non-target Blue | {record["path"]} | replay0.4x',font=font(25,True),fill='#21354a')
            draw.text((14,47),f'Original frame {frame:03d} /143 | t={frame/25:.2f}s | source25Hz,video10FPS | no new rollout',font=font(18),fill='#52647b')
            image.paste(left,(0,104));image.paste(right,(640,104));draw=ImageDraw.Draw(image)
            draw.text((14,79),'FRONT RGB + SAVED TARGET INPUT CLOUD',font=font(18,True),fill='#21354a')
            draw.text((654,79),'OBLIQUE PHYSICAL REPLAY + EE PATH',font=font(18,True),fill='#21354a')
            c=contacts[frame-1] if frame else np.zeros(4,int)
            text=[f'Yellow target lift: {gain[target]:.1f} mm | Blue non-target lift: {gain[wrong]:.1f} mm',
                f'Jaw contacts now: Yellow {c[target]}/2 | Blue {c[wrong]}/2 | Blue maximum in trace: {contacts[:,wrong].max()}/2',
                f'Cyan = GT-selected target input,128points | source frame {cloud_frames[ci]} | age {frame-cloud_frames[ci]} frames | {counts[ci]} filtered pixels',
                'Evaluator flags any non-target lift >40mm; bilateral grasp is NOT required for this failure label.',
                'Saved scene geometry/action trace replayed exactly. Input cloud is not re-predicted or re-sampled.']
            for j,line in enumerate(text):draw.text((14,607+j*40),line,font=font(19,True if j==0 else False),fill='#21354a' if j<4 else '#52647b')
            writer.stdin.write(np.asarray(image).tobytes())
            if frame in sample_frames:
                name='start' if frame==0 else 'last' if frame==len(states)-1 else 'contact'
                image.save(OUT/f'{name}.png');images.append(image.copy())
            metrics.append(dict(frame=frame,time_s=frame/25,target_lift_mm=float(gain[target]),non_target_lift_mm=float(gain[wrong]),
                target_jaw_contacts=int(c[target]),non_target_jaw_contacts=int(c[wrong]),input_cloud_frame=int(cloud_frames[ci]),cloud_age=frame-int(cloud_frames[ci])))
    writer.stdin.close();assert writer.wait()==0
    count=0
    with av.open(str(OUT/'wrong_object.mp4')) as video:
        assert float(video.streams.video[0].average_rate)==10
        for item in video.decode(video=0):count+=1
    assert count==len(states) and pose_error<1e-6
    assert hashes=={str(p):sha(p) for p in inputs}
    sheet=Image.new('RGB',(1280,415*len(images)),'white')
    for i,image in enumerate(images):sheet.paste(image.resize((640,415)),(320,i*415))
    sheet.save(OUT/'contact_sheet.png')
    summary=dict(status='complete',record=record,video_frames=count,video_seconds=count/10,physical_seconds=record['frames']/25,
        maximum_blue_jaw_contacts=int(contacts[:,wrong].max()),blue_bilateral_contact_frames=int((contacts[:,wrong]==2).sum()),
        final_blue_lift_mm=metrics[-1]['non_target_lift_mm'],final_blue_contacts=metrics[-1]['non_target_jaw_contacts'],
        input_cloud_max_distance_to_yellow_surface_mm=max(cloud_errors)*1000,
        minimum_input_centroid_distance_to_blue_center_mm=min(cloud_wrong_distances),
        interpretation='Non-target Blue lifted above40mm after contact;no bilateral Blue grasp recorded. Cloud selection remains Yellow.')
    write(OUT/'metrics.json',metrics);write(OUT/'summary.json',summary)
    write(OUT/'verification.json',dict(status='passed',video_fully_decoded=True,source_hashes_unchanged=True,
        replay_object_pose_max_error_m=pose_error,all_saved_clouds_on_target_surface=True,input_cloud_cadence_verified=True))
    config['status']='complete';write(OUT/'config.json',config)
    text=NOTE.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nResults:\n'+json.dumps(summary,indent=2)+'\n'
    NOTE.write_text(text);(OUT/'README.md').write_text(text);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
