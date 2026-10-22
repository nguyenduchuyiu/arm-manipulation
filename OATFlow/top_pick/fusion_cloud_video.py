"""Replay E120 with actual RGB-D geometry and predicted target-cloud diagnostics."""
import json
from pathlib import Path
import shutil
import subprocess

import av
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion,median_filter
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES,BASE
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.simulation.render import visual_model,path_overlay
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.cloud_video import font,project
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.fusion import ROOT,FusionPolicy
from OATFlow.top_pick.xyz_map import camera_calibration,backproject

ORANGE='#ed8b26';CYAN='#00a9c9';GREEN='#169d64'
# Fixed oblique basis; all clouds still originate from the one front depth camera.
FORWARD=np.array([-.75,.75,-.65]);FORWARD/=np.linalg.norm(FORWARD)
RIGHT=np.cross(FORWARD,[0.,0.,1.]);RIGHT/=np.linalg.norm(RIGHT)
UP=np.cross(RIGHT,FORWARD)


def coordinates(points,center,span,size):
    relative=points-center;scale=size[0]/span
    uv=np.c_[relative@RIGHT,-relative@UP]*scale+np.array(size)/2
    return uv,relative@FORWARD


def cloud_view(points,colors,truth,prediction,ee,center,span,size,zoom=False):
    image=np.full((size[1],size[0],3),246,np.uint8)
    uv,z=coordinates(points,center,span,size);uv=np.rint(uv).astype(int)
    good=(uv[:,0]>=0)&(uv[:,0]<size[0]-1)&(uv[:,1]>=0)&(uv[:,1]<size[1]-1)
    uv=uv[good];z=z[good];colors=colors[good]
    order=np.argsort(z);uv=uv[order];colors=colors[order]
    _,first=np.unique(uv[:,1]*size[0]+uv[:,0],return_index=True)
    uv=uv[first];colors=colors[first]
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        image[uv[:,1]+dy,uv[:,0]+dx]=colors
    image=Image.fromarray(image);draw=ImageDraw.Draw(image)
    for cloud,color,radius in ((truth,ORANGE,3 if zoom else 2),(prediction,CYAN,3 if zoom else 2)):
        pixels,_=coordinates(cloud,center,span,size)
        for x,y in pixels:
            if 0<=x<size[0] and 0<=y<size[1]:
                draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
    pixel,_=coordinates(ee[None],center,span,size);x,y=pixel[0]
    if 0<=x<size[0] and 0<=y<size[1]:
        draw.line((x-7,y,x+7,y),fill=GREEN,width=3);draw.line((x,y-7,x,y+7),fill=GREEN,width=3)
    if zoom:
        length=.05*size[0]/span
        draw.line((24,size[1]-25,24+length,size[1]-25),fill='#243746',width=3)
        draw.text((24,size[1]-49),'50 mm',font=font(15),fill='#243746')
    return image


def metric(prediction,truth,ee,target):
    result=dict(pred_centroid_to_EE_mm=float(np.linalg.norm(prediction.mean(0)-ee)*1000),
                EE_to_target_center_mm=float(np.linalg.norm(ee-target)*1000),
                target_visible_pixels=len(truth),cloud_label_valid=len(truth)>=8)
    if len(truth)>=8:
        distance=np.linalg.norm(prediction[:,None]-truth[None],axis=-1)
        result.update(surface_chamfer_mm=float(.5*(distance.min(1).mean()+distance.min(0).mean())*1000),
            surface_centroid_error_mm=float(np.linalg.norm(prediction.mean(0)-truth.mean(0))*1000))
    else:result.update(surface_chamfer_mm=None,surface_centroid_error_mm=None)
    return result


def render(record,test,out,model,policy,experiment='E120',training='final23520 /80epochs',physical_overview=False,clean_depth_labels=False,k=10):
    assert 1<=k<=25
    device=next(policy.parameters()).device
    trace=test/(record['path'].replace('/','_')+'.npz');trace_hash=sha(trace)
    with np.load(trace) as z:qpos=np.vstack((z['initial_qpos'][None],z['qpos']));contacts=z['contacts']
    name=('success_' if record['success'] else 'failure_')+record['group']
    target_id=NAMES.index(record['target']);body=model.body(record['target']).id
    state=mujoco.MjData(model);site=model.site('grasp_site').id
    option=mujoco.MjvOption();option.geomgroup[3]=0
    writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1344x1100','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','2',
        '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/(name+'.mp4'))],stdin=subprocess.PIPE)
    rows=[];replans=[];clouds=[];prediction=None;initial_height=None;ee_path=[]
    original_samples=model.vis.quality.offsamples
    front=mujoco.Renderer(model,height=240,width=320)
    model.vis.quality.offsamples=0;seg=mujoco.Renderer(model,height=240,width=320)
    model.vis.quality.offsamples=original_samples;seg.enable_segmentation_rendering()
    if physical_overview:
        overview_renderer=mujoco.Renderer(model,height=480,width=640,max_geom=1400)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.54,-.20,.14]
        camera.distance=1.04;camera.azimuth=135;camera.elevation=-32
    with front,seg:
        try:
            for frame in tqdm(range(len(qpos)),desc=f'Cloud {name}',mininterval=5,leave=False):
                state.qpos[:]=qpos[frame];mujoco.mj_forward(model,state)
                target=state.xpos[body]-BASE;ee=state.site_xpos[site]-BASE
                if initial_height is None:initial_height=target[2]
                front.update_scene(state,camera='cam_front',scene_option=option);rgb=front.render().copy()
                front.enable_depth_rendering();depth=front.render();front.disable_depth_rendering()
                valid=np.isfinite(depth)&(depth>0)&(depth<=6.5535)
                codes=np.zeros(depth.shape,np.uint16);codes[valid]=np.rint(depth[valid]/.002).astype(np.uint16)
                K,T=camera_calibration(model,state)
                scene={k:torch.as_tensor(v,device=device,dtype=torch.float32)[None] for k,v in
                       zip(('depth','intrinsics','camera_to_base'),(codes,K,T))}
                xyz,valid_tensor=backproject(**scene);xyz=xyz[0].permute(1,2,0).cpu().numpy()
                valid=codes>0
                if frame%k==0 and frame<record['frames']:
                    pixels=torch.as_tensor(rgb,device=device).permute(2,0,1)[None]
                    query=torch.tensor([target_id],device=device)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        prediction=policy.predict_cloud(policy.encode(pixels),query,scene)[0].float().cpu().numpy()
                    assert np.isfinite(prediction).all();replans.append(frame);clouds.append(prediction.copy())
                seg.update_scene(state,camera='cam_front',scene_option=option);labels=seg.render()
                geometry=(labels[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
                ids=labels[...,0].clip(0,model.ngeom-1)
                mask=binary_erosion(geometry&(model.geom_bodyid[ids]==body),iterations=1)&valid
                if mask.any():
                    median=np.median(codes[mask])*.002
                    mask&=np.abs(codes.astype(np.float64)*.002-median)<=.060
                    if clean_depth_labels:
                        smooth=median_filter(codes,size=3,mode='nearest')
                        mask&=np.abs(codes.astype(np.int32)-smooth.astype(np.int32))*.002<=.010
                truth=xyz[mask];m=metric(prediction,truth,ee,target)
                m.update(inference_frame=replans[-1],prediction_age_frames=frame-replans[-1])
                rows.append(dict(frame=frame,**m))
                left=Image.fromarray(rgb).resize((640,480),Image.Resampling.BICUBIC);draw=ImageDraw.Draw(left)
                for points,color in ((truth,ORANGE),(prediction,CYAN)):
                    uv,positive=project(points+BASE,state,model)
                    for x,y in uv[positive]*2:
                        if 0<=x<640 and 0<=y<480:draw.ellipse((x-1.4,y-1.4,x+1.4,y+1.4),fill=color)
                points=xyz[valid];colors=rgb[valid]
                if physical_overview:
                    ee_path.append(state.site_xpos[site].copy())
                    overview_renderer.update_scene(state,camera=camera,scene_option=option)
                    path_overlay(overview_renderer.scene,ee_path[::2],(40,205,130),.0013)
                    overview=Image.fromarray(overview_renderer.render())
                else:
                    overview=cloud_view(points,colors,truth,prediction,ee,
                        np.array([.54,-.20,.16])-BASE,1.3,(640,480))
                nearby=np.linalg.norm(points-target,axis=1)<.18
                zoom=cloud_view(points[nearby],(colors[nearby]*.35+160).clip(0,255).astype(np.uint8),
                    truth,prediction,ee,target,.25,(640,360),zoom=True)
                image=Image.new('RGB',(1344,1100),'white');image.paste(left,(20,112));image.paste(overview,(684,112));image.paste(zoom,(20,644))
                draw=ImageDraw.Draw(image);outcome='SUCCESS' if record['success'] else record['reason'].upper()
                draw.text((20,16),f'{experiment} | {outcome} | {record["target"]} | RGB-D + predicted target cloud',font=font(24,True),fill='#17324d')
                draw.text((20,55),f'{record["path"]} | {record["group"]} | {training} | H25/K{k} | t={frame/25:.2f}s',font=font(18),fill='#607587')
                draw.text((20,88),'FRONT RGB + GT / PREDICTED TARGET SURFACE',font=font(16,True),fill='#17324d')
                draw.text((684,88),'OBLIQUE VIEW / SAVED PHYSICAL TRAJECTORY' if physical_overview else 'FULL DENSE CLOUD FROM THE FRONT DEPTH CAMERA',font=font(16,True),fill='#17324d')
                draw.text((20,615),'TARGET ZOOM / SAME OBLIQUE BASIS / 25 cm WIDE',font=font(16,True),fill='#17324d')
                draw.text((684,615),'LOCALIZATION AND CONTROL DIAGNOSTICS',font=font(16,True),fill='#17324d')
                def number(value):return 'N/A (target not visible)' if value is None else f'{value:.1f} mm'
                texts=[('Pred vs GT surface Chamfer',number(m['surface_chamfer_mm'])),
                    ('Pred vs GT surface centroid',number(m['surface_centroid_error_mm'])),
                    ('Pred centroid vs EE',number(m['pred_centroid_to_EE_mm'])),
                    ('EE vs target body center',number(m['EE_to_target_center_mm'])),
                    ('Input depth / visible target',f'{valid.sum():,} / {len(truth)} pixels'),
                    ('Target lift / jaw contacts',f'{(target[2]-initial_height)*1000:.1f} mm / {int(contacts[frame-1,target_id]) if frame else 0}/2'),
                    ('Prediction age / refresh interval',f'{m["prediction_age_frames"]} frames / K{k}')]
                for i,(label,value) in enumerate(texts):
                    draw.text((684,660+i*48),label,font=font(17),fill='#607587')
                    draw.text((684,681+i*48),value,font=font(19,True),fill='#17324d')
                draw.text((20,1030),'Orange: GT visible target cloud | Cyan: auxiliary prediction | Green cross: measured EE',font=font(18,True),fill='#17324d')
                draw.text((20,1065),'Full-scene depth is the input. The predicted cloud is diagnostic; FM receives the decoder hidden.',font=font(17),fill='#607587')
                if frame==0:image.save(out/(name+'_start.png'))
                if frame==len(qpos)-1:image.save(out/(name+'_poster.png'))
                writer.stdin.write(np.asarray(image).tobytes())
        finally:
            writer.stdin.close();assert writer.wait()==0
            if physical_overview:overview_renderer.close()
    decoded=0
    with av.open(str(out/(name+'.mp4'))) as video:
        for frame in video.decode(video=0):decoded+=1
    assert decoded==len(qpos) and sha(trace)==trace_hash
    with (out/(name+'_metrics.jsonl')).open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    np.savez_compressed(out/(name+'_predictions.npz'),frames=np.array(replans),cloud_base_m=np.array(clouds))
    valid_rows=[r for r in rows if r['surface_chamfer_mm'] is not None]
    return dict(video=name+'.mp4',trace_sha256=trace_hash,record=record,decoded_frames=decoded,
        duration_s=decoded/25,first=rows[0],last=rows[-1],
        median_surface_chamfer_mm=float(np.median([r['surface_chamfer_mm'] for r in valid_rows])) if valid_rows else None,
        trace_unchanged=True)


@torch.inference_mode()
def main():
    test=ROOT/'test_k10';out=test/'cloud_1';out.mkdir(exist_ok=False)
    shutil.copyfile('docs/research/e120_cloud_video_README.md',out/'README.md')
    previous=json.loads((test/'videos_1/config.json').read_text());selected=previous['selected']
    assert sha(test/'checkpoint.pt')==previous['checkpoint_sha256']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    saved=torch.load(test/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
    policy=FusionPolicy(initialize=False).cuda();policy.load_state_dict(saved['model'],strict=True);policy.eval();del saved
    config=dict(status='rendering',checkpoint_step=previous['checkpoint_step'],checkpoint_sha256=previous['checkpoint_sha256'],
        selected=selected,H=25,K=10,flow_steps=10,noise_seed=0,frame_budget=300,fps=25,resolution=[1344,1100],
        same_cases_as='videos_1',auxiliary='recompute at actual pre-action replan frames0,10,...;hold between replans;no final extra replan',
        geometry='actual quantized2mm front depth + same GPU backprojection/calibration as policy;all valid pixels',
        GT='current eroded visible target instance segmentation,median depth outlier rejection;diagnostics only',
        plot='fixed oblique orthographic3D view;target zoom25cm wide;points2pixel splats',
        no_physics_rerun=True,no_action_resampling=True,source_sha256=sha(Path(__file__)))
    write(out/'config.json',config)
    model=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml');clip_planes(model)
    model.vis.global_.offwidth=320;model.vis.global_.offheight=240
    results=[]
    for record in tqdm(selected,desc='E120 cloud diagnosis videos',mininterval=5):
        results.append(render(record,test,out,model,policy))
        write(out/'progress.json',dict(status='rendering',complete=len(results),total=len(selected)))
    config['status']='complete';write(out/'config.json',config)
    write(out/'verification.json',dict(status='passed',clips=results,all_trace_hashes_unchanged=True,all_videos_fully_decoded=True))
    write(out/'progress.json',dict(status='complete',complete=6,total=6))
    print(json.dumps(dict(status='complete',out=str(out),clips=results)),flush=True)


if __name__=='__main__':main()
