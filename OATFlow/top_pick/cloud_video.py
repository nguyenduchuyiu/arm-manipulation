"""Render one completed E119 K10 trace; overlay the auxiliary cloud for diagnosis."""
import json
from pathlib import Path
import shutil
import subprocess

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES, BASE
from OATFlow.simulation.pose import ARM_JOINTS
from OATFlow.simulation.render import visual_model, add_marker, path_overlay
from OATFlow.top_pick.cloud import ROOT, DATA, CloudPickPolicy
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.features import sha


def font(size, bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',size)


def project(points, state, model):
    cid=model.camera('cam_front').id
    rotation=state.cam_xmat[cid].reshape(3,3)@np.diag([1.,-1.,-1.])
    local=(points-state.cam_xpos[cid])@rotation
    focal=120/np.tan(np.deg2rad(model.cam_fovy[cid])/2)
    pixels=local[:,:2]/local[:,2:]*focal+[159.5,119.5]
    return pixels,local[:,2]>0


@torch.no_grad()
def main():
    test=ROOT/'test_k10';out=test/'video_1';out.mkdir(exist_ok=False)
    shutil.copyfile('docs/research/e119_video_README.md',out/'README.md')
    with (test/'results.jsonl').open() as f:record=json.loads(next(f))
    path=record['path'];trace=test/(path.replace('/','_')+'.npz')
    with np.load(trace) as z:
        qpos=np.vstack((z['initial_qpos'][None],z['qpos']))
        objects=z['objects'];contacts=z['contacts']
    saved=torch.load(test/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
    step=saved['step'];checkpoint_hash=sha(test/'checkpoint.pt')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    policy=CloudPickPolicy(initialize=False).cuda();policy.load_state_dict(saved['model'],strict=True);policy.eval()
    del saved
    model=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml')
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    state=mujoco.MjData(model)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    camera=mujoco.MjvCamera();camera.lookat[:]=[.54,-.20,.14]
    camera.distance=1.04;camera.azimuth=135;camera.elevation=-32
    target_id=NAMES.index(record['target']);body=model.body(record['target']).id
    site=model.site('grasp_site').id
    joint_ids=[model.joint(n).id for n in [*ARM_JOINTS,'right_jaw_slide_joint']]
    qadr=model.jnt_qposadr[joint_ids]
    config=dict(status='rendering',trace=str(trace),trace_sha256=sha(trace),record=record,
        checkpoint_step=step,checkpoint_sha256=checkpoint_hash,selection='first completed test case,regardless of outcome',
        H=25,K=10,flow_steps=10,noise_seed=0,frame_budget=300,source_policy_hz=25,
        video_fps=25,resolution=[1344,700],panels=['native policy front RGB enlarged2x','oblique overview'],
        auxiliary_inference='recompute cloud from current front RGB,targetID at saved replan states;hold between replans',
        GT='target center and physical contacts are overlays/scoring only',
        overlay_colors=dict(target='orange',predicted_cloud='cyan',measured_EE_path='green'),
        no_physics_rerun=True,source_sha256=sha(Path(__file__)))
    write(out/'config.json',config)
    command=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1344x700','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','2',
        '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'rollout.mp4')]
    writer=subprocess.Popen(command,stdin=subprocess.PIPE)
    points=[];ee=[];predicted=None;native_samples={}
    with mujoco.Renderer(model,height=240,width=320) as front,mujoco.Renderer(model,height=480,width=640,max_geom=1400) as overview:
        try:
            for frame in tqdm(range(len(qpos)),desc='Render E119 first K10 test video',mininterval=5):
                state.qpos[:]=qpos[frame];mujoco.mj_forward(model,state)
                target=state.xpos[body].copy();ee.append(state.site_xpos[site].copy())
                front.update_scene(state,camera='cam_front',scene_option=option);raw=front.render().copy()
                if predicted is None or (frame%10==0 and frame<300):
                    rgb=torch.from_numpy(raw).permute(2,0,1)[None].cuda()
                    query=torch.tensor([target_id],device='cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        predicted=policy.predict_cloud(policy.encode(rgb),query)[0].float().cpu().numpy()
                    assert np.isfinite(predicted).all();points.append((frame,predicted.copy()))
                left=Image.fromarray(raw).resize((640,480),Image.Resampling.BICUBIC)
                draw=ImageDraw.Draw(left)
                uv,valid=project(predicted+BASE,state,model)
                for x,y in uv[valid]*2:
                    if 0<=x<640 and 0<=y<480:draw.ellipse((x-1.3,y-1.3,x+1.3,y+1.3),fill='#27c8eb')
                uv,valid=project(target[None],state,model)
                if valid[0]:
                    x,y=uv[0]*2;draw.ellipse((x-12,y-12,x+12,y+12),outline='#ffb24d',width=3)
                    draw.text((x+15,y-14),'TARGET',font=font(15,True),fill='#ffb24d')
                overview.update_scene(state,camera=camera,scene_option=option)
                add_marker(overview.scene,target,[1.,.55,.08,1.],radius=.004)
                path_overlay(overview.scene,ee[::2],(50,220,140),.0015)
                for point in predicted:add_marker(overview.scene,point+BASE,[.1,.8,.95,.8],radius=.0014)
                image=Image.new('RGB',(1344,700),'#ffffff');image.paste(left,(20,112))
                image.paste(Image.fromarray(overview.render()),(684,112));draw=ImageDraw.Draw(image)
                draw.text((20,15),f'E119 | K10 pick rollout | final step {step:,} | {record["target"]}',font=font(25,True),fill='#17324d')
                draw.text((20,56),f'{path} | {record["group"]}: unseen color permutation | 4-layout / 256-demo training subset',font=font(17),fill='#607587')
                draw.text((20,88),'FRONT CAMERA / POLICY OBSERVATION',font=font(16,True),fill='#17324d')
                draw.text((684,88),'OBLIQUE VIEW / RECORDED PHYSICAL ROLLOUT',font=font(16,True),fill='#17324d')
                k=max(0,frame-1);lift=(objects[k,target_id,2]-objects[0,target_id,2])*1000
                gap=float(np.linalg.norm(state.site_xpos[site]-target)*1000)
                draw.text((20,612),f't={frame/25:5.2f}s | target lift={lift:5.1f} mm | jaw contacts={int(contacts[k,target_id])}/2 | EE-center={gap:5.1f} mm',font=font(18,True),fill='#17324d')
                draw.text((20,651),'Orange: GT target | Cyan: auxiliary predicted cloud | Green: measured EE path. Cloud is not fed to FM.',font=font(17),fill='#607587')
                if frame==len(qpos)-1:
                    draw.text((1040,612),record['reason'].upper(),font=font(20,True),fill='#af5f26')
                    image.save(out/'poster.png')
                if frame in (0,100,200):image.save(out/f'frame_{frame:03d}.png')
                writer.stdin.write(np.asarray(image).tobytes())
        finally:
            writer.stdin.close();assert writer.wait()==0
    with av.open(str(out/'rollout.mp4')) as video:
        decoded=sum(1 for _ in video.decode(video=0))
    assert decoded==len(qpos)
    np.savez_compressed(out/'auxiliary_clouds.npz',frames=np.array([p[0] for p in points]),
        predicted_cloud_base_m=np.array([p[1] for p in points]),EE_world_m=np.array(ee))
    assert sha(trace)==config['trace_sha256']
    write(out/'verification.json',dict(status='passed',decoded_frames=decoded,fps=25,
        duration_s=decoded/25,physical_trace_unchanged=True,checkpoint_step=step,
        source='saved test trajectory, not rerun;auxiliary predictions are diagnostic overlays'))
    config['status']='complete';write(out/'config.json',config)
    with (out/'README.md').open('a') as f:f.write(f'\nComplete:{decoded}frames/25FPS. Result:{record["reason"]}. Checkpoint{step}.\n')
    print(json.dumps(dict(status='complete',video=str(out/'rollout.mp4'),record=record)),flush=True)


if __name__=='__main__':main()
