"""Paired 20/32/48mm cloud-size test at one original held-out reach goal."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess

import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image,ImageDraw
import torch
from tqdm import tqdm

from OATFlow.reach.cloud_data import ROOT,DATA,camera,voxel_depth,sha,write
from OATFlow.reach.cloud_eval import rollouts
from OATFlow.reach.pointcloud import PointCloudReach
from OATFlow.reach.cloud_rollout import font
from OATFlow.simulation.render import visual_model,path_overlay,add_marker

OUT=ROOT/'test_size_k15'
SIDES=[.020,.032,.048]


def clouds_at_goal(index,center,cam):
    pixels=np.load(ROOT/'clouds/visible_pixels.npy')
    rng=np.random.default_rng(118)
    for count in tqdm(pixels[:index],desc='Recover original cloud sampler state',mininterval=5):
        rng.choice(int(count),128,replace=int(count)<128)
    state=copy.deepcopy(rng.bit_generator.state);clouds=[];counts=[]
    for side in SIDES:
        native,uv,codes,error=voxel_depth(center,cam,side=side)
        local=np.random.default_rng();local.bit_generator.state=copy.deepcopy(state)
        ids=local.choice(len(native),128,replace=len(native)<128)
        clouds.append(native[ids]);counts.append(len(native))
    original=np.load(ROOT/'clouds/clouds.npy',mmap_mode='r')[index]
    np.testing.assert_array_equal(clouds[0],original)
    return np.stack(clouds),counts


def input_plot(cloud,goal,path):
    fig=plt.figure(figsize=(2.8,2.65),dpi=100,facecolor='white')
    ax=fig.add_subplot(111,projection='3d');ax.scatter(*cloud.T,c='#05a7c9',s=10)
    ax.scatter(*goal,c='#eb8d2b',marker='+',s=55)
    for i,label in enumerate(('x','y','z')):
        getattr(ax,'set_'+label+'lim')(goal[i]-.032,goal[i]+.032)
        getattr(ax,'set_'+label+'label')(label.upper()+' (m)',fontsize=7,labelpad=0)
    ax.tick_params(labelsize=6,pad=0);ax.set_box_aspect([1,1,1]);ax.view_init(30,-90)
    fig.subplots_adjust(left=0,right=.92,bottom=.04,top=.99);fig.savefig(path);plt.close(fig)
    return Image.open(path).convert('RGB')


def render(arrays,records,clouds,cam):
    scene=DATA.parent/'scene.xml'
    model=visual_model(scene,lighting_scene='/workspace/robotics_assets/robot/nexarm/visual_reference.xml')
    # This display camera exactly matches the camera that generated the input depth.
    state=mujoco.MjData(model);model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    view=mujoco.MjvCamera();view.type=mujoco.mjtCamera.mjCAMERA_FREE
    look=cam['base']+[0,0,.25];delta=cam['position']-look
    view.lookat[:]=look;view.distance=np.linalg.norm(delta)
    view.azimuth=np.rad2deg(np.arctan2(delta[1],delta[0]))+180
    view.elevation=-np.rad2deg(np.arcsin(delta[2]/view.distance))
    model.vis.global_.fovy=70
    option=mujoco.MjvOption();option.geomgroup[3]=0
    insets=[input_plot(c,arrays['target'][i]-cam['base'],OUT/f'input_{round(side*1000)}mm.png') for i,(c,side) in enumerate(zip(clouds,SIDES))]
    writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1920x960','-r','12.5','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast',
        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'comparison.mp4')],stdin=subprocess.PIPE)
    with mujoco.Renderer(model,height=480,width=640,max_geom=2400) as renderer:
        try:
            for frame in tqdm(range(0,301,2),desc='Render paired cloud-size comparison',mininterval=5):
                image=Image.new('RGB',(1920,960),'white');draw=ImageDraw.Draw(image)
                draw.text((22,14),'E118 original reach | same goal + camera + state + noise; change cloud size only',font=font(29,True),fill='#17324d')
                draw.text((22,58),'Held-out voxel30254 | source120,000 | H25/K15/Euler10 | noise0 | 128points per cloud | no physical voxel collision',font=font(20),fill='#607587')
                for i,side in enumerate(SIDES):
                    x=i*640;goal=arrays['target'][i]
                    state.qpos[:]=arrays['qpos'][i,frame];mujoco.mj_forward(model,state)
                    np.testing.assert_allclose(state.site_xpos[model.site('grasp_site').id],arrays['ee'][i,frame],atol=2e-6,rtol=0)
                    renderer.update_scene(state,camera=view,scene_option=option)
                    # MuJoCo's free-camera convention is checked against the input camera.
                    camera_eye=(renderer.scene.camera[0].pos+renderer.scene.camera[1].pos)*.5
                    np.testing.assert_allclose(camera_eye,cam['position'],atol=2e-6,rtol=0)
                    forward=renderer.scene.camera[0].forward/np.linalg.norm(renderer.scene.camera[0].forward)
                    np.testing.assert_allclose(forward,cam['rotation'][:,2],atol=2e-6,rtol=0)
                    for axis in range(3):
                        other=[j for j in range(3) if j!=axis]
                        for a in (-1,1):
                            for b in (-1,1):
                                line=np.tile(goal,(2,1));line[:,axis]+=[-side/2,side/2]
                                line[:,other]+=[a*side/2,b*side/2]
                                path_overlay(renderer.scene,line,(238,144,45),.00065)
                    for point in np.unique(clouds[i],axis=0):add_marker(renderer.scene,point+cam['base'],[.02,.7,.85,.95],radius=.0011)
                    path_overlay(renderer.scene,arrays['ee'][i,:frame+1:2],(30,195,125),.0014)
                    rgb=Image.fromarray(renderer.render())
                    # Display-only digital zoom; input cloud/camera/physics are unchanged.
                    rgb=rgb.crop((200,150,440,330)).resize((640,480),Image.Resampling.BICUBIC)
                    image.paste(rgb,(x,130))
                    draw.text((x+20,99),f'{round(side*1000)} mm'+(' / TRAINING SIZE' if i==0 else ''),font=font(23,True),fill='#17324d')
                    image.paste(insets[i],(x+2,626));error=float(np.linalg.norm(arrays['ee'][i,frame]-goal)*1000)
                    draw.text((x+298,647),f't={frame/25:.2f}s',font=font(20,True),fill='#17324d')
                    draw.text((x+298,692),f'EE error: {error:.2f} mm',font=font(21,True),fill='#17324d')
                    draw.text((x+298,737),f'Final: {records[i]["final_error_mm"]:.2f} mm',font=font(19),fill='#607587')
                    if frame==300:
                        passed=records[i]['success_5mm'];status='STABLE 5mm PASS' if passed else 'STABLE 5mm FAIL'
                        draw.text((x+298,784),status,font=font(19,True),fill='#16814d' if passed else '#b74835')
                draw.text((22,900),'Cyan: actual input cloud | Orange: voxel boundary/goal | Green: measured EE path | Insets share the same metric scale',font=font(20),fill='#17324d')
                draw.text((22,935),'Source-camera RGB digitally enlarged2.67x for display only. One held-out goal/noise seed; not an overall robustness benchmark.',font=font(17),fill='#607587')
                if frame==0:image.save(OUT/'start.png')
                if frame==300:image.save(OUT/'poster.png')
                writer.stdin.write(np.asarray(image).tobytes())
        finally:
            writer.stdin.close();assert writer.wait()==0
    with av.open(str(OUT/'comparison.mp4')) as video:decoded=sum(1 for _ in video.decode(video=0))
    assert decoded==151
    return dict(status='passed',frames=decoded,fps=12.5,resolution=[1920,960],
        display_camera_matches_input_camera=True,display_only_crop_xyxy=[200,150,440,330],replay_EE_matches_trace=True)


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=False);shutil.copyfile('docs/research/e118_size_README.md',OUT/'README.md')
    source=ROOT/'train/best.pt';os.link(source,OUT/'checkpoint.pt')
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True);assert saved['step']==120000
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    policy=PointCloudReach().cuda();policy.load_state_dict(saved['model'],strict=True);policy.eval();policy.requires_grad_(False);del saved
    with np.load(ROOT/'train/evaluation_indices.npz') as z:index=int(z['test_episodes'][0])
    with np.load(ROOT/'train/split.npz') as z:
        assert index in z['test_trajectories'] and index not in z['train_trajectories']
    rows=[json.loads(s) for s in (DATA/'trajectories.jsonl').read_text().splitlines()];row=rows[index]
    cam=camera();clouds,counts=clouds_at_goal(index,np.array(row['target_world_m']),cam)
    config=dict(status='running',checkpoint_step=120000,checkpoint_source=str(source),checkpoint_sha256=sha(source),
        original_trajectory_index=index,voxel_id=row['voxel_id'],goal_world_m=row['target_world_m'],
        selection='first original fixed held-out goal;not selected for baseline success',
        cube_sides_m=SIDES,native_visible_depth_pixels=counts,points_per_input=128,baseline20mm_bit_exact_original_cache=True,
        camera=dict(position_world_m=cam['position'].tolist(),rotation_cv_to_world=cam['rotation'].tolist(),
                    K=cam['K'].tolist(),resolution=[640,480],fovy_deg=70),
        sampling='same original per-goal RNG state and sampling rule for each size',
        environment='original reach physics/home;isolated voxel is input/visual only,not a physical collider',
        H=25,K=15,flow_steps=10,noise_seed=0,paired_identical_noise_each_replan=True,frame_budget=300,
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('cloud_data.py'),Path(__file__).with_name('cloud_eval.py')]})
    write(OUT/'config.json',config);np.save(OUT/'clouds.npy',clouds)
    result,arrays=rollouts(policy,clouds,np.arange(3),[row]*3,seed=0,paired_noise=True)
    for i,record in enumerate(result['records']):record.update(cube_side_mm=round(SIDES[i]*1000),original_trajectory_index=index)
    write(OUT/'results.json',result);np.savez_compressed(OUT/'rollout.npz',**arrays)
    print(json.dumps(result),flush=True);del policy;torch.cuda.empty_cache()
    trace_hash=sha(OUT/'rollout.npz');verification=render(arrays,result['records'],clouds,cam)
    assert sha(OUT/'rollout.npz')==trace_hash and sha(source)==config['checkpoint_sha256']
    verification.update(physical_trace_unchanged=True,checkpoint_unchanged=True,baseline_cloud_bit_exact=True)
    write(OUT/'verification.json',verification);config['status']='complete';write(OUT/'config.json',config)
    print(json.dumps(dict(status='complete',output=str(OUT),records=result['records'])),flush=True)


if __name__=='__main__':main()
