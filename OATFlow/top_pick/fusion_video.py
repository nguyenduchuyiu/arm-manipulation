"""Replay three successful and three failed E120 physical test trajectories."""
import json
from pathlib import Path
import shutil
import subprocess

import av
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.simulation.render import visual_model,path_overlay
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.cloud_video import font,project
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.fusion import ROOT

GROUPS=dict(composition='UNSEEN PERMUTATION / TRAINING LAYOUT',
            spatial='UNSEEN LAYOUT / SEEN PERMUTATION',
            combined='UNSEEN LAYOUT + UNSEEN PERMUTATION')


def render(record,test,out,model,checkpoint_step):
    trace=test/(record['path'].replace('/','_')+'.npz');trace_hash=sha(trace)
    with np.load(trace) as z:qpos=np.vstack((z['initial_qpos'][None],z['qpos']));contacts=z['contacts']
    assert len(qpos)==record['frames']+1
    name=('success_' if record['success'] else 'failure_')+record['group']
    video_path=out/(name+'.mp4');state=mujoco.MjData(model)
    target_body=model.body(record['target']).id;target_id=NAMES.index(record['target'])
    site=model.site('grasp_site').id;option=mujoco.MjvOption();option.geomgroup[3]=0
    camera=mujoco.MjvCamera();camera.lookat[:]=[.54,-.20,.14]
    camera.distance=1.04;camera.azimuth=135;camera.elevation=-32
    command=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
             '-s','1344x700','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','2',
             '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video_path)]
    writer=subprocess.Popen(command,stdin=subprocess.PIPE)
    ee=[];initial_height=None;final_image=None
    with mujoco.Renderer(model,height=240,width=320) as front,mujoco.Renderer(model,height=480,width=640,max_geom=1400) as overview:
        try:
            for frame in tqdm(range(len(qpos)),desc=f'Render {name}',mininterval=5,leave=False):
                state.qpos[:]=qpos[frame];mujoco.mj_forward(model,state)
                target=state.xpos[target_body].copy();ee.append(state.site_xpos[site].copy())
                if initial_height is None:initial_height=target[2]
                front.update_scene(state,camera='cam_front',scene_option=option)
                left=Image.fromarray(front.render()).resize((640,480),Image.Resampling.BICUBIC)
                draw=ImageDraw.Draw(left);uv,valid=project(target[None],state,model)
                if valid[0]:
                    x,y=uv[0]*2
                    draw.ellipse((x-15,y-15,x+15,y+15),outline='#ffad32',width=2)
                    draw.text((max(5,min(x+18,525)),max(5,y-22)),'TARGET',font=font(16,True),fill='#ffad32')
                overview.update_scene(state,camera=camera,scene_option=option)
                path_overlay(overview.scene,ee[::2],(40,205,130),.0013)
                image=Image.new('RGB',(1344,700),'white');image.paste(left,(20,112))
                image.paste(Image.fromarray(overview.render()),(684,112));draw=ImageDraw.Draw(image)
                outcome='SUCCESS' if record['success'] else record['reason'].upper()
                color='#16814d' if record['success'] else '#b74835'
                draw.text((20,15),f'E120 | {outcome} | {record["target"]} | final step {checkpoint_step:,}',font=font(25,True),fill=color)
                draw.text((20,56),f'{record["path"]} | {GROUPS[record["group"]]}',font=font(18),fill='#607587')
                draw.text((20,88),'FRONT CAMERA / POLICY RGB VIEW',font=font(16,True),fill='#17324d')
                draw.text((684,88),'OBLIQUE VIEW / SAVED PHYSICAL TRAJECTORY',font=font(16,True),fill='#17324d')
                lift=(target[2]-initial_height)*1000
                contact=int(contacts[frame-1,target_id]) if frame else 0
                draw.text((20,612),f't={frame/25:5.2f}s | target lift={lift:6.1f} mm | jaw contacts={contact}/2 | H25/K10',font=font(19,True),fill='#17324d')
                draw.text((20,651),'Orange: GT target annotation only | Green: EE path | Replay of recorded qpos; no new rollout.',font=font(17),fill='#607587')
                if frame==0:image.save(out/(name+'_start.png'))
                if frame==len(qpos)-1:
                    draw.text((1040,612),'END: '+outcome,font=font(18,True),fill=color)
                    image.save(out/(name+'_poster.png'));final_image=image
                writer.stdin.write(np.asarray(image).tobytes())
        finally:
            writer.stdin.close();assert writer.wait()==0
    with av.open(str(video_path)) as video:
        decoded=0
        for frame in video.decode(video=0):
            assert frame.width==1344 and frame.height==700
            decoded+=1
        assert decoded==len(qpos)
    assert sha(trace)==trace_hash
    return dict(video=video_path.name,trace=str(trace),trace_sha256=trace_hash,record=record,
                decoded_frames=len(qpos),duration_s=len(qpos)/25,physical_trace_unchanged=True),final_image


def main():
    test=ROOT/'test_k10';out=test/'videos_1';out.mkdir(exist_ok=False)
    shutil.copyfile('docs/research/e120_video_README.md',out/'README.md')
    results=[json.loads(s) for s in (test/'results.jsonl').read_text().splitlines()]
    protocol=json.loads((test/'config.json').read_text())
    assert sha(test/'checkpoint.pt')==protocol['checkpoint_sha256']
    selected=[]
    for group in GROUPS:
        for success in (True,False):
            selected.append(next(r for r in results if r['group']==group and r['success']==success))
    config=dict(status='rendering',checkpoint_step=protocol['checkpoint_step'],
        checkpoint_sha256=protocol['checkpoint_sha256'],training=protocol['training'],
        H=protocol['H'],K=protocol['K'],flow_steps=protocol['flow_steps'],noise_seed=protocol['noise_seed'],
        frame_budget=protocol['frame_budget'],resolution=[1344,700],fps=25,
        selection='First completed success and first completed failure in each of three groups,from fixed results snapshot',
        results_snapshot_cases=len(results),snapshot_successes=sum(r['success'] for r in results),
        selected=selected,no_policy_rerun=True,no_auxiliary_inference=True,
        overlays='GT target annotation and measured EE path only; no changes to observations/actions/physics',
        source_sha256=sha(Path(__file__)))
    write(out/'config.json',config)
    model=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml');clip_planes(model)
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    rendered=[];contact_sheet=Image.new('RGB',(1344,1050),'white')
    for i,record in enumerate(tqdm(selected,desc='E120 success/failure videos',mininterval=5)):
        result,poster=render(record,test,out,model,protocol['checkpoint_step']);rendered.append(result)
        contact_sheet.paste(poster.resize((672,350),Image.Resampling.LANCZOS),((i%2)*672,(i//2)*350))
        write(out/'progress.json',dict(status='rendering',complete=len(rendered),total=len(selected)))
    contact_sheet.save(out/'contact_sheet.png')
    config['status']='complete';write(out/'config.json',config)
    write(out/'verification.json',dict(status='passed',clips=rendered,video_count=len(rendered),
        source_checkpoint_verified=True,all_saved_traces_unchanged=True,all_videos_fully_decoded=True))
    with (out/'README.md').open('a') as f:
        f.write('\nComplete. All six videos decoded and original trace hashes unchanged.\n')
        for r in rendered:f.write(f'\n- {r["video"]}: {r["record"]["path"]}, {r["record"]["reason"]}, {r["duration_s"]:.2f}s.\n')
    print(json.dumps(dict(status='complete',out=str(out),clips=rendered)),flush=True)


if __name__=='__main__':main()
