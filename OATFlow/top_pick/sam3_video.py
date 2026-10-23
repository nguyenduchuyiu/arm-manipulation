"""Replay saved E133 cases with exact SAM3 masks and policy cloud inputs."""
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
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.cloud_video import font
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.sam3_fusion import ROOT
from OATFlow.top_pick.two_view import Rig,VIEWS,calibration

OUT=ROOT/'videos'
NOTE=Path('docs/research/e133_video_README.md')


def project(points,K,T):
    local=(points-T[:3,3])@T[:3,:3];pixels=local@K.T
    return pixels[:,:2]/pixels[:,2:],local[:,2]>0


def cloud_zoom(cloud,center,rotation,vertices,ee):
    image=Image.new('RGB',(640,360),'#edf2f6');draw=ImageDraw.Draw(image)
    def xy(points):return (points-center)@rotation[:,:2]*2000+[320,180]
    corners=xy(vertices)
    for i in range(8):
        for bit in (1,2,4):
            j=i^bit
            if j>i:draw.line((*corners[i],*corners[j]),fill='#e48b20',width=2)
    for x,y in xy(cloud):
        if 0<=x<640 and 0<=y<360:draw.ellipse((x-2.5,y-2.5,x+2.5,y+2.5),fill='#009cbd')
    x,y=xy(ee[None])[0];draw.line((x-6,y,x+6,y),fill='#a3349c',width=3);draw.line((x,y-6,x,y+6),fill='#a3349c',width=3)
    draw.line((22,332,122,332),fill='#243746',width=3);draw.text((22,305),'50 mm',font=font(15),fill='#243746')
    return image


def render(label,suite,row,rig):
    test=ROOT/f'test_{suite}';name=row['path'].replace('/','_');trace=test/(name+'.npz')
    metadata=test/(name+'_observations.json');hashes={str(p):sha(p) for p in [trace,metadata]}
    with np.load(trace) as z:
        states=np.concatenate([z['initial_qpos'][None],z['qpos']]);objects=z['objects'];contacts=z['contacts']
        cloud_frames=z['cloud_replan_frames'];clouds=z['input_cloud_base_m'];mask_frames=z['mask_replan_frames']
        masks=np.unpackbits(z['predicted_mask_packed'],axis=-1,count=320,bitorder='little').astype(bool)
    observations=json.loads(metadata.read_text());target=NAMES.index(row['target']);body=rig.model.body(row['target']).id
    bodies=[rig.model.body(n).id for n in NAMES[:4]];half=rig.model.geom(row['target']+'_visual').size.copy()
    signs=np.array([[1 if i&(1<<k) else -1 for k in range(3)] for i in range(8)])
    writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x1100',
        '-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(OUT/f'{label}.mp4')],stdin=subprocess.PIPE)
    pose_error=0.;last_thumbs=None
    for frame,qpos in enumerate(tqdm(states,desc=label,mininterval=5)):
        rig.state.qpos[:]=qpos;mujoco.mj_forward(rig.model,rig.state)
        centers=rig.state.xpos[bodies].copy()
        if frame==0:initial=centers.copy();zoom_center=centers[target]-BASE+[0,0,.055]
        else:pose_error=max(pose_error,float(abs(centers-objects[frame-1]).max()))
        ci=int(np.searchsorted(cloud_frames,frame,side='right')-1);mi=int(np.searchsorted(mask_frames,frame,side='right')-1)
        cloud=clouds[ci];input_frame=int(mask_frames[mi]);age=frame-input_frame
        image=Image.new('RGB',(1280,1100),'#f7f9fc');draw=ImageDraw.Draw(image)
        draw.text((14,10),f'E133 | {row["target"]} | {label.replace("_"," ")} | {row["reason"].upper()}',font=font(25,True),fill='#17324d')
        draw.text((14,49),f'{row["path"]} | frame {frame}/{len(states)-1} | simulated t={frame/25:.2f}s | 1x physics replay',font=font(17),fill='#52647b')
        thumbs=[];front_rotation=None
        for j,camera in enumerate(VIEWS):
            rig.renderer.update_scene(rig.state,camera=camera,scene_option=rig.option);rgb=rig.renderer.render().copy()
            K,T=calibration(rig.model,rig.state,camera)
            if j==0:front_rotation=T[:3,:3]
            panel=Image.fromarray(rgb).resize((640,480),Image.Resampling.BICUBIC);d=ImageDraw.Draw(panel)
            uv,valid=project(cloud,K,T)
            for (u,v),good in zip(uv*2,valid):
                if good and 0<=u<640 and 0<=v<480:d.ellipse((u-1.5,v-1.5,u+1.5,v+1.5),fill='#00cbe9')
            uv,valid=project((centers[target]-BASE)[None],K,T)
            if valid[0]:
                x,y=uv[0]*2;d.ellipse((x-19,y-19,x+19,y+19),outline='#ffae3d',width=2)
            image.paste(panel,(j*640,110));draw.text((j*640+14,85),f'{["FRONT","LEFT"][j]} CURRENT RGB + LAST INPUT CLOUD',font=font(17,True),fill='#17324d')
            if frame==input_frame:
                pixels=rgb.astype(float);pixels[masks[mi,j]]=pixels[masks[mi,j]]*.45+np.array([0,200,255])*.55
                thumbs.append(Image.fromarray(pixels.clip(0,255).astype(np.uint8)))
        if frame==input_frame:last_thumbs=thumbs
        assert last_thumbs is not None and len(last_thumbs)==2
        vertices=(signs*half)@rig.state.xmat[body].reshape(3,3).T+centers[target]-BASE
        ee=rig.state.site_xpos[rig.model.site('grasp_site').id]-BASE
        image.paste(cloud_zoom(cloud,zoom_center,front_rotation,vertices,ee),(0,640))
        for j,thumb in enumerate(last_thumbs):image.paste(thumb,(640+j*320,640))
        draw=ImageDraw.Draw(image);draw.text((14,612),'EXACT FM INPUT CLOUD / FRONT-ALIGNED ZOOM',font=font(17,True),fill='#17324d')
        draw.text((654,612),f'SAM3 INPUT RGB + MASK / FRAME {input_frame} (HELD)',font=font(17,True),fill='#17324d')
        gain=(centers[:,2]-initial[:,2])*1000;c=contacts[frame-1] if frame else np.zeros(4,int)
        info=[f'Target lift {gain[target]:.1f} mm | jaw contacts {int(c[target])}/2',
            f'Input frame {cloud_frames[ci]} | cloud age {frame-int(cloud_frames[ci])} frames',
            f'Filtered points front/left: {observations[mi]["points_per_view"]}']
        if row['wrong_object']:
            wrong=NAMES.index(row['wrong_object']);info.append(f'{row["wrong_object"]} lift {gain[wrong]:.1f} mm | jaws {int(c[wrong])}/2')
        for i,line in enumerate(info):draw.text((654,905+i*30),line,font=font(17,True if i==0 else False),fill='#17324d')
        draw.text((14,1012),'Cyan: saved 128-point input | Orange: current GT cube (diagnosis only) | Magenta: measured EE',font=font(17),fill='#52647b')
        draw.text((14,1045),'25 FPS simulated playback; inference waits omitted. Masks shown only on their original input images.',font=font(17),fill='#52647b')
        writer.stdin.write(np.asarray(image).tobytes())
        if frame==len(states)-1:image.save(OUT/f'{label}.png')
    writer.stdin.close();assert writer.wait()==0
    with av.open(str(OUT/f'{label}.mp4')) as video:
        assert float(video.streams.video[0].average_rate)==25
        decoded=sum(1 for _ in video.decode(video=0))
    assert decoded==len(states) and pose_error<1e-6 and hashes=={str(p):sha(Path(p)) for p in hashes}
    return dict(label=label,suite=suite,record=row,video=f'{label}.mp4',frames=decoded,duration_s=decoded/25,
        source_hashes=hashes,replay_pose_max_error_m=pose_error,video_fully_decoded=True)


def main():
    OUT.mkdir(exist_ok=False);shutil.copyfile(NOTE,OUT/'README.md')
    held=list(map(json.loads,(ROOT/'test_heldout/results.jsonl').read_text().splitlines()))
    ood=list(map(json.loads,(ROOT/'test_ood/results.jsonl').read_text().splitlines()))
    selected=[('heldout_success','heldout',next(r for r in held if r['success'] and r['target']=='RedCube')),
        ('ood_success','ood',next(r for r in ood if r['success'] and r['target']=='GreenCube' and r['group']=='outside' and r['band_index']==2 and r['edge']=='x_high_y_high')),
        ('ood_timeout','ood',next(r for r in ood if not r['success'] and r['target']=='BlueCube' and r['group']=='outside' and r['band_index']==2 and r['edge']=='x_low_y_high')),
        ('non_target_lift','heldout',next(r for r in held if r['reason']=='wrong_object' and r['target']=='YellowCube'))]
    write(OUT/'config.json',dict(status='rendering',checkpoint_step=23520,H=25,K=15,flow_steps=10,noise_seed=0,frame_budget=300,
        selected=[dict(label=l,suite=s,record=r) for l,s,r in selected],video_fps=25,resolution=[1280,1100],source_sha256=sha(Path(__file__)),
        policy_inference=False,new_physics=False,training=False,input_cloud='exact saved FM inputs; held between replans',
        masks='exact predicted SAM3 masks on their original input RGB; held thumbnails',playback='1x simulated time; inference waits omitted'))
    rig=Rig(ROOT/'scene.xml');results=[render(*case,rig) for case in selected];rig.close()
    write(OUT/'verification.json',dict(status='passed',all_four_videos_fully_decoded=True,source_traces_unchanged=True,cases=results))
    cfg=json.loads((OUT/'config.json').read_text());cfg['status']='complete';write(OUT/'config.json',cfg)
    text=NOTE.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nVideos verified:\n'+json.dumps(results,indent=2)+'\n'
    NOTE.write_text(text);(OUT/'README.md').write_text(text);print(json.dumps(results),flush=True)


if __name__=='__main__':main()
