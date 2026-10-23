"""Compare rear-camera heights with identical saved motion, intrinsics and XY."""
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image,ImageDraw
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES,write_json
from OATFlow.data_tools.rear_episode import ROOT as SOURCE,LOOK_AT,EPISODE,decode_check
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.simulation.render import visual_model
from OATFlow.simulation.io import encoder
from OATFlow.tracking.mask_video import digest,font

ROOT=SOURCE/'height_1'
NOTE=Path('docs/research/e127_height_README.md')
HEIGHTS=(.6,.8,1.)


def main():
    ROOT.mkdir(exist_ok=False);shutil.copyfile(NOTE,ROOT/'README.md');started=time.perf_counter()
    source_hash={name:digest(SOURCE/name) for name in ('scene.xml','sim_state.npz','calibration.npz','depth_rear.mkv')}
    tree=ET.parse(SOURCE/'scene.xml');world=tree.getroot().find('worldbody');original=world.find("camera[@name='cam_rear']")
    names=[];positions=[];angles=[]
    for height in HEIGHTS:
        cam=ET.SubElement(world,'camera',dict(original.attrib));name=f'rear_z{int(height*100):03d}'
        position=np.fromstring(original.get('pos'),sep=' ');position[2]=height
        backward=position-LOOK_AT;backward/=np.linalg.norm(backward)
        right=np.cross([0.,0.,1.],backward);right/=np.linalg.norm(right);up=np.cross(backward,right)
        cam.set('name',name);cam.set('pos',' '.join(map(str,position)))
        cam.set('xyaxes',' '.join(map(str,np.r_[right,up])))
        names.append(name);positions.append(position.tolist())
        angles.append(float(np.rad2deg(np.arctan2(height-LOOK_AT[2],np.linalg.norm(position[:2]-LOOK_AT[:2])))))
    tree.write(ROOT/'scene.xml');model=visual_model(ROOT/'scene.xml',object_bodies=NAMES);clip_planes(model)
    model.vis.global_.offwidth=320;model.vis.global_.offheight=240
    state=mujoco.MjData(model);option=mujoco.MjvOption();option.geomgroup[3]=0
    only=mujoco.MjvOption();only.geomgroup[:]=0;only.geomgroup[5]=1
    geoms=[np.flatnonzero(model.geom_bodyid==model.body(name).id) for name in NAMES[:4]]
    with np.load(SOURCE/'sim_state.npz') as z:states={k:z[k] for k in z.files}
    with np.load(SOURCE/'supervision.npz') as z:phases=z['phase'].copy()
    n=len(states['qpos']);visible=np.zeros((n,3,4),int);amodal=np.zeros_like(visible)
    samples=[0,int(np.flatnonzero(np.char.endswith(phases,'_engage'))[-1]),n-1]
    sheets=[];writer=encoder(ROOT/'height_comparison.mp4',1440,440)
    config=dict(status='rendering',heights_m=list(HEIGHTS),camera_positions_world_m=positions,
        downward_pitch_deg=angles,look_at_world_m=LOOK_AT.tolist(),FOVY_deg=float(original.get('fovy')),
        resolution=[320,240],fps=25,frames=n,episode=EPISODE,
        methodology='sameXY,FOV,look-at,state;only height and corresponding look-at pitch vary',
        visibility='visible instance pixels / target-alone projected pixels;>=8visible pixels counts as visible',
        source_sha256=digest(Path(__file__)),source_hashes=source_hash,
        training=False,checkpoint=None,actions='none;replay stored states',H=None,K=None,
        depth='RGB/visibility diagnostic only;candidate heights do not add depth files to the source dataset')
    write_json(ROOT/'config.json',config)
    rgb_renderer=mujoco.Renderer(model,height=240,width=320)
    samples_msaa=model.vis.quality.offsamples;model.vis.quality.offsamples=0
    seg=mujoco.Renderer(model,height=240,width=320);seg.enable_segmentation_rendering()
    model.vis.quality.offsamples=samples_msaa
    with rgb_renderer,seg:
        for frame in tqdm(range(n),desc='Compare rear-camera heights',mininterval=10):
            for key in ('qpos','qvel','ctrl','mocap_pos','mocap_quat','eq_active'):getattr(state,key)[:]=states[key][frame]
            mujoco.mj_forward(model,state)
            image=Image.new('RGB',(1440,440),'#f4f6f9');draw=ImageDraw.Draw(image)
            draw.text((12,7),f'Rear camera height | {EPISODE} | frame{frame:03d} | {phases[frame]} | identical XY and FOV',font=font(21,True),fill='#21354a')
            for h,camera in enumerate(names):
                rgb_renderer.update_scene(state,camera=camera,scene_option=option);rgb=rgb_renderer.render().copy()
                image.paste(Image.fromarray(rgb).resize((480,360),Image.Resampling.BICUBIC),(h*480,76))
                draw.text((h*480+12,43),f'Height {HEIGHTS[h]:.1f}m | downward {angles[h]:.1f}deg',font=font(19,True),fill='#21354a')
                seg.update_scene(state,camera=camera,scene_option=option);labels=seg.render().copy()
                valid=labels[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM)
                for target,g in enumerate(geoms):
                    vis=valid & np.isin(labels[...,0],g);visible[frame,h,target]=int(vis.sum())
                    old=model.geom_group[g].copy();model.geom_group[g]=5
                    seg.update_scene(state,camera=camera,scene_option=only);mask=seg.render()[...,0]>=0
                    model.geom_group[g]=old
                    assert not (vis & ~mask).any()
                    amodal[frame,h,target]=int(mask.sum())
            writer.stdin.write(np.asarray(image).tobytes())
            if frame in samples:image.save(ROOT/f'frame_{frame:03d}.png');sheets.append(image.copy())
    writer.stdin.close();assert writer.wait()==0;decode_check(ROOT/'height_comparison.mp4',n)
    sheet=Image.new('RGB',(1440,440*len(sheets)),'white')
    for i,image in enumerate(sheets):sheet.paste(image,(0,i*440))
    sheet.save(ROOT/'comparison.png')
    fractions=np.divide(visible,amodal,out=np.full(visible.shape,np.nan),where=amodal>0)
    summary=[]
    for h,height in enumerate(HEIGHTS):
        summary.append(dict(height_m=height,pitch_deg=angles[h],
            visible_frames_per_object=(visible[:,h]>=8).sum(0).tolist(),
            frames_all4_visible=int((visible[:,h]>=8).all(1).sum()),
            mean_visible_fraction_per_object=np.nanmean(fractions[:,h],axis=0).tolist(),
            mean_visible_fraction=float(np.nanmean(fractions[:,h])),
            mean_unoccluded_object_pixels=float(amodal[:,h].mean()),
            mean_visible_object_pixels=float(visible[:,h].mean()),
            target_out_of_fov_frames=(amodal[:,h]==0).sum(0).tolist()))
    np.savez_compressed(ROOT/'visibility.npz',visible_pixels=visible,unoccluded_pixels=amodal,
                        heights_m=np.array(HEIGHTS),timestamp_s=states['timestamp_s'])
    assert all(digest(SOURCE/name)==value for name,value in source_hash.items())
    write_json(ROOT/'summary.json',summary)
    write_json(ROOT/'verification.json',dict(status='passed',source_unchanged=True,
        every_visible_mask_subset_of_unoccluded=True,frames=n,heights=len(HEIGHTS),video_fully_decoded=True))
    config.update(status='complete',elapsed_s=time.perf_counter()-started);write_json(ROOT/'config.json',config)
    text=NOTE.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nResults:\n'+json.dumps(summary,indent=2)+'\n'
    NOTE.write_text(text);(ROOT/'README.md').write_text(text)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
