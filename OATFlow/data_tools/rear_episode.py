"""Replay one pick episode with a mirrored fixed rear RGB-D camera."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import av
from matplotlib import colormaps
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import BASE,NAMES,write_json
from OATFlow.data_tools.render_dynamic import clip_planes,depth_encoder,UNIT,MAX_DEPTH
from OATFlow.simulation.render import visual_model
from OATFlow.simulation.io import encoder
from OATFlow.tracking.mask_video import digest,font

DATA=Path('/workspace/robotics_assets/datasets/pick_wide_rgbd')
EPISODE='g000/p00/t0'
ROOT=Path('/workspace/robotics_assets/experiments/e127_back')
NOTE=Path('docs/research/e127_README.md')
LOOK_AT=np.array([.54,-.12,.035])
RANGE=(.25,1.25)
LUT=(colormaps['turbo_r'](np.linspace(0,1,256))[:,:3]*255).astype(np.uint8)


def scene(root=ROOT,view="rear"):
    tree=ET.parse(DATA/'scene.xml');world=tree.getroot().find('worldbody')
    front=world.find("camera[@name='cam_front']");rear=ET.SubElement(world,'camera',dict(front.attrib))
    position=np.fromstring(front.get('pos'),sep=' ')
    position[:2]=2*LOOK_AT[:2]-position[:2]
    axes=np.fromstring(front.get('xyaxes'),sep=' ').reshape(2,3)@np.diag([-1.,-1.,1.])
    if view=='top':
        position=np.array([.54,-.10,1.20]);axes=np.array([[1.,0.,0.],[0.,1.,0.]])
    rear.set('name',f'cam_{view}');rear.set('pos',' '.join(map(str,position)))
    rear.set('xyaxes',' '.join(map(str,axes.ravel())))
    tree.write(root/'scene.xml')
    return dict(front_position_world_m=np.fromstring(front.get('pos'),sep=' ').tolist(),
                **{f'{view}_position_world_m':position.tolist()},
                look_at_world_m=([.54,-.10,0.] if view=='top' else LOOK_AT.tolist()),
                fovy_deg=float(front.get('fovy')),rotation=('90degree downward;image right +worldX,image up +worldY;same FOV' if view=='top' else
                          '180degree yaw about the common look-at point;same height/elevation/FOV'))


def depth_image(codes):
    metres=codes.astype(np.float32)*UNIT
    bins=np.rint(np.clip((metres-RANGE[0])/(RANGE[1]-RANGE[0]),0,1)*255).astype(int)
    pixels=LUT[bins].copy();pixels[codes==0]=(11,20,36)
    return Image.fromarray(pixels).resize((640,480),Image.Resampling.NEAREST)


def comparison(front,rear,front_depth,rear_depth,frame,phase,view="rear"):
    image=Image.new('RGB',(1280,1140),'#f4f6f9');draw=ImageDraw.Draw(image)
    draw.text((14,8),f'Front / {view} RGB-D | {EPISODE} | frame {frame:03d} | {frame/25:.2f}s | {phase}',font=font(22,True),fill='#21354a')
    for column,(name,rgb,depth) in enumerate((('Existing front',front,front_depth),(f'New {view}',rear,rear_depth))):
        x=column*640
        draw.text((x+14,45),name+' - RGB',font=font(20,True),fill='#21354a')
        image.paste(Image.fromarray(rgb).resize((640,480),Image.Resampling.BICUBIC),(x,74))
        draw.text((x+14,568),name+' - optical Z depth',font=font(20,True),fill='#21354a')
        image.paste(depth_image(depth),(x,598))
    gradient=np.repeat(LUT[None],12,axis=0)
    image.paste(Image.fromarray(gradient).resize((320,16)),(420,1094))
    draw.text((294,1090),'Near 0.25m',font=font(16),fill='#21354a')
    draw.text((752,1090),'Far 1.25m',font=font(16),fill='#21354a')
    draw.text((14,1118),'Same saved physical trajectory | uint16 FFV1, 2mm/unit | dark=invalid | fixed shared depth color scale',font=font(16),fill='#52647b')
    return image


def decode_check(path,n,depth=False,expected=None):
    h=hashlib.sha256();count=0
    with av.open(str(path)) as video:
        video.streams.video[0].thread_count=1
        assert float(video.streams.video[0].average_rate)==25.
        for item in video.decode(video=0):
            array=item.to_ndarray(format='gray16le' if depth else 'rgb24')
            if depth:h.update(array.astype('<u2',copy=False).tobytes())
            count+=1
    assert count==n,(path,count,n)
    if expected is not None:assert h.hexdigest()==expected
    return dict(frames=count,depth_sha256=h.hexdigest() if depth else None)


def main(root=ROOT,note=NOTE,view="rear"):
    assert view in ("rear","top")
    root.mkdir(exist_ok=False);shutil.copyfile(note,root/'README.md')
    source=DATA/EPISODE;started=time.perf_counter()
    files=('sim_state.npz','supervision.npz','labels.npz','episode.json','rgb_front.mp4','depth_front.mkv','calibration.npz','masks.npz')
    before={name:digest(source/name) for name in files};before['scene.xml']=digest(DATA/'scene.xml')
    for name in ('sim_state.npz','supervision.npz','labels.npz','episode.json','rgb_front.mp4','depth_front.mkv'):
        shutil.copyfile(source/name,root/name)
    specification=scene(root,view);model=visual_model(root/'scene.xml',object_bodies=NAMES);clip_planes(model)
    model.vis.global_.offwidth=320;model.vis.global_.offheight=240
    state=mujoco.MjData(model);option=mujoco.MjvOption();option.geomgroup[3]=0
    with np.load(source/'sim_state.npz') as z:states={k:z[k] for k in z.files}
    with np.load(source/'supervision.npz') as z:phases=z['phase'].copy()
    with np.load(source/'calibration.npz') as z:original_calibration={k:z[k] for k in z.files}
    with np.load(source/'masks.npz') as z:front_masks=z['instance'][:,0].copy()
    n=len(states['qpos']);assert n==138 and len(phases)==n
    assert states['qpos'].shape[1]==model.nq
    cameras=['cam_front',f'cam_{view}'];ids=[model.camera(c).id for c in cameras]
    np.testing.assert_array_equal(model.cam_fovy[ids],np.repeat(specification['fovy_deg'],2))
    transforms=np.tile(np.eye(4),(n,2,1,1));instance=np.zeros((n,2,240,320),np.uint8);instance[:,0]=front_masks
    lut=np.zeros(model.ngeom,np.uint8)
    for i,name in enumerate(NAMES,1):lut[model.geom_bodyid==model.body(name).id]=i
    for g in range(model.ngeom):
        if model.geom_bodyid[g]!=0 and model.body(int(model.geom_bodyid[g])).name not in NAMES:lut[g]=7
    lut[model.geom('table_top').id]=8
    config=dict(status='rendering',episode=EPISODE,frames=n,fps=25,resolution=[320,240],views=['front',view],
        depth_views=['front',view],camera=specification,source_dataset=str(DATA),source_hashes=before,
        source_sha256=digest(Path(__file__)),depth_unit_m=UNIT,depth_codec='FFV1 gray16le',invalid_depth_code=0,
        depth_model='ideal geometric optical Z;no sensor noise/focus/min-range model',
        depth_visualization=dict(colormap='turbo_r',range_m=list(RANGE),invalid='dark',resize='nearest'),
        replay='stored qpos/qvel/ctrl/mocap/equality states only;no physics stepping,new IK,policy or actions',
        front='copy existing RGB/depth bytes;unchanged calibration',training=False,checkpoint=None,H=None,K=None)
    write_json(root/'config.json',config)
    samples={0,int(np.flatnonzero(np.char.endswith(phases,'_engage'))[-1]),n-1}
    raw_hash=hashlib.sha256();error_sum=0.;error_max=0.;valid_count=0;counts=[]
    rgb_writer=encoder(root/f'rgb_{view}.mp4',320,240);depth_writer=depth_encoder(root/f'depth_{view}.mkv')
    preview_writer=encoder(root/'comparison.mp4',1280,1140)
    original_samples=model.vis.quality.offsamples
    rgb_renderer=mujoco.Renderer(model,height=240,width=320)
    model.vis.quality.offsamples=0;seg=mujoco.Renderer(model,height=240,width=320)
    model.vis.quality.offsamples=original_samples;seg.enable_segmentation_rendering()
    with rgb_renderer,seg,av.open(str(source/'rgb_front.mp4')) as rgb_video,av.open(str(source/'depth_front.mkv')) as depth_video:
        rgb_video.streams.video[0].thread_count=depth_video.streams.video[0].thread_count=1
        for frame,front,front_depth in tqdm(zip(range(n),rgb_video.decode(video=0),depth_video.decode(video=0),strict=True),
                                          total=n,desc=f'{view.capitalize()} RGB-D episode',mininterval=10):
            for key in ('qpos','qvel','ctrl','mocap_pos','mocap_quat','eq_active'):getattr(state,key)[:]=states[key][frame]
            state.time=float(states['timestamp_s'][frame]);mujoco.mj_forward(model,state)
            for vi,cid in enumerate(ids):
                transforms[frame,vi,:3,:3]=state.cam_xmat[cid].reshape(3,3)@np.diag([1.,-1.,-1.])
                transforms[frame,vi,:3,3]=state.cam_xpos[cid]
            np.testing.assert_allclose(transforms[frame,0],original_calibration['camera_to_world'][frame,0],atol=1e-12,rtol=0)
            rgb_renderer.update_scene(state,camera=f'cam_{view}',scene_option=option);rgb=rgb_renderer.render().copy()
            rgb_renderer.enable_depth_rendering();depth=rgb_renderer.render();rgb_renderer.disable_depth_rendering()
            valid=np.isfinite(depth)&(depth>0)&(depth<=MAX_DEPTH)
            codes=np.zeros(depth.shape,'<u2');codes[valid]=np.rint(depth[valid]/UNIT).astype(np.uint16)
            delta=np.abs(codes[valid].astype(np.float64)*UNIT-depth[valid])
            valid_count+=int(valid.sum());error_sum+=float(delta.sum());error_max=max(error_max,float(delta.max()))
            assert error_max<=.0011
            payload=codes.tobytes();raw_hash.update(payload);depth_writer.stdin.write(payload);rgb_writer.stdin.write(rgb.tobytes())
            seg.update_scene(state,camera=f'cam_{view}',scene_option=option);labels=seg.render()
            valid_geom=labels[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM)
            instance[frame,1][valid_geom]=lut[labels[...,0][valid_geom]]
            front_rgb=front.to_ndarray(format='rgb24');front_codes=front_depth.to_ndarray(format='gray16le')
            image=comparison(front_rgb,rgb,front_codes,codes,frame,str(phases[frame]),view);preview_writer.stdin.write(np.asarray(image).tobytes())
            if frame in samples:
                image.save(root/f'sample_{frame:03d}.png')
                Image.fromarray(rgb).save(root/f'rgb_{view}_{frame:03d}.png')
                depth_image(codes).save(root/f'depth_{view}_{frame:03d}.png')
            counts.append([[int((instance[frame,v]==t+1).sum()) for t in range(4)] for v in range(2)])
    for writer in (rgb_writer,depth_writer,preview_writer):writer.stdin.close();assert writer.wait()==0
    checks={}
    for name in ('rgb_front.mp4',f'rgb_{view}.mp4','comparison.mp4'):checks[name]=decode_check(root/name,n)
    checks['depth_front.mkv']=decode_check(root/'depth_front.mkv',n,depth=True)
    checks[f'depth_{view}.mkv']=decode_check(root/f'depth_{view}.mkv',n,depth=True,expected=raw_hash.hexdigest())
    focal=120/np.tan(np.deg2rad(specification['fovy_deg'])/2)
    K=np.repeat(np.array([[[focal,0,159.5],[0,focal,119.5],[0,0,1.]]]),2,axis=0)
    np.testing.assert_allclose(K[0],original_calibration['intrinsics'][0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(transforms[:,1],np.broadcast_to(transforms[0,1],transforms[:,1].shape),atol=1e-12,rtol=0)
    base=transforms.copy();base[...,:3,3]-=BASE
    np.savez_compressed(root/'calibration.npz',intrinsics=K,camera_to_world=transforms,camera_to_base=base,
        timestamp_s=states['timestamp_s'],views=np.array(['front',view]),depth_views=np.array(['front',view]),
        projection_model=np.array(['pinhole','pinhole']),distortion_coefficients=np.zeros((2,4)),
        depth_unit_m=UNIT,invalid_depth_code=0,depth_max_m=MAX_DEPTH,resolution=np.array([320,240]),
        visualization_rotation_deg=np.array([0,0]))
    np.savez_compressed(root/'masks.npz',instance=instance,views=np.array(['front',view]),timestamp_s=states['timestamp_s'])
    after={name:digest(source/name) for name in files};after['scene.xml']=digest(DATA/'scene.xml');assert before==after
    counts=np.array(counts)
    summary=dict(status='complete',frames=n,seconds=n/25,camera=specification,
        new_view_depth_quantization_mae_mm=1000*error_sum/valid_count,new_view_depth_quantization_max_mm=1000*error_max,
        initial_visible_pixels={'front':counts[0,0].tolist(),view:counts[0,1].tolist()},
        visible_frames_per_object={'front':(counts[:,0]>=8).sum(0).tolist(),view:(counts[:,1]>=8).sum(0).tolist()},
        video_checks=checks,source_dataset_unchanged=True,elapsed_s=time.perf_counter()-started)
    write_json(root/'summary.json',summary);write_json(root/'verification.json',dict(summary,status='passed'))
    config.update(status='complete',elapsed_s=summary['elapsed_s']);write_json(root/'config.json',config)
    text=note.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nResults:\n'+json.dumps(summary,indent=2)+'\n'
    note.write_text(text);(root/'README.md').write_text(text)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
