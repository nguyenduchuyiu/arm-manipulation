"""Fuse synchronized front/top RGB-D into per-frame, base-coordinate voxel clouds."""
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import av
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import BASE, NAMES, write_json
from OATFlow.data_tools.rear_episode import decode_check
from OATFlow.simulation.io import encoder
from OATFlow.tracking.mask_video import digest, font

SOURCE=Path('/workspace/robotics_assets/experiments/e128_top')
ROOT=Path('/workspace/robotics_assets/experiments/e129_fuse')
NOTE=Path('docs/research/e129_README.md')
VOXEL=.003
SIZE=(480,360)


def backproject(codes,K,T,unit):
    """Optical-Z depth -> dense XYZ in the common robot-base frame."""
    v,u=np.indices(codes.shape);z=codes.astype(np.float64)*unit
    local=np.stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z),-1)
    return local@T[:3,:3].T+T[:3,3]


def voxel_fuse(points,colors,views,size=VOXEL):
    """Centroid and mean RGB per fixed base-frame voxel; preserve view counts."""
    keys=np.floor(points/size).astype(np.int64)
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    counts=np.bincount(inverse);n=len(counts)
    xyz=np.stack([np.bincount(inverse,weights=points[:,k],minlength=n)/counts for k in range(3)],1)
    rgb=np.stack([np.bincount(inverse,weights=colors[:,k],minlength=n)/counts for k in range(3)],1)
    sources=np.stack([np.bincount(inverse,weights=(views==v),minlength=n) for v in range(2)],1)
    assert int(sources.sum())==len(points)
    assert (sources<=np.iinfo(np.uint16).max).all()
    return xyz.astype(np.float32),np.rint(rgb).clip(0,255).astype(np.uint8),sources.astype(np.uint16)


def splat(uv,z,colors,size,radius=1):
    uv=np.rint(uv).astype(int);width,height=size
    offsets=np.array([(x,y) for x in range(-radius,radius+1) for y in range(-radius,radius+1)])
    pixels=(uv[:,None]+offsets).reshape(-1,2)
    depth=np.repeat(z,len(offsets));rgb=np.repeat(colors,len(offsets),axis=0)
    good=(pixels[:,0]>=0)&(pixels[:,0]<width)&(pixels[:,1]>=0)&(pixels[:,1]<height)&(depth>0)
    pixels=pixels[good];depth=depth[good];rgb=rgb[good]
    order=np.argsort(depth,kind='stable');pixels=pixels[order];rgb=rgb[order]
    _,first=np.unique(pixels[:,1]*width+pixels[:,0],return_index=True)
    image=np.full((height,width,3),(23,32,44),np.uint8)
    pixels=pixels[first];image[pixels[:,1],pixels[:,0]]=rgb[first]
    return Image.fromarray(image)


def full_view(points,colors,K,T):
    local=(points-T[:3,3])@T[:3,:3];good=local[:,2]>0
    local=local[good];uv=local@K.T;uv=uv[:,:2]/uv[:,2:]
    return splat(uv*1.5,local[:,2],colors[good],SIZE)


def target_view(points,colors,center,T):
    local=(points-center)@T[:3,:3]
    uv=local[:,:2]*(480/.085)+[240,150]
    return splat(uv,local[:,2]+1,colors,(480,300),radius=2)


def write_ply(path,points,colors):
    array=np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for k,name in enumerate(('x','y','z')):array[name]=points[:,k]
    for k,name in enumerate(('red','green','blue')):array[name]=colors[:,k]
    header=('ply\nformat binary_little_endian 1.0\n'+f'element vertex {len(points)}\n'+
            'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as f:f.write(header.encode());f.write(array.tobytes())


def surface_distance(points,center,rotation,half):
    q=np.abs((points-center)@rotation)-half
    return np.abs(np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(1),0))


def main():
    ROOT.mkdir(exist_ok=False);shutil.copyfile(NOTE,ROOT/'README.md');started=time.perf_counter()
    files=('calibration.npz','masks.npz','labels.npz','supervision.npz','scene.xml',
           'rgb_front.mp4','rgb_top.mp4','depth_front.mkv','depth_top.mkv')
    before={name:digest(SOURCE/name) for name in files}
    with np.load(SOURCE/'calibration.npz') as z:cal={k:z[k] for k in z.files}
    assert cal['views'].tolist()==['front','top'];n=len(cal['timestamp_s']);assert n==138
    with np.load(SOURCE/'masks.npz') as z:masks=z['instance'].copy();np.testing.assert_array_equal(z['timestamp_s'],cal['timestamp_s'])
    with np.load(SOURCE/'labels.npz') as z:centers=z['entity_xyz_base_m'].copy();rotations=z['entity_rotation_world'].copy();np.testing.assert_array_equal(z['timestamp_s'],cal['timestamp_s'])
    with np.load(SOURCE/'supervision.npz') as z:phases=z['phase'].copy()
    scene=ET.parse(SOURCE/'scene.xml')
    half=np.array([np.fromstring(scene.find(f".//geom[@name='{name}_visual']").get('size'),sep=' ') for name in NAMES[:4]])
    config=dict(status='running',source=str(SOURCE),source_hashes=before,source_sha256=digest(Path(__file__)),
        views=['front','top'],frames=n,fps=25,voxel_size_m=VOXEL,coordinate_frame='robot base;metres',
        method='backproject optical Z;apply calibrated extrinsics;concatenate synchronized views;voxel centroid and mean RGB',
        roi_world_m=[[-.01,-.53,-.004],[1.09,.33,.65]],
        depth_unit_m=float(cal['depth_unit_m']),fusion_mask_input=False,
        diagnostic='GT masks isolate objects only for plots and surface-error checks;GT geometry never modifies fused XYZ',
        temporal_accumulation=False,ICP=False,TSDF=False,training=False,checkpoint=None,H=None,K=None,
        visualization='all columns use front camera perspective;target zoom uses its orientation;point splats,not a reconstructed mesh')
    write_json(ROOT/'config.json',config)
    video=encoder(ROOT/'comparison.mp4',1440,850)
    streams=[av.open(str(SOURCE/name)) for name in ('rgb_front.mp4','rgb_top.mp4','depth_front.mkv','depth_top.mkv')]
    for stream in streams:stream.streams.video[0].thread_count=1
    allxyz=[];allrgb=[];allsources=[];offsets=[0];records=[];roundtrip=0.
    surface=[[],[],[]];novel=[];table_errors=[[],[]];fusion_seconds=[]
    for frame,items in enumerate(tqdm(zip(*(s.decode(video=0) for s in streams),strict=True),total=n,desc='Fuse front/top RGB-D',mininterval=10)):
        rgbs=[items[i].to_ndarray(format='rgb24') for i in range(2)]
        codes=[items[i+2].to_ndarray(format='gray16le') for i in range(2)]
        points=[];colors=[];target=[];target_rgb=[];object_points=[[],[]]
        for v in range(2):
            xyz=backproject(codes[v],cal['intrinsics'][v],cal['camera_to_base'][frame,v],float(cal['depth_unit_m']))
            valid=codes[v]>0;world=xyz+BASE
            valid&=((world>=[-.01,-.53,-.004])&(world<=[1.09,.33,.65])).all(-1)
            points.append(xyz[valid]);colors.append(rgbs[v][valid])
            for obj in range(4):object_points[v].append(xyz[valid&(masks[frame,v]==obj+1)])
            selected=valid&(masks[frame,v]==1);target.append(xyz[selected]);target_rgb.append(rgbs[v][selected])
            table=xyz[valid&(masks[frame,v]==8)]
            table_errors[v].append(np.abs(table[:,2]))
            local=(xyz[valid]-cal['camera_to_base'][frame,v,:3,3])@cal['camera_to_base'][frame,v,:3,:3]
            uv=local@cal['intrinsics'][v].T;uv=uv[:,:2]/uv[:,2:]
            y,x=np.where(valid);roundtrip=max(roundtrip,float(np.abs(uv-np.c_[x,y]).max()))
        tick=time.perf_counter()
        fused,frgb,sources=voxel_fuse(np.concatenate(points),np.concatenate(colors),np.repeat([0,1],[len(p) for p in points]))
        fusion_seconds.append(time.perf_counter()-tick)
        fused_target,fc,_=voxel_fuse(np.concatenate(target),np.concatenate(target_rgb),np.repeat([0,1],[len(p) for p in target]))
        for obj in range(4):
            pair=[object_points[v][obj] for v in range(2)]
            fobj,_,_=voxel_fuse(np.concatenate(pair),np.zeros((sum(map(len,pair)),3)),np.repeat([0,1],list(map(len,pair))))
            for mode,p in enumerate([*pair,fobj]):surface[mode].append(surface_distance(p,centers[frame,obj],rotations[frame,obj],half[obj]))
        novel.append(float((cKDTree(target[0]).query(target[1])[0]>VOXEL).mean()))
        allxyz.append(fused);allrgb.append(frgb);allsources.append(sources);offsets.append(offsets[-1]+len(fused))
        provenance=np.array([[0,0,0],[0,187,220],[244,155,49],[182,122,241]],np.uint8)
        flags=(sources[:,0]>0).astype(int)+2*(sources[:,1]>0)
        image=Image.new('RGB',(1440,850),'#f4f6f9');draw=ImageDraw.Draw(image)
        draw.text((14,8),f'Front + top | calibrated voxel fusion 3 mm | frame {frame:03d} | {phases[frame]}',font=font(22,True),fill='#21354a')
        for c,(label,p,col,tar,tc) in enumerate(zip(('Front only','Top only','Fused'),[*points,fused],[*colors,frgb],[*target,fused_target],[*target_rgb,fc])):
            x=c*480;draw.text((x+12,45),f'{label}: {len(p):,} points',font=font(20,True),fill='#21354a')
            image.paste(full_view(p,col,cal['intrinsics'][0],cal['camera_to_base'][frame,0]),(x,76))
            draw.text((x+12,450),f'Red target zoom: {len(tar)} points',font=font(19,True),fill='#21354a')
            image.paste(target_view(tar,tc,centers[frame,0],cal['camera_to_base'][frame,0]),(x,482))
        draw.text((14,792),'Same front viewpoint | full-scene fusion uses RGB-D + calibration only | GT mask used for target zoom only',font=font(19),fill='#21354a')
        draw.text((14,823),'Per-frame fusion, no temporal accumulation or hidden-surface completion | dark pixels: no projected point',font=font(18),fill='#52647b')
        video.stdin.write(np.asarray(image).tobytes())
        if frame in (0,99,137):
            image.save(ROOT/f'sample_{frame:03d}.png');write_ply(ROOT/f'fused_{frame:03d}.ply',fused,frgb)
            write_ply(ROOT/f'sources_{frame:03d}.ply',fused,provenance[flags])
        records.append(dict(frame=frame,input_points=list(map(len,points)),fused_points=len(fused),target_input_points=list(map(len,target)),
            fused_target_points=len(fused_target),voxels_seen_by_both=int((flags==3).sum()),top_target_points_farther_than_3mm_from_front=novel[-1]))
    assert len(records)==n and roundtrip<1e-8
    video.stdin.close();assert video.wait()==0
    for stream in streams:stream.close()
    decode_check(ROOT/'comparison.mp4',n)
    xyz=np.concatenate(allxyz);rgb=np.concatenate(allrgb);sources=np.concatenate(allsources)
    np.savez_compressed(ROOT/'clouds.npz',xyz_base_m=xyz,rgb=rgb,source_counts=sources,offsets=np.array(offsets),timestamp_s=cal['timestamp_s'],voxel_size_m=VOXEL,views=cal['views'])
    with np.load(ROOT/'clouds.npz') as z:
        np.testing.assert_array_equal(z['xyz_base_m'],xyz);np.testing.assert_array_equal(z['rgb'],rgb);np.testing.assert_array_equal(z['source_counts'],sources)
        assert z['offsets'][-1]==len(xyz) and len(z['offsets'])==n+1 and np.isfinite(z['xyz_base_m']).all()
    assert before=={name:digest(SOURCE/name) for name in files}
    summary=dict(status='complete',frames=n,mean_input_points=np.mean([r['input_points'] for r in records],axis=0).tolist(),
        mean_fused_points=float(np.mean([r['fused_points'] for r in records])),
        initial_target_points=records[0]['target_input_points']+[records[0]['fused_target_points']],
        object_surface_mae_mm=[float(np.concatenate(a).mean()*1000) for a in surface],
        object_surface_p95_mm=[float(np.percentile(np.concatenate(a),95)*1000) for a in surface],
        table_plane_mae_mm=[float(np.concatenate(a).mean()*1000) for a in table_errors],
        mean_top_target_novel_fraction=float(np.mean(novel)),
        mean_concat_voxel_fusion_ms=float(np.mean(fusion_seconds)*1000),
        clouds_file_MiB=(ROOT/'clouds.npz').stat().st_size/2**20,elapsed_s=time.perf_counter()-started)
    write_json(ROOT/'summary.json',summary);write_json(ROOT/'records.json',records)
    write_json(ROOT/'verification.json',dict(status='passed',source_unchanged=True,frames=n,video_fully_decoded=True,
        projection_roundtrip_max_pixel_error=roundtrip,cloud_archive_roundtrip_exact=True,provenance_counts_conserve_input_points=True))
    config.update(status='complete');write_json(ROOT/'config.json',config)
    text=NOTE.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nResults:\n```json\n'+json.dumps(summary,indent=2)+'\n```\n'
    NOTE.write_text(text);(ROOT/'README.md').write_text(text);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
