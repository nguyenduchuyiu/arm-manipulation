"""Replay saved E122 traces to score clouds against hidden-object simulator GT."""
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import BASE, NAMES
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.simulation.render import visual_model
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.geometry_probe import state_hash, cloud_metric
from OATFlow.top_pick.patch_fusion import PatchFusionPolicy
from OATFlow.top_pick.xyz_map import camera_calibration
from OATFlow.tracking.mask_video import font

PARENT=Path('/workspace/robotics_assets/experiments/e122_full')
ROOT=Path('/workspace/robotics_assets/experiments/e130_probe/test_occlusion')
NOTE=Path('docs/research/e130_occlusion_README.md')
CHECKPOINT=PARENT/'train/final.pt'
K=15


def backproject(depth,K,T,mask):
    v,u=np.where(mask);z=depth[v,u]
    p=np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z]
    return p@T[:3,:3].T+T[:3,3]


def box_distance(points,center,rotation,half):
    q=np.abs((points-center)@rotation)-half
    return np.abs(np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0))


def metrics(points,truth,center,rotation,half,ee):
    distance=box_distance(points,center,rotation,half)
    return dict(oracle_surface_centroid_error_mm=float(np.linalg.norm(points.mean(0)-truth.mean(0))*1000),
        oracle_surface_chamfer_mm=cloud_metric(points,truth)['chamfer_mm'],
        box_surface_mean_distance_mm=float(distance.mean()*1000),box_surface_inlier_fraction_5mm=float((distance<=.005).mean()),
        centroid_to_body_center_mm=float(np.linalg.norm(points.mean(0)-center)*1000),
        centroid_to_EE_mm=float(np.linalg.norm(points.mean(0)-ee)*1000))


def choose_cases():
    result=[];test=PARENT/'test_k15';rows=list(map(json.loads,(test/'results.jsonl').read_text().splitlines()))
    for group in ('composition','spatial','combined'):
        for success in (True,False):
            row=next(r for r in rows if r['group']==group and r['success']==success)
            result.append(dict(family='heldout',test=str(test),record=row))
    test=PARENT/'test_xy';rows=list(map(json.loads,(test/'results.jsonl').read_text().splitlines()))
    for row in rows:result.append(dict(family='xy_'+row['group'],test=str(test),record=row))
    assert len(result)==22
    return result


def show_sample(sample,held,record,path):
    rgb=sample['rgb'];Kmat,T=sample['K'],sample['T']
    image=Image.new('RGB',(960,820),'#f4f6f9');image.paste(Image.fromarray(rgb).resize((960,720)),(0,68));draw=ImageDraw.Draw(image)
    for cloud,color in ((sample['truth'],'#ffb52d'),(held,'#02d6f0')):
        local=(cloud-T[:3,3])@T[:3,:3];uv=local@Kmat.T;uv=uv[:,:2]/uv[:,2:]
        for (u,v),z in zip(uv,local[:,2]):
            if z>0 and 0<=u<320 and 0<=v<240:
                x,y=u*3,v*3+68;draw.ellipse((x-1.8,y-1.8,x+1.8,y+1.8),fill=color)
    draw.text((12,7),f'{record["family"]} / {record["target"]} / f{record["frame"]} / visible {record["visible_fraction"]*100:.1f}%',font=font(24,True),fill='#21354a')
    draw.text((12,39),f'Held cloud age {record["held_age"]} | oracle-centroid error {record["held"]["oracle_surface_centroid_error_mm"]:.1f}mm',font=font(22),fill='#21354a')
    draw.text((12,792),'Orange: oracle unoccluded GT surface | Cyan: held prediction | same camera, x-ray diagnostic',font=font(17),fill='#21354a')
    image.save(path)


def summarize(records):
    result={}
    for family in ['heldout','xy_inside','xy_outside']:
        result[family]={}
        for category in ['mostly_visible','partial','heavy','closing_window','fully_hidden']:
            chosen=[r for r in records if r['family']==family and
                ((r['visibility_group']==category) if category in ('mostly_visible','partial','heavy') else
                 (0<=r['relative_to_close']<=15) if category=='closing_window' else r['visible_pixels']<8)]
            entry=dict(samples=len(chosen),cases=len({r['case'] for r in chosen}))
            if chosen:
                entry['median_visible_fraction']=float(np.median([r['visible_fraction'] for r in chosen]))
                for kind in ['fresh','held']:
                    entry[kind]={key:float(np.median([r[kind][key] for r in chosen])) for key in chosen[0][kind]}
                    entry[kind]['oracle_centroid_error_p90_mm']=float(np.percentile([r[kind]['oracle_surface_centroid_error_mm'] for r in chosen],90))
            result[family][category]=entry
    return result


@torch.inference_mode()
def main():
    ROOT.mkdir(exist_ok=False);shutil.copyfile(NOTE,ROOT/'README.md');started=time.perf_counter()
    torch.set_num_threads(4);torch.manual_seed(130);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint_hash=sha(CHECKPOINT);saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==23520
    policy=PatchFusionPolicy(initialize=False);policy.load_state_dict(saved['model'],strict=True);policy.requires_grad_(False);policy.cuda().eval()
    config=dict(status='running',checkpoint=str(CHECKPOINT),checkpoint_sha256=checkpoint_hash,step=saved['step'],training=saved['config'],
        device='CUDA FP32,TF32 disabled',K=K,action_resampling=False,physics_steps=False,architecture_changes=False,
        checkpoint_visual_memory=False,head_points=128,source_sha256=sha(Path(__file__)),cases=choose_cases(),
        visibility_thresholds=[.25,.75],closing_command_threshold_m=-.005,closing_window_frames=[-15,30],closing_sample_stride=3,
        GT='oracle target-only rendering;simulation body pose;diagnostic only,never model input')
    del saved;write(ROOT/'config.json',config);before=state_hash(policy)
    model=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml');clip_planes(model)
    model.vis.global_.offwidth=320;model.vis.global_.offheight=240
    state=mujoco.MjData(model);option=mujoco.MjvOption();option.geomgroup[3]=0
    only=mujoco.MjvOption();only.geomgroup[:]=0;only.geomgroup[5]=1;assert not (model.geom_group==5).any()
    rgb_renderer=mujoco.Renderer(model,height=240,width=320)
    msaa=model.vis.quality.offsamples;model.vis.quality.offsamples=0
    seg=mujoco.Renderer(model,height=240,width=320);seg.enable_segmentation_rendering();model.vis.quality.offsamples=msaa
    data=[];trace_hashes={};case_info=[];GT_surface_errors=[]
    right=int(model.jnt_qposadr[model.joint('right_jaw_slide_joint').id]);site=model.site('grasp_site').id
    with rgb_renderer,seg:
        for ci,case in enumerate(tqdm(config['cases'],desc='Saved closing / occlusion states',mininterval=5)):
            row=case['record'];path=Path(case['test'])/(row['path'].replace('/','_')+'.npz');trace_hashes[str(path)]=sha(path)
            with np.load(path) as z:
                seq=np.concatenate([z['initial_qpos'][None],z['qpos']],0);commands=z['commands']
            close=np.flatnonzero(commands[:,5]<-.005);assert len(close)>0,row['path'];onset=int(close[0])
            last=len(commands)-1
            frames=sorted(set(range(0,last+1,K))|set(range(max(0,onset-15),min(last,onset+30)+1,3))|{onset,min(onset+1,last)})
            body=model.body(row['target']).id;geoms=np.flatnonzero(model.geom_bodyid==body)
            half=model.geom(row['target']+'_visual').size.copy()
            case_info.append(dict(index=ci,family=case['family'],path=row['path'],success=row['success'],target=row['target'],close_frame=onset,
                saved_frames=len(seq),sampled_frames=frames))
            for frame in frames:
                state.qpos[:]=seq[frame];mujoco.mj_forward(model,state)
                rgb_renderer.update_scene(state,camera='cam_front',scene_option=option);rgb=rgb_renderer.render().copy()
                rgb_renderer.enable_depth_rendering();depth=rgb_renderer.render().copy();rgb_renderer.disable_depth_rendering()
                valid=np.isfinite(depth)&(depth>0)&(depth<=6.5535);codes=np.zeros(depth.shape,np.uint16);codes[valid]=np.rint(depth[valid]/.002).astype(np.uint16)
                Kmat,T=camera_calibration(model,state)
                seg.update_scene(state,camera='cam_front',scene_option=option);labels=seg.render().copy()
                visible=np.isin(labels[...,0],geoms)&(labels[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
                original=model.geom_group[geoms].copy();model.geom_group[geoms]=5
                seg.update_scene(state,camera='cam_front',scene_option=only);labels=seg.render().copy();amodal=labels[...,0]>=0
                seg.disable_segmentation_rendering();seg.enable_depth_rendering();oracle_depth=seg.render().copy();seg.disable_depth_rendering();seg.enable_segmentation_rendering()
                model.geom_group[geoms]=original
                assert not (visible&~amodal).any() and amodal.sum()>=8
                oracle_depth=np.rint(oracle_depth/.002)*.002
                truth=backproject(oracle_depth,Kmat,T,amodal)
                center=state.xpos[body].copy()-BASE;rotation=state.xmat[body].reshape(3,3).copy();ee=state.site_xpos[site].copy()-BASE
                GT_surface_errors.append(box_distance(truth,center,rotation,half))
                fraction=float(visible.sum()/amodal.sum());vg='mostly_visible' if fraction>=.75 else 'partial' if fraction>=.25 else 'heavy'
                data.append(dict(case=ci,family=case['family'],target=row['target'],path=row['path'],success=row['success'],frame=frame,
                    relative_to_close=frame-onset,visible_fraction=fraction,visibility_group=vg,visible_pixels=int(visible.sum()),oracle_pixels=int(amodal.sum()),
                    right_jaw_m=float(state.qpos[right]),jaw_command_m=float(commands[frame,5]),rgb=rgb,depth=codes,K=Kmat,T=T,
                    truth=truth,center=center,rotation=rotation,half=half,ee=ee))
    predictions=[]
    for start in tqdm(range(0,len(data),16),desc='Fresh clouds / frozen checkpoint',mininterval=5):
        batch=data[start:start+16];rgb=torch.from_numpy(np.stack([d['rgb'] for d in batch])).permute(0,3,1,2).cuda()
        target=torch.tensor([NAMES.index(d['target']) for d in batch],device='cuda')
        scene={key:torch.as_tensor(np.stack([d[k] for d in batch]),device='cuda',dtype=torch.float32)
               for key,k in [('depth','depth'),('intrinsics','K'),('camera_to_base','T')]}
        predictions.extend(policy.predict_cloud(policy.encode(rgb),target,scene).cpu().numpy())
    predictions=np.stack(predictions);records=[];held_indices=[]
    lookup={(d['case'],d['frame']):i for i,d in enumerate(data)}
    for i,d in enumerate(tqdm(data,desc='Compare hidden-object geometry',mininterval=5)):
        held=lookup[(d['case'],d['frame']//K*K)];held_indices.append(held)
        row={k:d[k] for k in ['case','family','target','path','success','frame','relative_to_close','visible_fraction','visibility_group','visible_pixels','oracle_pixels','right_jaw_m','jaw_command_m']}
        row.update(held_frame=data[held]['frame'],held_age=d['frame']-data[held]['frame'])
        for kind,index in [('fresh',i),('held',held)]:row[kind]=metrics(predictions[index].astype(float),d['truth'],d['center'],d['rotation'],d['half'],d['ee'])
        records.append(row)
    write(ROOT/'records.json',records);write(ROOT/'cases.json',case_info)
    np.savez_compressed(ROOT/'clouds.npz',fresh_predictions_base_m=predictions,held_source_indices=np.array(held_indices),
        oracle_surface_xyz_base_m=np.concatenate([d['truth'] for d in data]),oracle_surface_offsets=np.r_[0,np.cumsum([len(d['truth']) for d in data])],
        body_centers_base_m=np.stack([d['center'] for d in data]),EE_base_m=np.stack([d['ee'] for d in data]))
    # Every case receives its own timeline; stable ordering is saved in cases.json.
    for info in case_info:
        ids=[i for i,d in enumerate(data) if d['case']==info['index']];rr=[records[i] for i in ids]
        fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True,layout='constrained');frames=[r['frame'] for r in rr]
        axes[0].plot(frames,[r['visible_fraction']*100 for r in rr],color='#4a875c');axes[0].set(ylabel='Visible target (%)',ylim=(-2,102))
        for key,color in [('fresh','#167dab'),('held','#e1872d')]:
            axes[1].plot(frames,[r[key]['oracle_surface_centroid_error_mm'] for r in rr],label=key,color=color)
            axes[2].plot(frames,[r[key]['box_surface_mean_distance_mm'] for r in rr],label=key,color=color)
        axes[1].set(ylabel='Oracle centroid error (mm)');axes[2].set(ylabel='Box surface distance (mm)',xlabel='Saved physical frame (25 Hz)')
        for ax in axes:ax.axvline(info['close_frame'],color='gray',ls='--');ax.grid(alpha=.2)
        axes[1].legend();fig.suptitle(f'{info["family"]} / {info["path"]} / {"success" if info["success"] else "failure"} | dashed = closing command begins')
        fig.savefig(ROOT/f'timeline_{info["index"]:02d}.png',dpi=150);plt.close(fig)
        near=[i for i in ids if 0<=data[i]['relative_to_close']<=30]
        chosen=min(near,key=lambda i:data[i]['visible_fraction'])
        show_sample(data[chosen],predictions[held_indices[chosen]],records[chosen],ROOT/f'sample_{info["index"]:02d}.png')
    summary=dict(status='complete',cases=len(case_info),samples=len(records),groups=summarize(records),
        oracle_GT_box_surface_MAE_mm=float(np.concatenate(GT_surface_errors).mean()*1000),elapsed_s=time.perf_counter()-started)
    write(ROOT/'summary.json',summary)
    assert state_hash(policy)==before and sha(CHECKPOINT)==checkpoint_hash
    assert all(sha(Path(p))==h for p,h in trace_hashes.items())
    assert all(r['held_frame']==r['frame']//K*K and 0<=r['held_age']<K for r in records)
    write(ROOT/'verification.json',dict(status='passed',checkpoint_unchanged=True,model_state_unchanged=True,trace_hashes_unchanged=True,
        no_physics_steps=True,no_action_resampling=True,no_architecture_changes=True,visible_mask_subset_of_oracle=True,held_cloud_cadence_verified=True))
    config.update(status='complete',samples=len(records),trace_hashes=trace_hashes);write(ROOT/'config.json',config)
    text=NOTE.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nResults:\n```json\n'+json.dumps(summary,indent=2)+'\n```\n'
    NOTE.write_text(text);(ROOT/'README.md').write_text(text);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
