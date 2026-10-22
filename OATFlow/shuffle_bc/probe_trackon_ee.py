"""Reconstruct saved rollouts and intervene on joints to diagnose XYZ/EE shortcuts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES, Scene
from OATFlow.shuffle_bc.context_model import FrozenWristViT
from OATFlow.shuffle_bc.data import DATA
from OATFlow.shuffle_bc.trackon import FrozenTrackOn
from OATFlow.shuffle_bc.trackon_model import TrackOnContext
from OATFlow.simulation.render import visual_model
from OATFlow.tracking.data import read_segment
from OATFlow.tracking.point_data import top_camera, project
from OATFlow.tracking.utils import write_json

ROOT = Path('/workspace/robotics_assets/experiments/e111_bc')
INITIAL = Path('/workspace/robotics_assets/experiments/e110_trackon/train_2/best.pt')


@torch.inference_mode()
def reconstruct(tracker, wrist, row, trace, camera, model, visual, renderer, option):
    scene = Scene(model); state = mujoco.MjData(model); display = mujoco.MjData(visual)
    context = read_segment(DATA, row['context_path'], 10, views=('top',), center_crop=True)
    click = torch.tensor(project(context['xyz'][0, row['target_id']], camera), device='cuda', dtype=torch.float32)
    memory = None
    for i in tqdm(range(len(context['raw_index'])), desc='Replay prior shuffle', mininterval=5, leave=False):
        with torch.autocast('cuda', dtype=torch.float16):
            features = tracker.features(context['rgb'][0, i:i+1].cuda())
            if memory is None: memory = tracker.initialize(features, click)
            hidden, uv, visible = tracker.step(features, memory)
    result = {k: [] for k in ('top', 'left', 'joints', 'ee', 'truth', 'native_uv', 'visible', 'frame', 'saved')}
    for frame in tqdm(range(len(trace['commands'])), desc=f'Reconstruct {row["path"]}', mininterval=5, leave=False):
        if frame % 3 and frame % 10: continue
        state.qpos[:] = trace['qpos'][frame]; mujoco.mj_kinematics(model, state)
        display.qpos[:] = state.qpos; mujoco.mj_forward(visual, display)
        if frame % 3 == 0:
            renderer.update_scene(display, camera='cam_top', scene_option=option)
            rgb = torch.from_numpy(renderer.render()[40:280].copy()).permute(2, 0, 1)[None].cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                hidden, uv, visible = tracker.step(tracker.features(rgb), memory)
        if frame % 10 == 0:
            renderer.update_scene(display, camera='cam_left', scene_option=option)
            result['left'].append(torch.from_numpy(renderer.render().copy()).permute(2, 0, 1))
            result['top'].append(hidden.half().cpu())
            result['joints'].append(state.qpos[scene.qadr].copy())
            result['ee'].append(state.site_xpos[scene.site].copy() - camera['base'])
            result['truth'].append(state.xpos[model.body(NAMES[row['target_id']]).id].copy() - camera['base'])
            result['native_uv'].append(uv.float().cpu().numpy()); result['visible'].append(float(visible))
            result['frame'].append(frame)
            i = int(np.flatnonzero(trace['prediction_frames'] == frame)[0])
            result['saved'].append(trace['predictions'][i])
            np.testing.assert_allclose(result['truth'][-1], trace['prediction_truth'][i], atol=1e-10, rtol=0)
    result['top'] = torch.stack(result['top'])
    left = torch.stack(result.pop('left')); features = []
    for start in range(len(left)):
        with torch.autocast('cuda', dtype=torch.bfloat16): features.append(wrist(left[start:start+1].cuda()).half().cpu())
    result['wrist'] = torch.cat(features)
    for key in result:
        if not isinstance(result[key], torch.Tensor): result[key] = torch.from_numpy(np.array(result[key]))
    return result


@torch.inference_mode()
def predictions(model, data, joints=None, top=None):
    result = []
    for start in range(len(data['top'])):
        # Match the original batch-one BF16 rollout; batching changes numerical kernels.
        ids = slice(start, start+1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value = model.xyz_from_features((data['top'] if top is None else top)[ids].cuda(),
                data['wrist'][ids].cuda(), (data['joints'] if joints is None else joints)[ids].float().cuda())
        result.append((value*model.xyz_std+model.xyz_mean).cpu().numpy())
    return np.concatenate(result)


def distances(pred, truth, ee, camera):
    gt = np.linalg.norm(pred-truth, axis=-1)*1000
    to_ee = np.linalg.norm(pred-ee, axis=-1)*1000
    gt_xy = np.linalg.norm((pred-truth)[:, :2], axis=-1)*1000
    ee_xy = np.linalg.norm((pred-ee)[:, :2], axis=-1)*1000
    return dict(mean_GT_mm=float(gt.mean()), median_GT_mm=float(np.median(gt)),
        mean_EE_mm=float(to_ee.mean()), mean_GT_xy_mm=float(gt_xy.mean()), mean_EE_xy_mm=float(ee_xy.mean()),
        closer_EE_than_GT_fraction=float(np.mean(to_ee < gt)), closer_EE_than_GT_xy_fraction=float(np.mean(ee_xy < gt_xy)),
        mean_projected_GT_px=float(np.linalg.norm(project(pred, camera)-project(truth, camera), axis=-1).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--inputs', type=Path, help='Reuse an exact saved reconstruction from this probe')
    args = parser.parse_args(); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(2.5*2**30/torch.cuda.get_device_properties(0).total_memory)
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e111_ee_probe_README.md', out/'README.md')
    original = json.loads((ROOT/'test_k10/config.json').read_text())
    config = dict(status='running', source_rollout=str(ROOT/'test_k10'), checkpoint=str(ROOT/'test_k10/checkpoint.pt'),
        checkpoint_sha256=original['checkpoint_sha256'], initial_context=str(INITIAL), cases=original['cases'],
        interventions='fixed visual inputs; initial/shift45 joints; circular shift45 native query control',
        training=original['training'], inputs_source=str(args.inputs) if args.inputs else None,
        input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest() if args.inputs else None,
        context_prediction_batch=1, source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path('OATFlow/shuffle_bc/trackon_model.py'),Path('OATFlow/shuffle_bc/trackon.py'))})
    write_json(out/'config.json', config); write_json(out/'progress.json', dict(status='loading frozen perception'))
    for name in config['source_sha256']:
        p=Path(name).resolve(); dest=out/'source'/p.relative_to(Path.cwd())
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    camera = top_camera(DATA)
    data=[]; started=time.perf_counter()
    if args.inputs:
        with np.load(args.inputs) as inputs:
            for i in range(len(original['cases'])):
                prefix=f'{i}_'
                data.append({k[len(prefix):]:torch.from_numpy(inputs[k].copy()) for k in inputs.files if k.startswith(prefix)})
        for d in data:
            assert d['top'].shape==(90,256) and d['wrist'].shape==(90,64,768)
            assert d['joints'].shape==(90,6)
            np.testing.assert_array_equal(d['frame'].numpy(),np.arange(0,900,10))
        # Earlier reconstruction encoded side RGB in batches of16; match live batch1 kernels.
        wrist=FrozenWristViT().cuda()
        visual=visual_model(DATA/'scene.xml',object_bodies=NAMES)
        visual.vis.global_.offwidth=visual.vis.global_.offheight=320
        display=mujoco.MjData(visual);option=mujoco.MjvOption();option.geomgroup[3]=0
        with mujoco.Renderer(visual,height=320,width=320) as renderer:
            for row,d in zip(original['cases'],data):
                features=[]
                with np.load(ROOT/'test_k10'/(row['path'].replace('/','_')+'.npz')) as trace:
                    for frame in tqdm(d['frame'].tolist(),desc='Match original side batch1',mininterval=5,leave=False):
                        display.qpos[:]=trace['qpos'][frame];mujoco.mj_forward(visual,display)
                        renderer.update_scene(display,camera='cam_left',scene_option=option)
                        rgb=torch.from_numpy(renderer.render().copy()).permute(2,0,1)[None].cuda()
                        with torch.autocast('cuda',dtype=torch.bfloat16):features.append(wrist(rgb).half().cpu())
                d['wrist']=torch.cat(features)
        del wrist;torch.cuda.empty_cache()
    else:
        tracker = FrozenTrackOn().cuda(); wrist = FrozenWristViT().cuda()
        physics=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
        visual=visual_model(DATA/'scene.xml', object_bodies=NAMES)
        visual.vis.global_.offwidth=visual.vis.global_.offheight=320
        option=mujoco.MjvOption(); option.geomgroup[3]=0
        with mujoco.Renderer(visual,height=320,width=320) as renderer:
            for row in tqdm(original['cases'],desc='EE bias probe cases',mininterval=5):
                write_json(out/'progress.json',dict(status='reconstructing native track',case=row['path']))
                with np.load(ROOT/'test_k10'/(row['path'].replace('/','_')+'.npz')) as trace:
                    data.append(reconstruct(tracker,wrist,row,trace,camera,physics,visual,renderer,option))
        del tracker,wrist;torch.cuda.empty_cache()
    arrays={f'{i}_{k}':v.numpy() for i,d in enumerate(data) for k,v in d.items()}
    np.savez_compressed(out/'inputs.npz',**arrays)
    summaries=[]; details={}; plotted={}
    for label, checkpoint in (('e110_before_BC',INITIAL),('e111_step10600',ROOT/'test_k10/checkpoint.pt')):
        saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        model=TrackOnContext()
        if label.startswith('e110'): model.load_context_state(saved['model'])
        else: model.load_state_dict(saved['model'],strict=True)
        model.cuda().eval();del saved
        for row,d in zip(original['cases'],data):
            pred=predictions(model,d); truth=d['truth'].numpy();ee=d['ee'].numpy()
            home=d['joints'][:1].expand_as(d['joints']); shifted=d['joints'].roll(45,0)
            home_pred=predictions(model,d,joints=home);shift_pred=predictions(model,d,joints=shifted)
            query_pred=predictions(model,d,top=d['top'].roll(45,0))
            delta=shift_pred-pred;ee_delta=np.roll(ee,45,0)-ee
            xy_delta=delta[:,:2];xy_ee=ee_delta[:,:2]
            valid=np.linalg.norm(xy_ee,axis=-1)>.03
            slopes=np.sum(xy_delta[valid]*xy_ee[valid],axis=-1)/np.sum(xy_ee[valid]**2,axis=-1)
            cosine=np.sum(xy_delta[valid]*xy_ee[valid],axis=-1)/(np.linalg.norm(xy_delta[valid],axis=-1)*np.linalg.norm(xy_ee[valid],axis=-1)+1e-12)
            result=dict(model=label,case=row['path'],**distances(pred,truth,ee,camera),
                mean_XYZ_shift_joint_swap_mm=float(np.linalg.norm(delta,axis=-1).mean()*1000),
                mean_XYZ_shift_home_joint_mm=float(np.linalg.norm(home_pred-pred,axis=-1).mean()*1000),
                mean_XYZ_shift_query_swap_mm=float(np.linalg.norm(query_pred-pred,axis=-1).mean()*1000),
                joint_swap_EE_XY_follow_gain_median=float(np.median(slopes)),
                joint_swap_EE_XY_direction_cosine_mean=float(cosine.mean()),joint_swap_valid_points=int(valid.sum()))
            if label.startswith('e111'):
                parity=np.linalg.norm(pred-d['saved'].numpy(),axis=-1)*1000
                result.update(reconstruction_max_difference_mm=float(parity.max()),reconstruction_mean_difference_mm=float(parity.mean()))
                # Re-encoded low-precision visual features need not be bit identical.
                # Record the actual discrepancy; keep it small relative to the diagnostic shifts.
                assert parity.max()<10. and parity.mean()<1.,f'Reconstruction mismatch: {parity.max()}mm'
            native=d['native_uv'].numpy()
            result['native_mean_error_GT_px']=float(np.linalg.norm(native-project(truth,camera),axis=-1).mean())
            result['native_mean_error_EE_px']=float(np.linalg.norm(native-project(ee,camera),axis=-1).mean())
            key=label+':'+row['path'];details[key]=dict(prediction=pred.tolist(),joint_home=home_pred.tolist(),
                joint_shift=shift_pred.tolist(),query_shift=query_pred.tolist(),result=result)
            summaries.append(result);plotted[key]=pred
            print(json.dumps(result),flush=True)
        del model;torch.cuda.empty_cache()
    # Audit persisted XYZ labels separately from model behavior, without reading RGB.
    rows=json.loads((ROOT/'train/train_episodes.json').read_text()); cached=np.load(ROOT/'train/train_xyz.npy',mmap_mode='r')
    audits=[];offset=0
    for row in tqdm(rows,desc='Verify cached target labels',mininterval=5):
        nc=row['context_frames'];frames=np.r_[np.unique(np.r_[np.arange(0,nc,10),nc-1]), nc+np.arange(0,row['expert_frames']-1,3)]
        truth=[];ee=[]
        for ref,chosen in ((row['context_path'],frames[frames<nc]),(row['path'],frames[frames>=nc]-nc)):
            with np.load(DATA/ref/'labels.npz') as z:truth.append(z['entity_xyz_base_m'][chosen,row['target_id']].astype(np.float32))
            with np.load(DATA/ref/'observation.npz') as z:ee.append(z['ee_position_base_m'][chosen])
        truth=np.concatenate(truth);ee=np.concatenate(ee)
        np.testing.assert_array_equal(cached[offset:offset+len(frames)],truth)
        distance=np.linalg.norm(truth-ee,axis=-1)*1000
        expert=frames>=nc
        audits.append(dict(points=len(frames),expert_points=int(expert.sum()),
            expert_EE_within20mm=int((distance[expert]<20).sum()),context_EE_within20mm=int((distance[~expert]<20).sum())))
        offset+=len(frames)
    assert offset==len(cached)
    fig,axes=plt.subplots(4,3,figsize=(15,11),layout='constrained')
    for i,(row,d) in enumerate(zip(original['cases'],data)):
        t=d['frame'].numpy()/25;truth=d['truth'].numpy();ee=d['ee'].numpy()
        p0=plotted['e110_before_BC:'+row['path']];p1=plotted['e111_step10600:'+row['path']]
        ax=axes[i,0]
        for points,color,name in ((ee,'#d68a23','EE'),(truth,'#238b45','target GT'),(p0,'#2878b5','before BC'),(p1,'#cb328f','after BC')):
            ax.plot(points[:,0]*1000,points[:,1]*1000,color=color,label=name,linewidth=1.3)
        ax.set_title(row['evaluation_group']+' / '+row['path'],fontsize=10);ax.set_xlabel('Base X (mm)');ax.set_ylabel('Base Y (mm)');ax.set_aspect('equal',adjustable='datalim')
        ax=axes[i,1]
        ax.plot(t,np.linalg.norm(p0-truth,axis=-1)*1000,color='#2878b5',label='Before BC → GT')
        ax.plot(t,np.linalg.norm(p1-truth,axis=-1)*1000,color='#cb328f',label='After BC → GT')
        ax.plot(t,np.linalg.norm(p1-ee,axis=-1)*1000,color='#d68a23',label='After BC → EE',alpha=.8)
        ax.set_xlabel('Rollout time (s)');ax.set_ylabel('3D distance (mm)')
        ax=axes[i,2]
        a=np.array(details['e111_step10600:'+row['path']]['joint_shift']);b=np.array(details['e110_before_BC:'+row['path']]['joint_shift'])
        ax.plot(t,np.linalg.norm(a-p1,axis=-1)*1000,color='#cb328f',label='After BC')
        ax.plot(t,np.linalg.norm(b-p0,axis=-1)*1000,color='#2878b5',label='Before BC')
        ax.set_xlabel('Rollout time (s)');ax.set_ylabel('XYZ shift: joint-only swap (mm)')
        for ax in axes[i]:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    for ax in axes[0]:ax.legend(fontsize=7,loc='best')
    fig.suptitle('Does target XYZ follow the end effector? | Fixed visual inputs, joint-state interventions',fontsize=14)
    fig.savefig(out/'diagnostic.png',dpi=180);fig.savefig(out/'diagnostic.pdf');plt.close(fig)
    summary=dict(status='complete',points=sum(len(d['top']) for d in data),cases=summaries,
        cached_XYZ_labels_exact=True,labels_checked=offset,
        expert_EE_within20mm_fraction=sum(x['expert_EE_within20mm'] for x in audits)/sum(x['expert_points'] for x in audits),
        context_EE_within20mm_fraction=sum(x['context_EE_within20mm'] for x in audits)/sum(x['points']-x['expert_points'] for x in audits),
        elapsed_s=time.perf_counter()-started,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(out/'point_metrics.json',details);write_json(out/'summary.json',summary);write_json(out/'progress.json',summary)
    config['status']='complete';write_json(out/'config.json',config)
    with (out/'README.md').open('a') as stream:stream.write('\nActual result: '+json.dumps(summary)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
