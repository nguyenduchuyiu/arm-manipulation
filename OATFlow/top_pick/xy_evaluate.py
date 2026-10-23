"""Paired E122 K15 grasp tests just outside versus inside the train XY rectangle."""
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

from controllers.oracle_pick import GraspFailure, IKFailure
from OATFlow.data_tools.collect_shuffle import Scene, NAMES, PERMS
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.pick.scene import execute
from OATFlow.simulation.render import visual_model
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.cloud_evaluate import rollout, summarize
from OATFlow.top_pick.collect import setup
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.full_train import ROOT
from OATFlow.top_pick.patch_fusion import PatchFusionPolicy
from OATFlow.top_pick.wide import LOW, HIGH

OUT = ROOT/'test_xy'
NOTE = Path('docs/research/e122_xy_README.md')


def train_xy(physics):
    rows = [json.loads(s) for s in (ROOT/'train_manifest.jsonl').read_text().splitlines()]
    points = []
    for layout in tqdm(sorted({r['layout_id'] for r in rows}), desc='Audit actual train XY'):
        row = next(r for r in rows if r['layout_id'] == layout)
        scene = Scene(physics)
        with np.load(DATA/row['path']/'sim_state.npz') as z:
            scene.data.qpos[:] = z['initial_qpos']
        mujoco.mj_forward(physics, scene.data)
        points.extend(scene.target_positions()[name][:2].copy() for name in NAMES[:4])
    points = np.asarray(points)
    assert ((points >= LOW-1e-5) & (points <= HIGH+1e-5)).all()
    np.save(OUT/'training_xy.npy', points)
    write(OUT/'training_xy_audit.json', dict(layouts=32, centers=len(points),
          actual_min=points.min(0).tolist(), actual_max=points.max(0).tolist(),
          sampler_low=LOW.tolist(), sampler_high=HIGH.tolist(), frame='simulator world'))
    return points


def prepare(physics, visual, renderer, option):
    rng = np.random.default_rng(122031)
    rows, rejected, images = [], [], []
    display = mujoco.MjData(visual)
    # Keep three distractors in a shared interior arrangement for each candidate pair.
    for edge_index, (edge, axis, side) in enumerate((('x_low',0,-1),('x_high',0,1))):
        boundary = LOW[axis] if side == -1 else HIGH[axis]
        for target in tqdm(range(4), desc=f'Expert validation {edge}'):
            pair = edge_index*4+target
            for attempt in range(100):
                positions = rng.uniform(LOW+.065, HIGH-.065, (4,2))
                tangent = rng.uniform(LOW[1-axis]+.065, HIGH[1-axis]-.065)
                positions[target,1-axis] = tangent
                yaw = rng.uniform(-np.pi,np.pi,4)
                current, pictures = [], []
                try:
                    for group, direction in (('outside',1),('inside',-1)):
                        xy = positions.copy();xy[target,axis] = boundary+side*direction*.03
                        if min(np.linalg.norm(a-b) for i,a in enumerate(xy) for b in xy[i+1:]) < .055:
                            raise ValueError('object center separation below55mm')
                        scene = Scene(physics)
                        initial = setup(scene, dict(positions=xy,yaw=yaw), 0)
                        actual = np.stack(list(scene.target_positions().values()))
                        delta = side*(actual[target,axis]-boundary)
                        if not (delta > .028 if group == 'outside' else delta < -.028):
                            raise ValueError('target drifted from planned boundary offset')
                        display.qpos[:] = initial[0];mujoco.mj_forward(visual,display)
                        renderer.update_scene(display,camera='cam_front',scene_option=option)
                        image = renderer.render().copy()
                        renderer.enable_segmentation_rendering();labels=renderer.render();renderer.disable_segmentation_rendering()
                        geoms = np.flatnonzero(visual.geom_bodyid == visual.body(NAMES[target]).id)
                        pixels = int((np.isin(labels[...,0],geoms) & (labels[...,1] == int(mujoco.mjtObj.mjOBJ_GEOM))).sum())
                        if pixels < 16: raise ValueError(f'target visibility {pixels}px')
                        expert = execute(scene,NAMES[target],lambda phase:None)
                        row = dict(path=f'{edge}/t{target}/{group}',evaluation_group=group,
                            layout_id=pair,permutation=0,target_id=target,target_slot=PERMS[0].index(target),
                            edge=edge,pair=pair,attempt=attempt,target_xy=actual[target,:2].tolist(),
                            all_object_xyz=actual.tolist(),visible_target_pixels=pixels,
                            outside_distance_m=max(0.,float(delta)),expert=expert)
                        current.append((row,initial));pictures.append(image)
                except (GraspFailure, IKFailure, ValueError) as error:
                    rejected.append(dict(pair=pair,edge=edge,target=target,attempt=attempt,error=str(error)))
                    write(OUT/'rejections.json',rejected)
                    continue
                for (row,initial), image in zip(current,pictures):
                    path=OUT/'states'/row['path'];path.mkdir(parents=True)
                    np.savez_compressed(path/'sim_state.npz',**dict(zip(
                        ('initial_qpos','initial_qvel','initial_ctrl'),initial)))
                    Image.fromarray(image).save(path/'initial.png')
                    rows.append(row);images.append(image)
                break
            else: raise RuntimeError(f'No feasible matched pair: {edge} target{target}')
    assert len(rows)==16
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(OUT/'rejections.json',rejected)
    sheet=Image.new('RGB',(640,len(rows)//2*264),'white');draw=ImageDraw.Draw(sheet)
    for i,(row,image) in enumerate(zip(rows,images)):
        x=(i%2)*320;y=(i//2)*264
        draw.text((x+5,y+4),f"{row['edge']} {NAMES[row['target_id']]} {row['evaluation_group']}",fill='black')
        sheet.paste(Image.fromarray(image),(x,y+24))
    sheet.save(OUT/'initial_samples.png')
    return rows


def plot(points,rows,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(*points.T,c='#aeb8c4',s=12,label='Train spawn centers (32 layouts)')
    ax.add_patch(Rectangle(LOW,*(HIGH-LOW),fill=False,color='#546475',linestyle='--',label='Training sampler support'))
    for group,color in (('inside','#168575'),('outside','#c36b29')):
        for success,marker in ((True,'o'),(False,'x')):
            chosen=[row for row,r in zip(rows,results) if row['evaluation_group']==group and r['success']==success]
            if chosen:
                xy=np.array([r['target_xy'] for r in chosen]);ax.scatter(*xy.T,color=color,marker=marker,s=70,
                    label=f'{group}: '+('success' if success else 'failure'))
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',title='E122 K15: paired XY extrapolation, CPU FP32')
    ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(OUT/'xy_results.png',dpi=180);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=False);shutil.copyfile(NOTE,OUT/'README.md')
    config=dict(status='preparing',checkpoint=str(ROOT/'train/final.pt'),checkpoint_step=23520,
        checkpoint_sha256=sha(ROOT/'train/final.pt'),cases=16,device='cpu',precision='FP32',
        H=25,K=15,flow_steps=10,noise_seed=0,frame_budget=300,policy_hz=25,servo_hz=50,physics_hz=500,
        training_low=LOW.tolist(),training_high=HIGH.tolist(),offset_m=.03,
        excluded_edges='y_low: visibility; y_high: original physical expert failed in100candidates;details in local preparation note',
        changed='only target XY across each boundary; paired distractors/yaw/color/home fixed',
        feasibility='both original physical experts succeed;target at least16visible pixels',
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('xy_evaluate.py','cloud_evaluate.py')})
    write(OUT/'config.json',config)
    physics=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
    visual=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml');clip_planes(visual)
    visual.vis.global_.offwidth=320;visual.vis.global_.offheight=240
    option=mujoco.MjvOption();option.geomgroup[3]=0
    points=train_xy(physics)
    with mujoco.Renderer(visual,height=240,width=320) as renderer:
        rows=prepare(physics,visual,renderer,option)
        saved=torch.load(ROOT/'train/final.pt',map_location='cpu',weights_only=False,mmap=True)
        assert saved['step']==23520
        config['training']=saved['config'];config['status']='running';write(OUT/'config.json',config)
        torch.set_num_threads(4)
        policy=PatchFusionPolicy(initialize=False).cpu();policy.load_state_dict(saved['model'],strict=True)
        policy.eval();del saved
        results=[];started=time.perf_counter()
        for start in tqdm(range(0,len(rows),8),desc='Matched XY pilot batches'):
            current=rollout(rows[start:start+8],policy,physics,visual,renderer,option,OUT,k=15,data_root=OUT/'states')
            for r,row in zip(current,rows[start:start+8]):
                r.update(edge=row['edge'],pair=row['pair'],target_xy=row['target_xy'])
            with (OUT/'results.jsonl').open('a') as f:
                for r in current:f.write(json.dumps(r)+'\n')
            results+=current
            progress=summarize(results,time.perf_counter()-started);write(OUT/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    summary=summarize(results,time.perf_counter()-started)
    summary.update(status='complete',by_edge={})
    for edge in sorted({r['edge'] for r in results}):
        summary['by_edge'][edge]={group:summarize([r for r in results if r['edge']==edge and r['group']==group],0)
                                  for group in ('inside','outside')}
    pairs=[(results[i],results[i+1]) for i in range(0,len(results),2)]
    assert all(a['group']=='outside' and b['group']=='inside' and a['pair']==b['pair'] for a,b in pairs)
    summary['paired']=dict(Counter('both_success' if a['success'] and b['success'] else
        'outside_only' if a['success'] else 'inside_only' if b['success'] else 'both_failure' for a,b in pairs))
    assert sha(ROOT/'train/final.pt')==config['checkpoint_sha256']
    plot(points,rows,results);write(OUT/'summary.json',summary);write(OUT/'progress.json',summary)
    config['status']='complete';write(OUT/'config.json',config)
    note='\nCOMPLETE. Exact paired pilot results:\n'+json.dumps(summary)+'\n'
    text=NOTE.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+note
    NOTE.write_text(text);(OUT/'README.md').write_text(text)
    with Path('docs/research/REPORT.md').open('a') as f:f.write('\nE122 XY extrapolation pilot:'+note)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
