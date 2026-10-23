"""Mask IoU, task-identity checks and same-image four-query visualizations."""
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image,ImageDraw
import torch
from tqdm import tqdm

from OATFlow.task_mask.data import batch,cache,query_index,read,ROOT
from OATFlow.task_mask.model import overlay
from OATFlow.top_pick.cloud_data import write

NAMES=('Red','Blue','Green','Yellow')


@torch.inference_mode()
def evaluate(model,cached,index,description,save=None):
    model.eval();records=[];predictions=[]
    for start in tqdm(range(0,len(index),8),desc=description,mininterval=10):
        chosen=list(range(start,min(start+8,len(index))))
        features,target,truth,instance=batch(cached,index,chosen)
        device=model.task_prompt.weight.device
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            logits,quality=model(features.to(device),target.to(device))
        prediction=(logits>0).cpu();quality=quality.float().cpu()
        for j,point in enumerate(chosen):
            pred=prediction[j];gt=truth[j]
            intersection=int((pred & gt).sum());union=int((pred | gt).sum())
            pixels=int(gt.sum());pred_pixels=int(pred.sum())
            iou=intersection/union if union else 1.
            scores=[]
            for color in range(4):
                mask=instance[j]==color+1
                n=int((pred & mask).sum());d=int((pred | mask).sum())
                scores.append(n/d if d else 0.)
            selected=int(np.argmax(scores)) if max(scores)>0 else 4
            records.append(dict(**index[point],iou=iou,
                dice=2*intersection/(pixels+pred_pixels) if pixels+pred_pixels else 1.,
                visible_pixels=pixels,predicted_pixels=pred_pixels,intersection=intersection,union=union,
                identity=selected,correct_identity=selected==int(target[j]),quality=float(quality[j])))
            if save is not None:predictions.append(np.packbits(pred.cpu().numpy(),axis=-1,bitorder='little'))
    visible=[r for r in records if r['visible_pixels']>=8]
    def aggregate(current):
        current=[r for r in current if r['visible_pixels']>=8]
        if not current:return dict(visible_cases=0)
        return dict(visible_cases=len(current),mean_iou=float(np.mean([r['iou'] for r in current])),
            mean_dice=float(np.mean([r['dice'] for r in current])),
            identity_accuracy=float(np.mean([r['correct_identity'] for r in current])),
            iou_ge_50=float(np.mean([r['iou']>=.5 for r in current])),
            iou_ge_75=float(np.mean([r['iou']>=.75 for r in current])))
    summary=dict(cases=len(records),**aggregate(records),
        low_visibility_cases=len(records)-len(visible),
        by_group={group:aggregate([r for r in records if r['group']==group]) for group in sorted({r['group'] for r in records})},
        by_target={NAMES[t]:aggregate([r for r in records if r['target']==t]) for t in range(4)},
        mask_threshold=.5,visible_metric_min_pixels=8)
    confusion=np.zeros((4,5),int)
    for r in visible:confusion[r['target'],r['identity']]+=1
    summary['identity_confusion']=dict(rows=list(NAMES),columns=[*NAMES,'no_object_overlap'],counts=confusion.tolist())
    if save is not None:
        (save/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        np.savez_compressed(save/'predictions.npz',packed=np.asarray(predictions),bitorder='little')
        write(save/'summary.json',summary)
    return summary,records,predictions


def sample_sheet(cached,index,records,packed,out,filename='samples.png',worst=False):
    # Every selected source image is queried with all four target IDs.
    candidates=[]
    for image in sorted({r['image'] for r in records}):
        current=[(i,r) for i,r in enumerate(records) if r['image']==image]
        if len(current)==4 and {r['target'] for _,r in current}==set(range(4)):
            candidates.append((image,current))
    selected=[]
    for group in sorted({r['group'] for r in records}):
        current=[item for item in candidates if item[1][0][1]['group']==group]
        if current:
            selected.append(min(current,key=lambda item:np.mean([r['iou'] for _,r in item[1]])) if worst else current[0])
    if not selected:return
    sheet=Image.new('RGB',(1600,280*len(selected)),'white');draw=ImageDraw.Draw(sheet)
    for row,(image,current) in enumerate(selected):
        y=row*280;rgb=cached['images'][image]
        draw.text((8,y+5),f"{current[0][1]['group']} / {cached['jobs'][image]['path']} / frame{current[0][1]['frame']}",fill='black')
        sheet.paste(Image.fromarray(rgb),(0,y+36))
        for point,r in sorted(current,key=lambda item:item[1]['target']):
            target=r['target'];truth=cached['instances'][image].numpy()==target+1
            pred=np.unpackbits(packed[point],axis=-1,count=320,bitorder='little').astype(bool)
            x=(target+1)*320
            draw.text((x+6,y+17),f"{NAMES[target]} | IoU {r['iou']:.3f}",fill='black')
            sheet.paste(overlay(rgb,truth,pred),(x,y+36))
    sheet.save(out/filename)


def extra_jobs(rows):
    chosen=[]
    for group in ('composition','spatial','combined'):
        for target in range(4):
            candidates=[r for r in rows if r['evaluation_group']==group and r['target_id']==target]
            chosen.extend(candidates[:2])
    jobs=[]
    for row in chosen:
        with np.load(Path('/workspace/robotics_assets/datasets/pick_wide_rgbd')/row['path']/'supervision.npz') as z:
            phase=z['phase'].copy()
        for stage in ('approach','engage','lift'):
            found=np.flatnonzero(np.char.endswith(phase,'_'+stage));assert len(found)
            frame=int(found[-1])
            queries=[dict(row,target_id=t,path=row['path']+f'/f{frame}/q{t}',evaluation_group=group)
                     for t in range(4)]
            # Keep each original split group; phase is metadata, never a model input.
            for query in queries:query['evaluation_group']=row['evaluation_group']+'_'+stage
            jobs.append(dict(path=row['path'],frame=frame,rows=queries,stage=stage))
    assert len(jobs)==72
    return jobs


def outside_cache(model):
    import mujoco
    from OATFlow.data_tools.collect_shuffle import NAMES as BODIES
    from OATFlow.data_tools.render_dynamic import clip_planes
    from OATFlow.simulation.render import visual_model
    from OATFlow.top_pick.cloud import DATA
    source=ROOT.parent/'e122_full/test_xy'
    rows=list(map(json.loads,(source/'manifest.jsonl').read_text().splitlines()))
    physics=visual_model(DATA/'scene.xml',object_bodies=BODIES,lighting_scene=DATA/'scene.xml');clip_planes(physics)
    physics.vis.global_.offwidth=320;physics.vis.global_.offheight=240
    state=mujoco.MjData(physics);option=mujoco.MjvOption();option.geomgroup[3]=0
    features=[];images=[];instances=[];jobs=[]
    with mujoco.Renderer(physics,height=240,width=320) as renderer:
        for row in tqdm(rows,desc='Encode retained XY test observations',mininterval=10):
            with np.load(source/'states'/row['path']/'sim_state.npz') as z:state.qpos[:]=z['initial_qpos']
            mujoco.mj_forward(physics,state);renderer.update_scene(state,camera='cam_front',scene_option=option)
            rgb=renderer.render().copy()
            renderer.enable_segmentation_rendering();labels=renderer.render();renderer.disable_segmentation_rendering()
            instance=np.zeros((240,320),np.uint8)
            for target,name in enumerate(BODIES[:4]):
                geom=np.flatnonzero(physics.geom_bodyid==physics.body(name).id)
                instance[np.isin(labels[...,0],geom)&(labels[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM))]=target+1
            features.append(model.encode(rgb)[0].cpu().numpy().astype(np.float16));images.append(rgb);instances.append(instance)
            queries=[dict(path=row['path']+f'/q{t}',target_id=t,evaluation_group=row['edge']+'_'+row['evaluation_group'],
                          layout_id=row['layout_id'],permutation=0) for t in range(4)]
            jobs.append(dict(path=row['path'],frame=0,rows=queries))
    return dict(features=torch.from_numpy(np.stack(features)),images=np.stack(images),
                instances=torch.from_numpy(np.stack(instances)),jobs=jobs)


def run_tests(model,manifests,config):
    test=ROOT/'test';test.mkdir(exist_ok=False)
    note=Path('docs/research/e123_test_README.md');shutil.copyfile(note,test/'README.md')
    model.eval();all_summaries={}
    from OATFlow.task_mask.data import groups
    for name,jobs in (('home',groups(manifests['test'])),('dynamic',extra_jobs(manifests['test'])),('xy',None)):
        out=test/name;out.mkdir();shutil.copyfile(note,out/'README.md')
        cached=outside_cache(model) if jobs is None else cache(jobs,'Cache test '+name)
        index=query_index(cached)
        summary,records,packed=evaluate(model,cached,index,'Test '+name,save=out)
        sample_sheet(cached,index,records,packed,out)
        sample_sheet(cached,index,records,packed,out,filename='worst.png',worst=True)
        write(out/'config.json',dict(status='complete',test_name=name,checkpoint=str(ROOT/'best.pt'),
              checkpoint_sha256=config['best_checkpoint_sha256'],cases=len(index),input='RGB+learned task token only',
              sampling='all original heldout home queries' if name=='home' else 'fixed diagnostic samples;not all trajectory frames'))
        all_summaries[name]=summary;del cached
        print(json.dumps(dict(test=name,**summary)),flush=True)
    write(test/'summary.json',all_summaries)
    text=note.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\n'+json.dumps(all_summaries)+'\n'
    note.write_text(text);(test/'README.md').write_text(text)
    for name in all_summaries:(test/name/'README.md').write_text(text)
    return all_summaries
