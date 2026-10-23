"""Evaluate unmodified MobileSAM automatic mask proposals on a fixed pick pilot."""
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image,ImageDraw
from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm
from mobile_sam import sam_model_registry,SamAutomaticMaskGenerator

from OATFlow.task_mask.data import read
from OATFlow.task_mask.model import WEIGHTS,overlay
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.features import sha

ROOT=Path('/workspace/robotics_assets/experiments/e126_auto')
NOTE=Path('docs/research/e126_README.md')
NAMES=('Red','Blue','Green','Yellow')
COLORS=((240,70,70),(60,120,245),(50,190,80),(240,210,30))


def selected_jobs():
    rows=[json.loads(s) for s in (ROOT.parent/'e124_mask_full/val_manifest.jsonl').read_text().splitlines()]
    chosen=[]
    for layout in (32,33,34,35):
        for target in range(4):
            candidates=sorted([r for r in rows if r['layout_id']==layout and r['target_id']==target],key=lambda r:r['permutation'])
            chosen.extend(candidates[:2])
    assert len(chosen)==32
    jobs=[]
    for row in chosen:
        with np.load(DATA/row['path']/'supervision.npz') as z:phase=z['phase']
        indices=[('home',0)]
        for stage in ('approach','engage','lift'):
            found=np.flatnonzero(np.char.endswith(phase,'_'+stage));assert len(found)
            indices.append((stage,int(found[-1])))
        assert len(set(f for _,f in indices))==4
        jobs.append(dict(row=row,frames=indices))
    return jobs


def proposal_metrics(proposals,instance):
    masks=np.stack([a['segmentation'] for a in proposals]) if proposals else np.empty((0,240,320),bool)
    truth=np.stack([instance==t+1 for t in range(4)])
    visible=truth.sum((1,2))>=8
    intersections=np.einsum('nhw,thw->nt',masks.astype(np.int32),truth.astype(np.int32),optimize=True)
    areas=masks.sum((1,2));gt_areas=truth.sum((1,2))
    unions=areas[:,None]+gt_areas[None]-intersections
    iou=np.divide(intersections,unions,out=np.zeros_like(intersections,dtype=float),where=unions>0)
    best=iou.argmax(axis=0) if len(masks) else np.full(4,-1)
    target=[]
    for t in range(4):
        idx=int(best[t]);inter=int(intersections[idx,t]) if idx>=0 else 0
        target.append(dict(target=t,visible_pixels=int(gt_areas[t]),visible=bool(visible[t]),
            best_mask=idx,best_iou=float(iou[idx,t]) if idx>=0 else 0.,
            precision=inter/int(areas[idx]) if idx>=0 and areas[idx]>0 else 0.,
            recall=inter/int(gt_areas[t]) if gt_areas[t]>0 else 0.,
            proposals_iou50=int((iou[:,t]>=.5).sum()),proposals_iou75=int((iou[:,t]>=.75).sum())))
    # Coverage needs distinct proposals; maximize matched count before total IoU.
    recalls={}
    ids=np.flatnonzero(visible)
    for threshold in (.5,.75):
        sub=iou[:,ids]
        rows,cols=linear_sum_assignment(-((sub>=threshold).astype(float)*10+sub))
        count=int((sub[rows,cols]>=threshold).sum())
        recalls[str(threshold)]=dict(matched=count,total=len(ids),all_visible_covered=count==len(ids),
            all_four_covered=len(ids)==4 and count==4)
    maximum=iou[:,visible].max(axis=1) if visible.any() else np.zeros(len(masks))
    return masks,dict(targets=target,coverage=recalls,proposal_count=len(masks),
        proposals_with_object_iou50=int((maximum>=.5).sum()),
        other_or_partial_proposals=int((maximum<.5).sum()))


def figures(rgb,instance,proposals,metrics,out):
    out.mkdir();shutil.copyfile(NOTE,out/'README.md')
    canvas=rgb.astype(np.float32).copy();pen_image=Image.fromarray(rgb);pen=ImageDraw.Draw(pen_image)
    rng=np.random.default_rng(126)
    # Large masks first, small masks last; preserve IDs for the proposal gallery.
    for i in sorted(range(len(proposals)),key=lambda j:proposals[j]['area'],reverse=True):
        mask=proposals[i]['segmentation'];color=rng.uniform(35,245,3)
        canvas[mask]=canvas[mask]*.55+color*.45
    im=Image.fromarray(canvas.clip(0,255).astype(np.uint8));draw=ImageDraw.Draw(im)
    for i,a in enumerate(proposals):
        x,y,w,h=a['bbox'];draw.text((int(x+w/2),int(y+h/2)),str(i),fill='white',stroke_width=1,stroke_fill='black')
    im.resize((960,720)).save(out/'all_masks.png')
    sheet=Image.new('RGB',(1600,290),'white');draw=ImageDraw.Draw(sheet)
    sheet.paste(Image.fromarray(rgb),(0,40));draw.text((5,15),'RGB; no input points/boxes/masks',fill='black')
    for t,r in enumerate(metrics['targets']):
        pred=proposals[r['best_mask']]['segmentation'] if r['visible'] and r['best_mask']>=0 else np.zeros((240,320),bool)
        image=overlay(rgb,instance==t+1,pred)
        sheet.paste(image,((t+1)*320,40))
        label=f'{NAMES[t]} | best #{r["best_mask"]} | IoU {r["best_iou"]:.3f}' if r['visible'] else f'{NAMES[t]} | <8visible pixels; excluded'
        draw.text(((t+1)*320+4,15),label,fill='black')
    sheet.save(out/'object_coverage.png')
    columns=6;tile_w=200;tile_h=180;rows=max(1,int(np.ceil(len(proposals)/columns)))
    gallery=Image.new('RGB',(columns*tile_w,rows*tile_h),'white');draw=ImageDraw.Draw(gallery)
    for i,a in enumerate(proposals):
        x,y,w,h=a['bbox'];box=(max(0,int(x)-8),max(0,int(y)-8),min(320,int(x+w)+9),min(240,int(y+h)+9))
        image=overlay(rgb,np.zeros((240,320),bool),a['segmentation']).crop(box)
        image.thumbnail((196,140));col=i%columns;row=i//columns
        gallery.paste(image,(col*tile_w+(tile_w-image.width)//2,row*tile_h+35))
        draw.text((col*tile_w+4,row*tile_h+3),f'#{i} | q={a["predicted_iou"]:.2f} | px={a["area"]}',fill='black')
    gallery.save(out/'proposals.png')


def summarize(results):
    def part(current):
        targets=[t for r in current for t in r['targets'] if t['visible']]
        return dict(images=len(current),visible_objects=len(targets),
            mean_best_proposal_iou=float(np.mean([t['best_iou'] for t in targets])),
            mean_proposals=float(np.mean([r['proposal_count'] for r in current])),
            mean_other_or_partial_proposals=float(np.mean([r['other_or_partial_proposals'] for r in current])),
            distinct_recall_50=sum(r['coverage']['0.5']['matched'] for r in current)/len(targets),
            distinct_recall_75=sum(r['coverage']['0.75']['matched'] for r in current)/len(targets),
            all_visible_recalled_50=sum(r['coverage']['0.5']['all_visible_covered'] for r in current),
            all_visible_recalled_75=sum(r['coverage']['0.75']['all_visible_covered'] for r in current),
            frames_all_four_visible=sum(all(t['visible'] for t in r['targets']) for r in current),
            all_four_recalled_50=sum(r['coverage']['0.5']['all_four_covered'] for r in current),
            all_four_recalled_75=sum(r['coverage']['0.75']['all_four_covered'] for r in current))
    return dict(**part(results),by_phase={p:part([r for r in results if r['phase']==p]) for p in ('home','approach','engage','lift')},
        proposal_min=min(r['proposal_count'] for r in results),proposal_max=max(r['proposal_count'] for r in results))


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(126)
    ROOT.mkdir(exist_ok=False);shutil.copyfile(NOTE,ROOT/'README.md')
    jobs=selected_jobs();write(ROOT/'selected_jobs.json',jobs)
    model=sam_model_registry['vit_t'](checkpoint=str(WEIGHTS)).cuda().eval();model.requires_grad_(False)
    source_hash=sha(WEIGHTS)
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    settings=dict(points_per_side=32,points_per_batch=64,pred_iou_thresh=.88,stability_score_thresh=.95,
                  stability_score_offset=1.,box_nms_thresh=.7,crop_n_layers=0,crop_nms_thresh=.7,min_mask_region_area=0)
    generator=SamAutomaticMaskGenerator(model,**settings)
    config=dict(status='running',experiment=ROOT.name,pretrained=str(WEIGHTS),pretrained_sha256=source_hash,
        source_sha256=sha(Path(__file__)),input='RGB only;uniform fixed grid,NO GT-derived prompt,task ID,mask,box or depth',
        original_generator_settings=settings,device='CUDA FP32',images=128,episodes=32,
        selection='4val layouts x4source-trajectory targets x first2permutations;home,last approach,last engage,last lift',
        model='entire original MobileSAM checkpoint frozen;no E123/E124 weights;no training or optimizer',
        evaluation='GT only matches proposals offline;distinct matching for recall@IoU0.5/0.75;proposal recall is not task-selection accuracy',
        actions=None,H=None,K=None,training_epochs=0,learning_rate=None)
    write(ROOT/'config.json',config);(ROOT/'masks').mkdir();started=time.perf_counter();results=[]
    bar=tqdm(jobs,desc='MobileSAM automatic masks',mininterval=10)
    for job in bar:
        frames=[f for _,f in job['frames']];images,instances=read(DATA/job['row']['path'],frames)
        for (phase,frame),rgb,instance in zip(job['frames'],images,instances,strict=True):
            torch.cuda.synchronize();t=time.perf_counter();proposals=generator.generate(rgb);torch.cuda.synchronize()
            seconds=time.perf_counter()-t;masks,metrics=proposal_metrics(proposals,instance)
            i=len(results);record=dict(image=i,path=job['row']['path'],frame=frame,phase=phase,
                source_target=job['row']['target_id'],layout=job['row']['layout_id'],seconds=seconds,**metrics)
            results.append(record)
            np.savez_compressed(ROOT/'masks'/f'{i:03d}.npz',packed=np.packbits(masks,axis=-1,bitorder='little'),bitorder='little')
            write(ROOT/'masks'/f'{i:03d}.json',[{k:v for k,v in a.items() if k!='segmentation'} for a in proposals])
            with (ROOT/'results.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if i<4:figures(rgb,instance,proposals,metrics,ROOT/f'sample_{phase}')
            elapsed=time.perf_counter()-started
            progress=dict(status='running',images=len(results),total_images=128,elapsed_s=elapsed,ETA_s=elapsed/len(results)*(128-len(results)),
                          last_proposals=len(proposals),last_inference_s=seconds)
            write(ROOT/'progress.json',progress)
            if len(results)%8==0:print(json.dumps(progress),flush=True)
        bar.set_postfix(masks=len(proposals),inference=f'{seconds:.2f}s')
    assert len(results)==128 and sha(WEIGHTS)==source_hash
    assert all(torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items())
    summary=summarize(results);write(ROOT/'summary.json',summary)
    worst=min(results,key=lambda r:(r['coverage']['0.75']['matched'],-r['coverage']['0.75']['total'],sum(t['best_iou'] for t in r['targets'] if t['visible'])))
    rgb,instance=read(DATA/worst['path'],[worst['frame']]);proposals=[]
    with np.load(ROOT/'masks'/f'{worst["image"]:03d}.npz') as z:masks=np.unpackbits(z['packed'],axis=-1,count=320,bitorder='little').astype(bool)
    meta=json.loads((ROOT/'masks'/f'{worst["image"]:03d}.json').read_text())
    for mask,a in zip(masks,meta,strict=True):proposals.append(dict(a,segmentation=mask))
    figures(rgb[0],instance[0],proposals,worst,ROOT/'worst')
    config.update(status='complete',elapsed_s=time.perf_counter()-started);write(ROOT/'config.json',config)
    write(ROOT/'verification.json',dict(status='passed',original_model_unchanged=True,checkpoint_file_unchanged=True,
        generator_received_RGB_only=True,uniform_grid_not_GT_conditioned=True,images=128,proposal_masks_preserved=True))
    write(ROOT/'progress.json',dict(status='complete',summary=summary,elapsed_s=config['elapsed_s']))
    text=NOTE.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nResults:\n'+json.dumps(summary,indent=2)+'\n'
    NOTE.write_text(text);(ROOT/'README.md').write_text(text)
    print(json.dumps(dict(status='complete',summary=summary)),flush=True)


if __name__=='__main__':main()
