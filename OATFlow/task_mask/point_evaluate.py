"""Original MobileSAM with one oracle interior point on E124's exact validation queries."""
from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,wait
from collections import defaultdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
import torch
from tqdm import tqdm
from mobile_sam import sam_model_registry,SamPredictor

from OATFlow.task_mask.data import read
from OATFlow.task_mask.model import WEIGHTS,overlay
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.cloud_data import write

ROOT=Path('/workspace/robotics_assets/experiments/e125_point')
REFERENCE=ROOT.parent/'e124_mask_full/validation/epoch_1'
NOTE=Path('docs/research/e125_README.md')
NAMES=('Red','Blue','Green','Yellow')
MODES=('single','multimask')
PREDICTOR=None
FROZEN_HASH=None


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def initialize():
    global PREDICTOR,FROZEN_HASH
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(125)
    model=sam_model_registry['vit_t'](checkpoint=str(WEIGHTS)).eval()
    model.requires_grad_(False)
    PREDICTOR=SamPredictor(model);FROZEN_HASH=state_hash(model)


def score(pred,instance,target):
    gt=instance==target+1
    intersection=int(np.count_nonzero(pred & gt));union=int(np.count_nonzero(pred | gt))
    pixels=int(gt.sum());pred_pixels=int(pred.sum())
    scores=[]
    for other in range(4):
        mask=instance==other+1
        i=int(np.count_nonzero(pred & mask));u=int(np.count_nonzero(pred | mask))
        scores.append(i/u if u else 0.)
    selected=int(np.argmax(scores)) if max(scores)>0 else 4
    return dict(iou=intersection/union if union else 1.,
        dice=2*intersection/(pixels+pred_pixels) if pixels+pred_pixels else 1.,
        precision=intersection/pred_pixels if pred_pixels else 0.,recall=intersection/pixels if pixels else 0.,
        visible_pixels=pixels,predicted_pixels=pred_pixels,intersection=intersection,union=union,
        identity=selected,correct_identity=selected==target)


@torch.inference_mode()
def episode(job):
    predictor=PREDICTOR;model=predictor.model;started=time.perf_counter()
    frames=sorted(job['queries']);images,instances=read(DATA/job['path'],frames)
    results=[];verified=False
    for frame,rgb,instance in zip(frames,images,instances,strict=True):
        current=[];points=[]
        for index,reference in job['queries'][frame]:
            mask=instance==reference['target']+1
            assert int(mask.sum())==reference['visible_pixels']
            if int(mask.sum())<8:
                results.append(dict(index=index,skipped=True,visible_pixels=int(mask.sum())))
                continue
            # Deepest interior pixel: the arithmetic centroid can lie outside an occluded mask.
            y,x=np.unravel_index(np.argmax(distance_transform_edt(mask)),mask.shape)
            assert mask[y,x]
            current.append((index,reference));points.append([float(x),float(y)])
        if not current:continue
        predictor.set_image(rgb)
        coords=predictor.transform.apply_coords(np.asarray(points,np.float32),(240,320))
        coords=torch.from_numpy(coords)[:,None]
        labels=torch.ones((len(current),1),dtype=torch.int64)
        sparse,dense=model.prompt_encoder(points=(coords,labels),boxes=None,masks=None)
        low,quality=model.mask_decoder.predict_masks(image_embeddings=predictor.features,
            image_pe=model.prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense)
        full=model.postprocess_masks(low,predictor.input_size,predictor.original_size)
        if not verified:
            # Ensure the one-pass four-head call agrees with both original public modes.
            for multi,offset in ((False,0),(True,1)):
                masks,q,small=predictor.predict_torch(coords,labels,multimask_output=multi,return_logits=True)
                stop=1 if not multi else 4
                torch.testing.assert_close(masks,full[:,offset:stop],rtol=0,atol=0)
                torch.testing.assert_close(q,quality[:,offset:stop],rtol=0,atol=0)
                torch.testing.assert_close(small,low[:,offset:stop],rtol=0,atol=0)
            verified=True
        chosen=quality[:,1:].argmax(dim=1)+1
        for j,((index,reference),point) in enumerate(zip(current,points,strict=True)):
            modes={}
            for mode,head in (('single',0),('multimask',int(chosen[j]))):
                pred=(full[j,head]>model.mask_threshold).numpy()
                modes[mode]=dict(metrics=score(pred,instance,reference['target']),head=head,
                    quality=float(quality[j,head]),packed=np.packbits(pred,axis=-1,bitorder='little'))
            results.append(dict(index=index,skipped=False,point_xy=point,modes=modes))
    assert state_hash(model)==FROZEN_HASH
    return dict(results=results,seconds=time.perf_counter()-started,frozen_hash=FROZEN_HASH,
                original_predictor_modes_match=verified)


def aggregate(rows):
    valid=[r for r in rows if not r['skipped']]
    def part(items):
        if not items:return dict(visible_cases=0)
        return dict(visible_cases=len(items),mean_iou=float(np.mean([r['iou'] for r in items])),
            mean_dice=float(np.mean([r['dice'] for r in items])),
            identity_accuracy=float(np.mean([r['correct_identity'] for r in items])),
            iou_ge_50=float(np.mean([r['iou']>=.5 for r in items])),
            iou_ge_75=float(np.mean([r['iou']>=.75 for r in items])))
    confusion=np.zeros((4,5),int)
    for r in valid:confusion[r['target'],r['identity']]+=1
    return dict(cases=len(rows),skipped_low_visibility=len(rows)-len(valid),**part(valid),
        by_phase={phase:part([r for r in valid if (r['frame']==0)==(phase=='home')]) for phase in ('home','later')},
        by_target={NAMES[t]:part([r for r in valid if r['target']==t]) for t in range(4)},
        identity_confusion=dict(rows=list(NAMES),columns=[*NAMES,'no_object_overlap'],counts=confusion.tolist()))


def sheets(reference,records,packed):
    with np.load(REFERENCE/'predictions.npz') as z:task=z['packed'].copy()
    selections=[]
    for home in (True,False):
        for target in range(4):
            candidates=[i for i,r in enumerate(reference) if r['visible_pixels']>=8 and r['target']==target and (r['frame']==0)==home]
            failures=[i for i in candidates if not reference[i]['correct_identity']]
            selections.append((failures or candidates)[0])
    def draw(chosen,filename,caption):
        sheet=Image.new('RGB',(1280,290*len(chosen)+30),'white');pen=ImageDraw.Draw(sheet)
        pen.text((8,8),caption,fill='black')
        for row,i in enumerate(chosen):
            r=reference[i];rgb,instance=read(DATA/r['path'],[r['frame']]);rgb=rgb[0];gt=instance[0]==r['target']+1
            y=30+row*290;name=NAMES[r['target']]
            pen.text((8,y+4),f'{name} | {r["path"]} | frame{r["frame"]}',fill='black')
            image=Image.fromarray(rgb);marker=ImageDraw.Draw(image)
            x0,y0=records['single'][i]['point_xy'];marker.ellipse((x0-3,y0-3,x0+3,y0+3),fill='magenta',outline='white')
            sheet.paste(image,(0,y+40));pen.text((8,y+23),'RGB + positive point',fill='black')
            for col,(label,mask,iou) in enumerate((
                ('E124 task, epoch1',task[i],r['iou']),
                ('Original + point, single',packed['single'][i],records['single'][i]['iou']),
                ('Original + point, quality-selected',packed['multimask'][i],records['multimask'][i]['iou'])),1):
                pred=np.unpackbits(mask,axis=-1,count=320,bitorder='little').astype(bool)
                sheet.paste(overlay(rgb,gt,pred),(col*320,y+40))
                pen.text((col*320+4,y+23),f'{label} | IoU {iou:.3f}',fill='black')
        sheet.save(ROOT/filename)
    draw(selections,'comparison.png','Selected task-token failures by color and phase; orange=GT, cyan=prediction')
    worst=[]
    for target in range(4):
        valid=[i for i,r in enumerate(records['multimask']) if not r['skipped'] and r['target']==target]
        worst.append(min(valid,key=lambda i:records['multimask'][i]['iou']))
    draw(worst,'point_failures.png','Worst quality-selected original-point example per target; diagnostic selection')


def main():
    ROOT.mkdir(exist_ok=False);shutil.copyfile(NOTE,ROOT/'README.md')
    started=time.perf_counter();torch.set_num_threads(4)
    reference=[json.loads(s) for s in (REFERENCE/'results.jsonl').read_text().splitlines()]
    assert len(reference)==8192 and sum(r['visible_pixels']>=8 for r in reference)==8160
    by_path=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(reference):by_path[r['path']][r['frame']].append((i,r))
    jobs=[dict(path=path,queries=dict(frames)) for path,frames in by_path.items()]
    assert len(jobs)==256
    weight_hash=sha(WEIGHTS)
    config=dict(status='running',source='original official MobileSAM checkpoint only;no E123/E124 learned weights',
        pretrained=str(WEIGHTS),pretrained_sha256=weight_hash,device='CPU FP32',workers=3,threads_per_worker=8,
        frozen='entire model;no optimizer or updates',prompt='one positive point at maximum visible-GT distance transform per target/frame',
        inputs='RGB + one2Dpoint + positive label;no task token,box,mask input,depth,joints,XYZ',
        oracle='GT visible mask chooses only the positive point;GT never selects an output mask',
        cases=8192,visible_cases=8160,low_visibility_skipped=32,images=2048,episodes=256,
        temporal_sampling='exact E124 validation:8uniform frames per episode,4task queries per frame',
        single='original single-mask head0,matching the branch trained by E124',
        multimask='original candidate heads1..3;choose highest predicted quality,never GT IoU',
        preprocessing='original SamPredictor PIL longest-side1024 resize and original point-coordinate transform',
        reference=str(REFERENCE),reference_sha256=sha(REFERENCE/'results.jsonl'),
        inference='per-frame fresh oracle point,no temporal memory or tracking;validation diagnostic,no manipulation rollout',
        H=None,K=None,training_epochs=0,learning_rate=None,actions=None,
        source_sha256=sha(Path(__file__)))
    write(ROOT/'config.json',config)
    metadata=('image','target','path','group','layout','permutation','frame','visible_pixels')
    records={m:[dict({k:r[k] for k in metadata},skipped=True,point_xy=None) for r in reference] for m in MODES}
    packed={m:np.zeros((len(reference),240,40),np.uint8) for m in MODES};hashes=set();verified=0
    completed=0
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        pending={pool.submit(episode,jobs[i]):i for i in range(3)};cursor=3
        with tqdm(total=len(jobs),desc='Original MobileSAM point validation',mininterval=10) as bar:
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    index=pending.pop(future);result=future.result();hashes.add(result['frozen_hash'])
                    verified+=int(result['original_predictor_modes_match'])
                    for item in result['results']:
                        i=item['index']
                        if item['skipped']:continue
                        for mode in MODES:
                            value=item['modes'][mode]
                            records[mode][i].update(skipped=False,point_xy=item['point_xy'],
                                **value['metrics'],quality=value['quality'],head=value['head'])
                            packed[mode][i]=value['packed']
                    completed+=1;bar.update(1)
                    progress=dict(status='running',episodes=completed,total_episodes=len(jobs),elapsed_s=time.perf_counter()-started,
                                  ETA_s=(time.perf_counter()-started)/completed*(len(jobs)-completed),last_episode_s=result['seconds'])
                    write(ROOT/'progress.json',progress)
                    if completed==1 or completed%16==0:print(json.dumps(progress),flush=True)
                    if cursor<len(jobs):pending[pool.submit(episode,jobs[cursor])]=cursor;cursor+=1
    assert len(hashes)==1 and verified==256 and sha(WEIGHTS)==weight_hash
    summaries={}
    for mode in MODES:
        out=ROOT/mode;out.mkdir();shutil.copyfile(NOTE,out/'README.md')
        assert sum(not r['skipped'] for r in records[mode])==8160
        (out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records[mode]))
        np.savez_compressed(out/'predictions.npz',packed=packed[mode],bitorder='little')
        summaries[mode]=aggregate(records[mode]);write(out/'summary.json',summaries[mode])
    summaries['task_epoch1']=json.loads((REFERENCE/'summary.json').read_text())
    paired={}
    for mode in MODES:
        good=[i for i,r in enumerate(reference) if r['visible_pixels']>=8]
        paired[mode]=dict(both_correct=sum(reference[i]['correct_identity'] and records[mode][i]['correct_identity'] for i in good),
            point_only_correct=sum(not reference[i]['correct_identity'] and records[mode][i]['correct_identity'] for i in good),
            task_only_correct=sum(reference[i]['correct_identity'] and not records[mode][i]['correct_identity'] for i in good),
            both_wrong=sum(not reference[i]['correct_identity'] and not records[mode][i]['correct_identity'] for i in good))
    write(ROOT/'summary.json',summaries);write(ROOT/'paired.json',paired)
    sheets(reference,records,packed)
    config.update(status='complete',elapsed_s=time.perf_counter()-started);write(ROOT/'config.json',config)
    write(ROOT/'verification.json',dict(status='passed',pretrained_file_unchanged=True,model_state_unchanged=True,
        frozen_model_sha256=list(hashes)[0],episodes_verified_against_original_predictor=verified,
        exact_reference_query_order=True,GT_never_selects_mask=True,reference_visible_count_matches=True,
        training_process_unmodified=True))
    write(ROOT/'progress.json',dict(status='complete',elapsed_s=config['elapsed_s'],summaries=summaries,paired=paired))
    text=NOTE.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nResults:\n'+json.dumps(summaries,indent=2)+'\n'
    NOTE.write_text(text);(ROOT/'README.md').write_text(text)
    for mode in MODES:(ROOT/mode/'README.md').write_text(text)
    print(json.dumps(dict(status='complete',summaries=summaries,paired=paired)),flush=True)


if __name__=='__main__':main()
