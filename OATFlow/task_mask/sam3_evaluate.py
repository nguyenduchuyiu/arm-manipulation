"""Original official SAM 3 BF16 image/text segmentation on fixed pick samples."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

import av
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
import torch
from tqdm import tqdm

ROOT=Path('/workspace/robotics_assets/experiments/e132_sam')
DATA=Path('/workspace/robotics_assets/datasets/pick_wide_rgbd')
NOTE=Path('docs/research/e132_README.md')
REVISION='3c879f39826c281e95690f02c7821c4de09afae7'
PROMPTS=['red cube','blue cube','green cube','yellow cube']
STAGES=['home','approach','engage','lift']


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def prepare():
    ROOT.mkdir(exist_ok=False);shutil.copyfile(NOTE,ROOT/'README.md')
    rows=list(map(json.loads,Path('/workspace/robotics_assets/experiments/e131_object/val_manifest.jsonl').read_text().splitlines()))
    jobs=[];images=[];masks=[];hashes={}
    for target,layout in enumerate(range(32,36)):
        row=min((r for r in rows if r['layout_id']==layout and r['target_id']==target),key=lambda r:r['permutation'])
        source=DATA/row['path'];files=['rgb_front.mp4','masks.npz','supervision.npz']
        hashes.update({str(source/name):digest(source/name) for name in files})
        with np.load(source/'supervision.npz') as z:phase=z['phase'].copy()
        with np.load(source/'masks.npz') as z:instance=z['instance'][:,0].copy()
        frames=[0]+[int(np.flatnonzero(np.char.endswith(phase,'_'+stage))[-1]) for stage in STAGES[1:]]
        assert len(set(frames))==4
        wanted={f:stage for stage,f in zip(STAGES,frames)}
        selected={}
        with av.open(str(source/'rgb_front.mp4')) as stream:
            stream.streams.video[0].thread_count=1
            for frame,item in enumerate(stream.decode(video=0)):
                if frame in wanted:selected[frame]=item.to_ndarray(format='rgb24')
        for stage,frame in zip(STAGES,frames):
            jobs.append(dict(image=len(jobs),path=row['path'],layout=layout,source_target=target,stage=stage,frame=frame))
            images.append(selected[frame]);masks.append(instance[frame])
    assert len(jobs)==16
    np.savez_compressed(ROOT/'inputs.npz',rgb=np.stack(images),instance=np.stack(masks));write(ROOT/'jobs.json',jobs)
    sheet=Image.new('RGB',(1280,4*278),'white');draw=ImageDraw.Draw(sheet)
    for i,(job,rgb) in enumerate(zip(jobs,images)):
        x=(i%4)*320;y=(i//4)*278
        draw.text((x+5,y+6),f'g{job["layout"]:03d} / {job["stage"]} / f{job["frame"]}',fill='#21354a')
        sheet.paste(Image.fromarray(rgb),(x,y+32))
    sheet.save(ROOT/'input_samples.png')
    environment={p:version(p) for p in ['torch','torchvision','transformers','huggingface-hub','tokenizers','safetensors','regex','httpx2','typer']}
    write(ROOT/'environment.json',dict(packages=environment,CUDA_BF16_supported=torch.cuda.is_bf16_supported(),
        torch_path=torch.__file__,python=__import__('sys').version,model_import_available=True))
    write(ROOT/'config.json',dict(status='prepared_waiting_access',repo_id='facebook/sam3',revision=REVISION,prompts=PROMPTS,
        images=16,queries=64,source_hashes=hashes,inputs_sha256=digest(ROOT/'inputs.npz'),jobs_sha256=digest(ROOT/'jobs.json'),
        confidence_threshold=.5,mask_probability_threshold=.5,input='RGB+text only;GT masks are scoring data',
        planned_precision='BF16 weights and CUDA autocast;no quantization',training=False,source_sha256=digest(Path(__file__))))
    text=NOTE.read_text().replace('Status: PREPARING; waiting for authorized weight access.','Status: INPUTS PREPARED; waiting for authorized weight access.')
    text+='\nOffline preparation complete:16RGB/GT images and64fixed queries ready. Model inference has NOT RUN.\n'
    NOTE.write_text(text);(ROOT/'README.md').write_text(text)
    print(json.dumps(dict(status='prepared_waiting_access',images=16,queries=64,CUDA_BF16_supported=torch.cuda.is_bf16_supported())),flush=True)


def score(pred,gt,target):
    truth=gt==target+1;intersection=int((pred&truth).sum());union=int((pred|truth).sum())
    candidates=[]
    for t in range(4):
        mask=gt==t+1;n=int((mask&pred).sum());d=int((mask|pred).sum());candidates.append(n/d if d else 0.)
    selected=int(np.argmax(candidates)) if max(candidates)>0 else -1
    return dict(visible_pixels=int(truth.sum()),predicted_pixels=int(pred.sum()),iou=intersection/union if union else 1.,
        precision=intersection/int(pred.sum()) if pred.any() else 0.,recall=intersection/int(truth.sum()) if truth.any() else 0.,
        identity=selected,correct_identity=selected==target,object_IoUs=candidates)


def overlay(rgb,truth,pred):
    pixels=rgb.astype(float).copy();pixels[pred]=pixels[pred]*.45+np.array([0,190,245])*.55
    boundary=truth&~binary_erosion(truth)
    pixels[boundary]=[255,180,40]
    return Image.fromarray(pixels.clip(0,255).astype(np.uint8))


def figures(images,gt,jobs,records,selected,out):
    for stage in STAGES:
        ids=[j for j,job in enumerate(jobs) if job['stage']==stage]
        sheet=Image.new('RGB',(1600,290*len(ids)),'white');draw=ImageDraw.Draw(sheet)
        for row,i in enumerate(ids):
            y=row*290;sheet.paste(Image.fromarray(images[i]),(0,y+35));draw.text((5,y+6),f'{jobs[i]["path"]} f{jobs[i]["frame"]}',fill='black')
            for t,prompt in enumerate(PROMPTS):
                q=i*4+t;r=records[q];x=(t+1)*320
                sheet.paste(overlay(images[i],gt[i]==t+1,selected[q]),(x,y+35))
                draw.text((x+5,y+5),f'{prompt} | IoU {r["iou"]:.3f} | n={r["detections"]}',fill='black')
                draw.text((x+5,y+17),f'confidence {r["selected_confidence"]:.3f}' if r['selected_confidence'] is not None else 'no detection',fill='black')
        sheet.save(out/f'{stage}.png')


@torch.inference_mode()
def run():
    from huggingface_hub import snapshot_download
    from transformers import Sam3Model,Sam3Processor
    cfg=json.loads((ROOT/'config.json').read_text());assert cfg['status']=='prepared_waiting_access'
    assert digest(ROOT/'inputs.npz')==cfg['inputs_sha256'] and digest(ROOT/'jobs.json')==cfg['jobs_sha256']
    # Access is required. A denied download stops here without creating prediction artifacts.
    path=snapshot_download(repo_id='facebook/sam3',revision=REVISION,token=True,
        cache_dir='/workspace/robotics_assets/pretrained/sam3_hf',allow_patterns=['*.json','*.safetensors','*.txt'])
    weights=Path(path)/'model.safetensors';weight_hash=digest(weights)
    model=Sam3Model.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    processor=Sam3Processor.from_pretrained(path);assert torch.cuda.is_bf16_supported()
    assert {p.dtype for p in model.parameters() if p.is_floating_point()}=={torch.bfloat16}
    out=ROOT/'results';out.mkdir(exist_ok=False);shutil.copyfile(NOTE,out/'README.md')
    with np.load(ROOT/'inputs.npz') as z:images=z['rgb'];gt=z['instance']
    jobs=json.loads((ROOT/'jobs.json').read_text());torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    text_inputs=[processor(text=p,return_tensors='pt').to('cuda') for p in PROMPTS]
    with torch.autocast('cuda',dtype=torch.bfloat16):text_features=[model.get_text_features(**x) for x in text_inputs]
    records=[];selected=[];timings=[];parity_error=0.;raw_dtypes=set();vision_dtype=set()
    for i,rgb in enumerate(tqdm(images,desc='SAM3 BF16 RGB + four text prompts',mininterval=5)):
        started=time.perf_counter();image_inputs=processor(images=Image.fromarray(rgb),return_tensors='pt')
        pixels=image_inputs['pixel_values'].to(device='cuda',dtype=torch.bfloat16)
        torch.cuda.synchronize();tick=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):vision=model.get_vision_features(pixel_values=pixels)
        torch.cuda.synchronize();vision_ms=(time.perf_counter()-tick)*1000
        vision_dtype.update(str(a.dtype) for a in vision.fpn_hidden_states)
        prompt_times=[]
        for t,prompt in enumerate(PROMPTS):
            tick=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(vision_embeds=vision,text_embeds=text_features[t],attention_mask=text_inputs[t]['attention_mask'])
            torch.cuda.synchronize();raw_dtypes.add(str(output.pred_masks.dtype));head_ms=(time.perf_counter()-tick)*1000
            if i==0 and t==0:
                with torch.autocast('cuda',dtype=torch.bfloat16):reference=model(pixel_values=pixels,**text_inputs[t])
                parity_error=float((output.pred_masks.float()-reference.pred_masks.float()).abs().max());assert parity_error<.02
                del reference
            result=processor.post_process_instance_segmentation(output,threshold=.5,mask_threshold=.5,target_sizes=[(240,320)])[0]
            masks=result['masks'].cpu().numpy().astype(bool);scores=result['scores'].float().cpu().numpy();boxes=result['boxes'].float().cpu().numpy()
            best=int(scores.argmax()) if len(scores) else -1;pred=masks[best] if best>=0 else np.zeros((240,320),bool)
            q=i*4+t;np.savez_compressed(out/f'proposals_{q:03d}.npz',packed=np.packbits(masks,axis=-1,bitorder='little'),scores=scores,boxes=boxes,width=320)
            row=dict(**jobs[i],prompt=prompt,target=t,**score(pred,gt[i],t),detections=len(scores),selected_index=best,
                selected_confidence=float(scores[best]) if best>=0 else None,head_ms=head_ms)
            records.append(row);selected.append(pred);prompt_times.append(head_ms)
        torch.cuda.synchronize();timings.append(dict(image=i,vision_ms=vision_ms,prompt_ms=prompt_times,end_to_end_ms=(time.perf_counter()-started)*1000))
    visible=[r for r in records if r['visible_pixels']>=8]
    summary=dict(status='complete',images=len(images),queries=len(records),visible_queries=len(visible),
        mean_IoU=float(np.mean([r['iou'] for r in visible])),identity_accuracy=float(np.mean([r['correct_identity'] for r in visible])),
        IoU_ge50=float(np.mean([r['iou']>=.5 for r in visible])),IoU_ge75=float(np.mean([r['iou']>=.75 for r in visible])),
        no_detections=sum(r['detections']==0 for r in records),multiple_detections=sum(r['detections']>1 for r in records),
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,vision_dtype=list(vision_dtype),mask_logits_dtype=list(raw_dtypes),
        mean_vision_ms=float(np.mean([r['vision_ms'] for r in timings[1:]])),mean_prompt_head_ms=float(np.mean([r['prompt_ms'] for r in timings[1:]])))
    for field,values in [('stage',STAGES),('prompt',PROMPTS)]:
        summary['by_'+field]={str(value):dict(queries=len(rr),mean_IoU=float(np.mean([r['iou'] for r in rr])) if rr else None,
            identity_accuracy=float(np.mean([r['correct_identity'] for r in rr])) if rr else None)
            for value in values for rr in [[r for r in visible if r[field]==value]]}
    write(out/'records.json',records);write(out/'timings.json',timings);write(out/'summary.json',summary)
    np.savez_compressed(out/'predictions.npz',packed=np.packbits(np.stack(selected),axis=-1,bitorder='little'),width=320)
    figures(images,gt,jobs,records,selected,out)
    assert digest(weights)==weight_hash and digest(ROOT/'inputs.npz')==cfg['inputs_sha256']
    write(out/'verification.json',dict(status='passed',weight_file_unchanged=True,all_model_parameters_BF16=True,
        cached_feature_forward_max_logit_difference=parity_error,training=False,GT_used_as_input=False))
    write(out/'config.json',dict(cfg,status='complete',weights=str(weights),weights_sha256=weight_hash,precision='BF16 parameters + CUDA autocast',
        source_sha256=digest(Path(__file__)),environment=json.loads((ROOT/'environment.json').read_text())))
    cfg['status']='complete';write(ROOT/'config.json',cfg)
    text=NOTE.read_text();text=text[:text.find('\nOffline preparation complete:')] if '\nOffline preparation complete:' in text else text
    text=text.replace('Status: INPUTS PREPARED; waiting for authorized weight access.','Status: COMPLETE.')+'\nResults:\n'+json.dumps(summary,indent=2)+'\n'
    NOTE.write_text(text);(ROOT/'README.md').write_text(text);(out/'README.md').write_text(text)
    with Path('docs/research/REPORT.md').open('a') as report:report.write('\nE132 SAM3 BF16 text-prompt pilot:'+json.dumps(summary)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','run'])
    args=parser.parse_args();prepare() if args.mode=='prepare' else run()
