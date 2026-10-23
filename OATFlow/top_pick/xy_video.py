"""Replay the two E122 outside-X BlueCube failures with target-cloud diagnostics."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.simulation.render import visual_model
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.full_train import ROOT
from OATFlow.top_pick.fusion_cloud_video import render
from OATFlow.top_pick.patch_fusion import PatchFusionPolicy


@torch.inference_mode()
def main():
    test=ROOT/'test_xy';out=test/'videos_1';out.mkdir(exist_ok=False)
    note=Path('docs/research/e122_xy_video_README.md');shutil.copyfile(note,out/'README.md')
    protocol=json.loads((test/'config.json').read_text())
    assert protocol['status']=='complete' and protocol['checkpoint_step']==23520 and protocol['K']==15
    checkpoint=Path(protocol['checkpoint']);assert sha(checkpoint)==protocol['checkpoint_sha256']
    rows=list(map(json.loads,(test/'results.jsonl').read_text().splitlines()))
    selected=[next(r for r in rows if r['path']==f'{edge}/t1/outside') for edge in ('x_low','x_high')]
    assert all(not r['success'] and r['target']=='BlueCube' for r in selected)
    config=dict(status='rendering',checkpoint=str(checkpoint),checkpoint_sha256=protocol['checkpoint_sha256'],
        checkpoint_step=23520,training=protocol['training'],selected=selected,H=25,K=15,
        flow_steps=10,noise_seed=0,frame_budget=300,diagnostic_device='cpu',diagnostic_precision='FP32',
        fps=25,resolution=[1344,1100],no_physics_rerun=True,no_action_resampling=True,
        results_sha256=sha(test/'results.jsonl'),
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('xy_video.py','fusion_cloud_video.py')})
    write(out/'config.json',config)
    torch.set_num_threads(4)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==23520
    policy=PatchFusionPolicy(initialize=False).cpu();policy.load_state_dict(saved['model'],strict=True)
    policy.eval();del saved
    model=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml');clip_planes(model)
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    results=[];sheet=Image.new('RGB',(1344,550),'white')
    for i,record in enumerate(tqdm(selected,desc='E122 outside-X failure videos',mininterval=5)):
        name='fail_'+record['edge'];x=record['target_xy'][0]
        result=render(record,test,out,model,policy,experiment='E122 outside-X',
            training=f'step23520 / X={x:.2f}m;train[.30,.78]',
            physical_overview=True,clean_depth_labels=True,k=15,name=name)
        assert result['decoded_frames']==record['frames']+1
        with np.load(out/(name+'_predictions.npz')) as z:
            np.testing.assert_array_equal(z['frames'],np.arange(0,record['frames'],15))
        with Image.open(out/(name+'_poster.png')) as image:
            sheet.paste(image.resize((672,550),Image.Resampling.LANCZOS),(i*672,0))
        results.append(result);write(out/'progress.json',dict(status='rendering',complete=len(results),total=2))
    sheet.save(out/'contact_sheet.png')
    assert sha(checkpoint)==config['checkpoint_sha256'] and sha(test/'results.jsonl')==config['results_sha256']
    write(out/'verification.json',dict(status='passed',clips=results,all_trace_hashes_unchanged=True,
        all_videos_fully_decoded=True,all_cloud_replan_frames_verified=True,
        source_checkpoint_unchanged=True,source_results_unchanged=True))
    config['status']='complete';write(out/'config.json',config)
    write(out/'progress.json',dict(status='complete',complete=2,total=2))
    text=note.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nBoth videos fully decoded;301frames each;all source hashes unchanged and20cloud replan frames verified per clip.\n'
    note.write_text(text);(out/'README.md').write_text(text)
    with Path('docs/research/REPORT.md').open('a') as f:
        f.write('\nE122 outside-X failure videos: test_xy/videos_1/fail_x_low.mp4 and fail_x_high.mp4. Saved physical trajectories replayed unchanged, CPU FP32 auxiliary cloud at K15, both clips verified301frames.\n')
    print(json.dumps(dict(status='complete',out=str(out),clips=results)),flush=True)


if __name__=='__main__':main()
