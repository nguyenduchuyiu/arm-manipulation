"""Replay E122 K10outcome pairs or K15failures with RGB and cloud diagnostics."""
import argparse
import json
from pathlib import Path
import shutil

from PIL import Image
import numpy as np
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
def main(k=10,device='cuda'):
    assert k in (10,15)
    failures_only=k==15
    test=ROOT/f'test_k{k}';out=test/'videos_1';out.mkdir(exist_ok=False)
    shutil.copyfile('docs/research/e122_k15_video_README.md' if failures_only else 'docs/research/e122_video_README.md',out/'README.md')
    protocol=json.loads((test/'config.json').read_text())
    assert protocol['status']=='complete' and protocol['checkpoint_step']==23520 and protocol['K']==k
    checkpoint_hash=sha(test/'checkpoint.pt');assert checkpoint_hash==protocol['checkpoint_sha256']
    rows=[json.loads(s) for s in (test/'results.jsonl').read_text().splitlines()]
    assert len(rows)==1408
    outcomes=(False,) if failures_only else (True,False)
    selected=[next(r for r in rows if r['group']==group and r['success']==success)
              for group in ('composition','spatial','combined') for success in outcomes]
    config=dict(status='rendering',checkpoint_step=23520,checkpoint_sha256=checkpoint_hash,
        training=protocol['training'],H=25,K=k,flow_steps=10,noise_seed=0,frame_budget=300,
        diagnostic_device=device,diagnostic_precision='BF16' if device=='cuda' else 'FP32',
        fps=25,resolution=[1344,1100],selected=selected,
        selection=('first failure' if failures_only else 'first success/failure')+' in each of3groups from complete1408case result order;diagnostic selection,not representative rates',
        results_sha256=sha(test/'results.jsonl'),no_physics_rerun=True,no_action_resampling=True,
        panels=['front RGB + visible GT/predicted target cloud','oblique physical replay + measured EE path',
                '25cm target point-cloud zoom','localization/contact/lift diagnostics'],
        auxiliary=f'recompute checkpoint cloud at pre-action replans every{k}frames;hold between replans;no terminal extra inference',
        observations='re-render saved float32qpos,not original saved RGB/depth;small raster/numerical differences possible',
        GT='eroded visible target mask +2mmdepth/calibration,60mmglobal band and10mm3x3local median spike filter;diagnostics only',
        cloud_metric='symmetric unsquared surface Chamfer against all visible GT points;not 3Dcenter error',
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['full_video.py','fusion_cloud_video.py','patch_fusion.py']})
    write(out/'config.json',config)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    saved=torch.load(test/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==23520
    policy=PatchFusionPolicy(initialize=False).to(device);policy.load_state_dict(saved['model'],strict=True)
    policy.eval();del saved
    model=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml');clip_planes(model)
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    columns=len(outcomes);results=[];sheet=Image.new('RGB',(672*columns,1650),'white')
    for i,record in enumerate(tqdm(selected,desc=f'E122 K{k}outcome cloud videos',mininterval=5)):
        result=render(record,test,out,model,policy,experiment='E122',training='final23520 /10.22epochs',
                      physical_overview=True,clean_depth_labels=True,k=k)
        results.append(result)
        with Image.open(out/(Path(result['video']).stem+'_poster.png')) as poster:
            sheet.paste(poster.resize((672,550),Image.Resampling.LANCZOS),((i%columns)*672,(i//columns)*550))
        write(out/'progress.json',dict(status='rendering',complete=len(results),total=len(selected)))
    sheet.save(out/'contact_sheet.png')
    for result in results:
        assert result['decoded_frames']==result['record']['frames']+1
        with np.load(out/(Path(result['video']).stem+'_predictions.npz')) as z:
            np.testing.assert_array_equal(z['frames'],np.arange(0,result['record']['frames'],k))
    config['status']='complete';write(out/'config.json',config)
    verification=dict(status='passed',clips=results,all_trace_hashes_unchanged=True,all_videos_fully_decoded=True,
        all_cloud_replan_frames_verified=True,
        source_checkpoint_unchanged=sha(test/'checkpoint.pt')==checkpoint_hash,
        source_results_unchanged=sha(test/'results.jsonl')==config['results_sha256'])
    assert verification['source_checkpoint_unchanged'] and verification['source_results_unchanged']
    write(out/'verification.json',verification);write(out/'progress.json',dict(status='complete',complete=len(results),total=len(selected)))
    with (out/'README.md').open('a') as f:
        f.write(f'\nCOMPLETE: {len(results)}full-decoded videos; original physical traces/checkpoint/results unchanged.\n')
        for result in results:
            f.write(f'\n- {result["video"]}: {result["record"]["path"]}, {result["duration_s"]:.2f}s, {result["record"]["reason"]}.\n')
    print(json.dumps(dict(status='complete',out=str(out),clips=results)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k',type=int,choices=(10,15),default=10)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args();main(args.k,device=args.device)
