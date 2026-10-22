"""Fixed-image/state/noise query-swap probe at four post-shuffle initial states."""
import argparse
import itertools
import json
from pathlib import Path
import shutil

import av
import numpy as np
import torch
from tqdm import tqdm

from OATFlow.shuffle_bc.data import DATA
from OATFlow.shuffle_bc.direct_data import allocate
from OATFlow.shuffle_bc.direct_model import DirectPolicy, pack_top, unpack_top
from OATFlow.shuffle_bc.evaluate import cases
from OATFlow.shuffle_bc.train_direct import frozen_models
from OATFlow.simulation.io import write_json
from OATFlow.tracking.data import sample_indices
from OATFlow.tracking.linear_probe import fingerprint
from OATFlow.tracking.mask_data import read_segment
from OATFlow.tracking.mask_video import digest


def first_rgb(row, view):
    with av.open(str(DATA/row['path']/f'rgb_{view}.mp4')) as video:
        video.streams.video[0].thread_count = 1
        return next(video.decode(video=0)).to_ndarray(format='rgb24')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e115_query_README.md', out/'README.md')
    checkpoint = Path('/workspace/robotics_assets/experiments/e115_direct/train/best.pt')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    policy = DirectPolicy(); policy.load_state_dict(saved['model'], strict=True)
    policy.eval().requires_grad_(False).cuda()
    tracker, wrist = frozen_models()
    frozen = dict(TCOW=fingerprint(tracker),auxiliary_ViT=fingerprint(wrist))
    assert frozen == json.loads((checkpoint.parent/'frozen_hashes.json').read_text())
    all_rows = cases(); groups = []
    for group in ('train','composition','spatial','combined'):
        candidates = [r for r in all_rows if r['evaluation_group']==group]
        rows = sorted([r for r in candidates if r['context_path']==candidates[0]['context_path']], key=lambda r:r['target_id'])
        assert [r['target_id'] for r in rows] == [0,1,2,3]
        groups.append(rows)
    config = dict(status='running',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        epoch=saved['epoch'],step=saved['step'],cases=groups,H=25,K=None,flow_steps=10,noise_seeds=[0,1,2,3],
        fixed='same current top/context RGB, auxiliary RGB, measured joints and noise across all four queries',
        varied='only initial visible target mask; resulting frozen TCOW hidden is recomputed',
        endpoints='expert raw frame0 only; no policy motion',
        metric='sampled EE deltaXYZ against recorded expert H25 trajectory; opposite-cover query controls',
        training=saved['config'],source_sha256=digest(Path(__file__)))
    write_json(out/'config.json',config)
    shutil.copy2(Path(__file__),out/'source.py')
    records=[]; trace={}
    for rows in tqdm(groups,desc='Fixed-state query swaps',mininterval=5):
        labels=allocate(rows,validation=True)
        first=[begin for begin,end in labels['slices']]
        expert=labels['actions'][first].numpy()
        assert labels['mask'][first].all()
        joints=labels['joints'][first]
        assert torch.equal(joints,joints[:1].expand_as(joints))
        state=[]
        for row in rows:
            with np.load(DATA/row['path']/'sim_state.npz') as z:
                state.append(z['initial_qpos'])
        assert np.array_equal(state,np.repeat(np.asarray(state[:1]),4,axis=0))
        top=first_rgb(rows[0],'top');side=first_rgb(rows[0],'left')
        codec_difference=[]
        for row in rows[1:]:
            codec_difference.append(dict(target=row['target_id'],top_mae=float(np.abs(first_rgb(row,'top').astype(float)-top).mean()),
                left_mae=float(np.abs(first_rgb(row,'left').astype(float)-side).mean())))
        context=read_segment(DATA,rows[0]['context_path'],True)
        images=torch.cat((context['rgb'],torch.from_numpy(top[40:280].transpose(2,0,1).copy())[None]))
        chosen=sample_indices(len(images)-1,30)
        rgb=images[chosen].permute(1,0,2,3)[None].repeat(4,1,1,1,1).cuda()
        query=torch.zeros(4,1,30,240,320,dtype=torch.bool,device='cuda')
        query[:,0,0]=torch.from_numpy(context['prompt']).cuda()
        truth=[]
        for row in rows:
            with np.load(DATA/row['path']/'masks.npz') as z:
                truth.append(np.unpackbits(z['tcow_packed'][0,0,0,40:280],axis=-1,count=320,bitorder='little').astype(bool))
        truth=np.asarray(truth)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=unpack_top(*pack_top(tracker.features(rgb,query)[:,-1].contiguous()))
            aux=wrist(torch.from_numpy(side.transpose(2,0,1).copy())[None].cuda()).half().expand(4,-1,-1)
            predicted=tracker.masks_from_features(hidden[:,None])[:,0].gt(0).cpu().numpy()
        overlap=(predicted[:,None]&truth[None]).sum((-1,-2))/np.maximum((predicted[:,None]|truth[None]).sum((-1,-2)),1)
        covers=np.array([r['selected_cover'] for r in rows])
        pairs=[(i,j) for i,j in itertools.combinations(range(4),2) if covers[i]!=covers[j]]
        expert_separation=np.mean([np.linalg.norm(expert[i,:,:3]-expert[j,:,:3],axis=-1).mean() for i,j in pairs])*1000
        samples=[];metrics=[]
        for seed in range(4):
            generator=torch.Generator(device='cuda').manual_seed(seed)
            noise=policy.flow.sample_noise(1,'cuda',generator).expand(4,-1,-1)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                action=policy.sample_features(hidden,aux,joints.cuda(),noise,steps=10).float().cpu().numpy()
            assert np.isfinite(action).all()
            distances=np.sqrt(np.square(action[:,None,:,:3]-expert[None,:,:,:3]).sum(-1).mean(-1))*1000
            nearest=covers[distances.argmin(1)]
            correct=distances.diagonal()
            wrong=np.array([distances[covers!=covers[i],i].mean() for i in range(4)])
            separation=np.mean([np.linalg.norm(action[i,:,:3]-action[j,:,:3],axis=-1).mean() for i,j in pairs])*1000
            metrics.append(dict(seed=seed,correct_query_xyz_rmse_mm=correct.tolist(),opposite_cover_query_xyz_rmse_mm=wrong.tolist(),
                cover_correct=(nearest==covers).tolist(),nearest_expert_cover=nearest.tolist(),fm_opposite_cover_separation_mm=float(separation)))
            samples.append(action)
        record=dict(group=rows[0]['evaluation_group'],context=rows[0]['context_path'],
            target_mask_iou=overlap.diagonal().tolist(),mask_binding_correct=(overlap.argmax(1)==np.arange(4)).tolist(),
            selected_covers=covers.tolist(),recorded_codec_difference=codec_difference,
            expert_opposite_cover_separation_mm=float(expert_separation),metrics=metrics)
        records.append(record)
        key=rows[0]['evaluation_group'];trace[key+'_sampled_actions']=np.asarray(samples)
        trace[key+'_expert_actions']=expert;trace[key+'_mask_iou_matrix']=overlap
        print('RESULT',json.dumps(record),flush=True)
        del labels,context,images,rgb,query,hidden,aux
    all_metrics=[m for r in records for m in r['metrics']]
    summary=dict(status='complete',contexts=4,query_cases=16,noise_seeds=4,
        mask_bindings_correct=sum(sum(r['mask_binding_correct']) for r in records),
        mask_iou_mean=float(np.mean([r['target_mask_iou'] for r in records])),
        sampled_cover_correct=sum(sum(m['cover_correct']) for m in all_metrics),sampled_cover_total=64,
        correct_query_xyz_rmse_mm=float(np.mean([m['correct_query_xyz_rmse_mm'] for m in all_metrics])),
        opposite_cover_query_xyz_rmse_mm=float(np.mean([m['opposite_cover_query_xyz_rmse_mm'] for m in all_metrics])),
        fm_opposite_cover_separation_mm=float(np.mean([m['fm_opposite_cover_separation_mm'] for m in all_metrics])),
        expert_opposite_cover_separation_mm=float(np.mean([r['expert_opposite_cover_separation_mm'] for r in records])),records=records)
    assert frozen==dict(TCOW=fingerprint(tracker),auxiliary_ViT=fingerprint(wrist))
    assert digest(checkpoint)==config['checkpoint_sha256']
    np.savez_compressed(out/'traces.npz',**trace)
    write_json(out/'summary.json',summary);config['status']='complete';write_json(out/'config.json',config)
    print('COMPLETE',json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
