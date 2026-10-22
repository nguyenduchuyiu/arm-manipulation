"""Counterfactual target-ID probe: identical image/state/noise, four target queries."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import time

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from OATFlow.top_pick.collect import DATA,ROOT
from OATFlow.top_pick.features import labels
from OATFlow.top_pick.center import load_policy

OUT=ROOT/'probe_task'
PAIRS=tuple((i,j) for i in range(4) for j in range(i+1,4))


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def summarize(pred,expert,groups,std,goals,velocity):
    result={}
    scale=np.array(std,dtype=np.float32).copy();scale[9]=.0255/2
    for group in sorted(set(groups)):
        use=groups==group;p=pred[use];gt=expert[use]
        group_result=dict(scenes=int(use.sum()),queries=int(use.sum()*4),noise_seeds=pred.shape[1])
        for h in (10,25):
            # [scene,seed,requested query,expert target,H,D].
            difference=p[:,:,:,None,:h]-gt[:,None,None,:,:h]
            normalized=np.square(difference/scale).mean(axis=(-1,-2))
            xyz=np.square(difference[...,:3]).sum(-1).mean(-1)
            diagonal=np.arange(4)
            correct=normalized[:,:,diagonal,diagonal]
            correct_xyz=xyz[:,:,diagonal,diagonal]
            wrong=np.stack([normalized[:,:,i,j] for i in range(4) for j in range(4) if i!=j],axis=-1)
            wrong_xyz=np.stack([xyz[:,:,i,j] for i in range(4) for j in range(4) if i!=j],axis=-1)
            nearest=xyz.argmin(-1)
            confusion=np.zeros((4,4),dtype=int)
            for i in range(4):confusion[i]=np.bincount(nearest[:,:,i].ravel(),minlength=4)
            pred_sep=np.stack([np.linalg.norm(p[:,:,i,:h,:3]-p[:,:,j,:h,:3],axis=-1).mean(-1) for i,j in PAIRS],axis=-1)
            gt_sep=np.stack([np.linalg.norm(gt[:,i,:h,:3]-gt[:,j,:h,:3],axis=-1).mean(-1) for i,j in PAIRS],axis=-1)
            group_result[f'H{h}']=dict(
                correct_query_normalized_mse=float(correct.mean()),wrong_query_normalized_mse=float(wrong.mean()),
                wrong_over_correct_mse=float(wrong.mean()/correct.mean()),
                correct_query_xyz_rmse_mm=float(np.sqrt(correct_xyz.mean())*1000),
                wrong_query_xyz_rmse_mm=float(np.sqrt(wrong_xyz.mean())*1000),
                nearest_expert_query_accuracy=float((nearest==np.arange(4)[None,None]).mean()),
                fixed_query0_accuracy=.25,
                requested_vs_nearest_expert_confusion=confusion.tolist(),
                predicted_query_pair_separation_mm=float(pred_sep.mean()*1000),
                expert_query_pair_separation_mm=float(gt_sep.mean()*1000),
                predicted_over_expert_separation=float(pred_sep.mean()/gt_sep.mean()),
                predicted_query_pair_separation_p10_p50_p90_mm=(np.quantile(pred_sep,[.1,.5,.9])*1000).tolist())
        g=goals[use]
        g_delta=np.stack([np.linalg.norm(g[:,i]-g[:,j],axis=-1) for i,j in PAIRS],axis=-1)
        vel=velocity[use]
        v_delta=np.stack([np.linalg.norm(vel[:,i]-vel[:,j],axis=-1).mean(-1) for i,j in PAIRS],axis=-1)
        group_result['layer_sensitivity']=dict(
            goal_relative_query_change=float(g_delta.mean()/np.linalg.norm(g,axis=-1).mean()),
            velocity_relative_query_change_at_noise_t0=float(v_delta.mean()/np.linalg.norm(vel,axis=-1).mean()))
        result[group]=group_result
    return result


def ridge_probes(goals,slots,xyz,groups):
    train=np.repeat(groups=='train',4)
    outputs={}
    for name,values in [('goal_token',goals),('image_only_mean_of_four_query_tokens',np.repeat(goals.mean(1,keepdims=True),4,axis=1))]:
        x=values.reshape(-1,values.shape[-1]).astype(np.float64)
        y=np.concatenate((np.eye(4)[slots.reshape(-1)],xyz.reshape(-1,3)),axis=-1).astype(np.float64)
        mean=x[train].mean(0);std=np.maximum(x[train].std(0),1e-6)
        x=(x-mean)/std;y_mean=y[train].mean(0)
        gram=x[train].T@x[train];gram.flat[::len(gram)+1]+=10.
        weight=np.linalg.solve(gram,x[train].T@(y[train]-y_mean))
        predictions=x@weight+y_mean
        outputs[name]={}
        for group in sorted(set(groups)):
            use=np.repeat(groups==group,4)
            e=np.linalg.norm(predictions[use,4:]-xyz.reshape(-1,3)[use],axis=-1)*1000
            outputs[name][group]=dict(queries=int(use.sum()),slot_accuracy=float((predictions[use,:4].argmax(-1)==slots.reshape(-1)[use]).mean()),
                                     xyz_mean_error_mm=float(e.mean()),xyz_median_error_mm=float(np.median(e)))
    return dict(method='linear ridge,lambda10,train-only feature mean/std and fitting; no hyperparameter/test tuning',
                train_queries=int(train.sum()),target_id_only_slot_chance=.25,results=outputs,
                limitation='Linear decodability is a diagnostic,not a bound on nonlinear information or closed-loop competence.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'train/best.pt')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();out=args.output
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    model=load_policy(saved).cuda().eval()
    description=('# '+out.parent.name+' target conditioning probe\n\nStatus: planned.\n'
        f'Checkpoint {args.checkpoint},step{saved["step"]}. '
        'All928home scenes,four target queries per scene,identical current top RGB,joints and Gaussian noise. '
        'Seeds0/1/2,Euler10,H25;also evaluate the firstK10actions. No physical execution. '
        'References are four paired expert branches from the exact same home state. '
        'Measure correct/wrong target prediction error,query sensitivity,nearest expert branch selection, '
        'and fixed linear ridge(lambda10) target-slot/XYZ readout fit only on train activations. '
        'Chance and image-only controls25%. Linear decoding is not policy accuracy. '
        'Existing action model is fixed. Full training/auxiliary-head settings:config.json.\n')
    (out/'README.md').write_text(description)
    scenes=defaultdict(list)
    for row in [json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]:scenes[(row['layout_id'],row['permutation'])].append(row)
    scenes=[sorted(v,key=lambda r:r['target_id']) for k,v in sorted(scenes.items())]
    count=len(scenes);pred=np.empty((count,3,4,25,10),np.float32)
    expert=np.empty((count,4,25,10),np.float32);goals=np.empty((count,4,960),np.float32)
    xyz=np.empty((count,4,3),np.float32);slots=np.empty((count,4),np.int64)
    centers=np.empty_like(xyz) if saved['config'].get('aux_center') else None
    velocities=np.empty((count,4,25,10),np.float32)
    groups=np.array([s[0]['evaluation_group'] for s in scenes])
    config=dict(status='running',checkpoint=str(args.checkpoint),checkpoint_step=saved['step'],scenes=count,
                query_order=['RedCube','BlueCube','GreenCube','YellowCube'],query_intervention='change only target_id,hold RGB/joints/Gaussian noise fixed',
                noise_seeds=[0,1,2],flow_steps=10,H=25,executed_prefix=10,states='all independent dataset scene initial homes',
                image_source='same lossless start.png from target0 for every counterfactual query',
                branch_reference='all four physically demonstrated expert branches from exactly the same home state',
                metrics='query sensitivity,correct vs wrong query error,nearest expert branch retrieval,linear target slot/XYZ probe',
                action_training='none; checkpoint fixed; ridge readout fits only train home activations',
                training=saved['config'],ridge_lambda=10)
    write(out/'config.json',config);started=time.perf_counter()
    for begin in tqdm(range(0,count,16),desc='Same-input four-query causal probe',mininterval=5):
        batch=scenes[begin:begin+16];pixels=[];joints=[]
        for j,scene in enumerate(batch):
            assert [r['target_id'] for r in scene]==list(range(4))
            first=DATA/scene[0]['path']
            pixels.append(np.asarray(Image.open(first/'start.png').convert('RGB')))
            with np.load(first/'observation.npz') as z:state=z['joint_position'][0].astype(np.float32)
            joints.append(state)
            for target,row in enumerate(scene):
                lab=labels(row,np.array([0]));assert lab['mask'].all()
                np.testing.assert_array_equal(lab['state'][0],state)
                expert[begin+j,target]=lab['action'][0]
                slots[begin+j,target]=row['target_slot']
                with np.load(DATA/row['path']/'labels.npz') as z:xyz[begin+j,target]=z['entity_xyz_base_m'][0,target]
        n=len(batch)
        rgb=torch.from_numpy(np.stack(pixels)).permute(0,3,1,2).cuda()
        state=torch.from_numpy(np.stack(joints)).cuda().repeat_interleave(4,0)
        target=torch.arange(4,device='cuda').repeat(n)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            features=model.encode(rgb).repeat_interleave(4,0)
            hidden=model.visual_hidden(features,target)
            goal=model.goal(hidden)
            goals[begin:begin+n]=goal.float().cpu().numpy().reshape(n,4,960)
            if centers is not None:
                center=model.center_head(hidden).float()*model.center_std+model.center_mean
                centers[begin:begin+n]=center.cpu().numpy().reshape(n,4,3)
            context=model.conditioning(features,state,target)
            for seed in range(3):
                noise=model.flow.sample_noise(1,'cuda',torch.Generator(device='cuda').manual_seed(seed)).expand(n*4,-1,-1)
                if seed==0:
                    velocity=model.flow.velocity(noise,torch.zeros(n*4,device='cuda'),context)
                    velocities[begin:begin+n]=velocity.float().cpu().numpy().reshape(n,4,25,10)
                output=model.flow.sample_actions(noise,context,10)
                output=torch.cat((output[...,:9],-.0255*(output[...,9:]+1)/2),dim=-1)
                pred[begin:begin+n,seed]=output.float().cpu().numpy().reshape(n,4,25,10)
    assert np.isfinite(pred).all() and np.isfinite(goals).all()
    summary=summarize(pred,expert,groups,model.flow.action_std.cpu().numpy(),goals,velocities)
    write(out/'sensitivity.json',summary)
    print(json.dumps(dict(stage='counterfactual inference complete',elapsed_s=time.perf_counter()-started)),flush=True)
    readouts=ridge_probes(goals,slots,xyz,groups);write(out/'linear_probe.json',readouts)
    if centers is not None:
        errors=np.linalg.norm(centers-xyz,axis=-1)*1000
        nearest=np.linalg.norm(centers[:,:,None]-xyz[:,None],axis=-1).argmin(-1)
        center_metrics={}
        for group in sorted(set(groups)):
            use=groups==group;e=errors[use]
            center_metrics[group]=dict(queries=int(use.sum()*4),mean_mm=float(e.mean()),
                median_mm=float(np.median(e)),p90_mm=float(np.quantile(e,.9)),
                nearest_correct_object=float((nearest[use]==np.arange(4)[None]).mean()))
        write(out/'center_head.json',center_metrics)
    np.savez_compressed(out/'arrays.npz',predictions=pred,expert=expert,goals=goals,velocity=velocities,xyz_base_m=xyz,
                        slots=slots,groups=groups,scene_paths=np.array([s[0]['path'] for s in scenes]),
                        **({'center_predictions':centers} if centers is not None else {}))
    config.update(status='complete',elapsed_s=time.perf_counter()-started);write(out/'config.json',config)
    (out/'README.md').write_text(description.replace('Status: planned.','Status: complete.')+'\nActual results:sensitivity.json,linear_probe.json,arrays.npz.\n')
    for group,values in summary.items():
        print(json.dumps(dict(group=group,H25=values['H25'],sensitivity=values['layer_sensitivity'],linear_probe=readouts['results']['goal_token'][group])),flush=True)


if __name__=='__main__':main()
