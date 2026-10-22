"""Train dense RGB/XYZ-patch fusion on the E119 subset for 80 epochs."""
import gc
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time

import av
import mujoco
import numpy as np
import torch

from OATFlow.localization.model import state_hash
from OATFlow.top_pick.cloud import ROOT as PREVIOUS,SOURCE,DATA
from OATFlow.top_pick.cloud_data import jobs_and_labels,write
from OATFlow.top_pick.cloud_train import cache,train,batch,optimizer
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.fusion import ROOT,FusionPolicy
from OATFlow.top_pick.xyz_map import backproject,camera_calibration


def fresh():
    torch.manual_seed(102)
    return FusionPolicy().cuda()


def preflight(model,features,labels,ids,artifact_root=ROOT):
    source=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
    for name,value in model.flow.state_dict().items():
        assert torch.equal(value.cpu(),source['model']['flow.'+name]),name
    assert torch.equal(model.scene_mean.cpu(),source['model']['xyz_mean'])
    assert torch.equal(model.scene_std.cpu(),source['model']['xyz_std'])
    del source
    b=batch(features,labels,ids,model=model);scene=b[-1];frozen=state_hash(model.vision);model.eval()
    with torch.no_grad():
        xyz,valid=backproject(**scene)
        z=scene['depth'][0].cpu().numpy()*.002;K=scene['intrinsics'][0].cpu().numpy();T=scene['camera_to_base'][0].cpu().numpy()
        v,u=np.nonzero(z);chosen=np.random.default_rng(120).choice(len(u),128,replace=False);v,u=v[chosen],u[chosen]
        camera=np.c_[(u-K[0,2])*z[v,u]/K[0,0],(v-K[1,2])*z[v,u]/K[1,1],z[v,u]]
        reference=camera@T[:3,:3].T+T[:3,3]
        np.testing.assert_allclose(xyz[0,:,v,u].T.cpu().numpy(),reference,atol=2e-6,rtol=0)
        hidden=model.visual_hidden(b[0][:2],b[2][:2],{k:s[:2] for k,s in scene.items()})
        moved={k:s[:2].clone() for k,s in scene.items()};moved['camera_to_base'][:,:3,3]+=.02
        translated=model.visual_hidden(b[0][:2],b[2][:2],moved)
        difference=float((hidden-translated).abs().max());assert difference>1e-5
        assert model.geometry_tokens(scene).shape==(len(ids),model.position.shape[1],256)
        assert torch.isfinite(xyz).all() and valid.any()
    model.train();model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):parts=model.loss_parts(*b)
    parts[1].backward()
    for module in (model.geometry_patch,model.fusion[1],model.visual[1],model.target):
        assert module.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.flow.parameters())
    assert all(p.grad is None for p in model.goal.parameters())
    assert all(p.grad is None for p in model.vision.parameters())
    opt=optimizer(model);times=[];torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        started=time.perf_counter();opt.zero_grad(set_to_none=True)
        if features.dtype==torch.uint8:b=batch(features,labels,ids,model=model)
        with torch.autocast('cuda',dtype=torch.bfloat16):flow,aux,error=model.loss_parts(*b)
        loss=flow+aux;assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        assert model.flow.layers[-1].mlp.down_proj.weight.grad.abs().sum()>0
        opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-started)
    assert state_hash(model.vision)==frozen
    result=dict(status='passed',batch=128,dense_depth_resolution=[240,320],XYZ_channels=3,
        geometry_native_patches=400,fused_tokens=model.position.shape[1],patch_size=16,RGB_and_XYZ_letterbox_rows=[40,280],
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds_per_update=float(np.mean(times[1:])),estimated_training_hours=float(np.mean(times[1:])*23520/3600),
        online_frozen_ViT=features.dtype==torch.uint8,timing_includes_RGB_encode_and_depth_decode=features.dtype==torch.uint8,
        calibration_translation_changes_hidden=difference,backproject_matches_numpy=True,
        transferred_FM_exact=True,point_MLP_used=False,
        auxiliary_gradients='XYZ patch embedding,fusion,RGB projection,target decoder;not FM/goal/ViT',
        action_gradients_reach_FM=True,ViT_unchanged=True,
        supervision='same E119 target cloud labels; dense depth input independent of target')
    write(artifact_root/'preflight.json',result);print(json.dumps(result),flush=True)


def online_preflight(model,rows,artifact_root=ROOT):
    from OATFlow.data_tools.collect_shuffle import NAMES
    from OATFlow.data_tools.render_dynamic import clip_planes
    from OATFlow.simulation.render import visual_model
    from OATFlow.top_pick.cloud_evaluate import rollout
    physics=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
    visual=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml')
    clip_planes(visual)
    visual.vis.global_.offwidth=320;visual.vis.global_.offheight=240
    state=mujoco.MjData(visual)
    with np.load(DATA/rows[0]['path']/'sim_state.npz') as z:state.qpos[:]=z['initial_qpos']
    mujoco.mj_forward(visual,state);K,T=camera_calibration(visual,state)
    with np.load(DATA/rows[0]['path']/'calibration.npz') as z:
        np.testing.assert_allclose(K,z['intrinsics'][0],atol=2e-5,rtol=0)
        np.testing.assert_allclose(T,z['camera_to_base'][0,0],atol=2e-6,rtol=0)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    model.eval()
    with TemporaryDirectory(prefix='e120_online_') as tmp,mujoco.Renderer(visual,height=240,width=320) as renderer:
        renderer.update_scene(state,camera='cam_front',scene_option=option)
        renderer.enable_depth_rendering();depth=renderer.render();renderer.disable_depth_rendering()
        good=np.isfinite(depth)&(depth>0)&(depth<=6.5535)
        codes=np.zeros(depth.shape,np.uint16);codes[good]=np.rint(depth[good]/.002).astype(np.uint16)
        with av.open(str(DATA/rows[0]['path']/'depth_front.mkv')) as video:
            stored=next(video.decode(video=0)).to_ndarray(format='gray16le')
        difference=np.abs(codes.astype(np.int32)-stored.astype(np.int32))
        mismatch=float((difference>0).mean());assert mismatch<.01
        assert float(np.percentile(difference,99))<=1
        results=rollout(rows[:4],model,physics,visual,renderer,option,Path(tmp),frame_budget=20)
    assert len(results)==4 and all(r['frames']>0 for r in results)
    write(artifact_root/'online_preflight.json',dict(status='passed',cases=4,frames_per_case=20,H=25,K=10,
        calibration_matches_dataset=True,online_depth_pixel_mismatch_fraction=mismatch,
        online_depth_p99_error_mm=float(np.percentile(difference,99)*2),finite_online_depth_to_FM_to_IK=True,
        accuracy_result=False,temporary_physics_traces_removed=True))


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    ROOT.mkdir(exist_ok=False);shutil.copyfile('docs/research/e120_README.md',ROOT/'README.md')
    for name in ['partitions.json','train_manifest.jsonl','val_manifest.jsonl','test_manifest.jsonl',
                 *[f'part_{k:02d}.jsonl' for k in range(1,9)]]:
        shutil.copyfile(PREVIOUS/name,ROOT/name)
    manifests={k:[json.loads(s) for s in (ROOT/f'{k}_manifest.jsonl').read_text().splitlines()]
               for k in ('train','val','test')}
    jobs,labels=jobs_and_labels(manifests)
    write(ROOT/'cache_jobs.json',[dict(path=r['path'],starts=s.tolist(),offset=o) for r,s,o in jobs])
    model=fresh().eval();features,labels,mean,std=cache(model,jobs,labels,artifact_root=ROOT,include_depth=True)
    with np.load(PREVIOUS/'labels.npz') as old,np.load(ROOT/'labels.npz') as new:
        for key in old.files:np.testing.assert_array_equal(old[key],new[key])
    write(ROOT/'label_alignment.json',dict(status='passed',all_original_E119_labels_bit_exact=True,
        subset_sha256=sha(ROOT/'partitions.json'),geometry_input='full decoded depth+intrinsics+camera-to-base only;no target mask/pose',
        target_cloud_and_action_labels='unchanged from E119'))
    features=torch.from_numpy(features).cuda()
    labels={k:(v if k=='depth' else torch.from_numpy(v)) for k,v in labels.items()}
    model.cloud_mean.copy_(torch.tensor(mean,device='cuda'));model.cloud_std.copy_(torch.tensor(std,device='cuda'))
    frozen=state_hash(model.vision);ids=np.flatnonzero(labels['is_train'].numpy())[:128]
    preflight(model,features,labels,ids);online_preflight(model,manifests['train'])
    del model;gc.collect();torch.cuda.empty_cache();model=fresh()
    assert state_hash(model.vision)==frozen
    overrides=dict(experiment='e120_fusion',inputs='full front RGB + dense depth + intrinsics/extrinsics + targetID + measured joints',
        point_encoder='none;deterministic full-resolution XYZ map followed by fresh trainable Conv3to256 patch16',
        architecture='RGB frozen ViT64tokens + aligned XYZ64tokens -> channel fusion64tokens -> two256D target-query blocks ->960Dgoal + state ->FM',
        initialization='FM from E118best120k;ImageNet ViT frozen;fresh XYZ patch/fusion/query/decoder/goal/cloud head;E118point MLP not used',
        trainable='XYZ patch embedding,geometry norm,fusion,target-query decoder,visual/goal projections,aux cloud head,whole FM/state encoder',
        source_E119_weights=None,budget='80epochs=23520updates;approximately full E11710epoch budget23020updates',
        geometry=dict(native_resolution=[240,320],depth_unit_m=.002,XYZ_frame='robot base',
                      normalization='E118global train-only xyz mean/std;no per-image centering',
                      patch=16,letterbox=[40,40],grid_before_pool=[20,20],grid_after_pool=[8,8],
                      input_pixel_sampling=False,ROI_filter=False,GTmask_input=False,
                      invalid_depth='zero normalized XYZ before patch embedding',RGB_fusion='concat channels512 then LN+Linear256'),
        cache='RGB FP16 features in GPU RAM;raw uint16 depth and calibration in CPU RAM;XYZ computed per batch;no disk feature cache',
        code_fusion_sha256={n:sha(Path(__file__).with_name(n)) for n in ['fusion.py','fusion_train.py','xyz_map.py']})
    train(model,features,labels,mean,std,out=ROOT/'train',epochs=80,artifact_root=ROOT,config_overrides=overrides)


if __name__=='__main__':main()
