"""Read-only attention, RGB position and XYZ translation probes of E122."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, median_filter
import torch
from torch.nn import functional as F
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import BASE, NAMES
from OATFlow.data_tools.render_dynamic import clip_planes
from OATFlow.simulation.render import visual_model
from OATFlow.top_pick.cloud import DATA
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.patch_fusion import PatchFusionPolicy
from OATFlow.top_pick.xyz_map import backproject, camera_calibration
from OATFlow.tracking.mask_video import font

PARENT=Path('/workspace/robotics_assets/experiments/e122_full')
TEST=PARENT/'test_xy'
CHECKPOINT=PARENT/'train/final.pt'
OUT=Path('/workspace/robotics_assets/experiments/e130_probe')
NOTE=Path('docs/research/e130_README.md')
SHIFTS=(-4,-2,-1,1,2,4)
DELTAS=[(axis,mm) for axis in range(3) for mm in (-100,-60,-40,-20,-10,-5,5,10,20,40,60,100)]


def state_hash(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cosine(a,b):
    return F.cosine_similarity(a,b,dim=-1).cpu().numpy()


def fractions(masks):
    pixels=torch.from_numpy((masks[:,None]==np.arange(1,5)[None,:,None,None]).astype(np.float32))
    return F.avg_pool2d(F.pad(pixels,(0,0,40,40)),16).flatten(2)


def pooled(features,weights):
    return torch.einsum('bkp,bpd->bkd',weights,features)/weights.sum(-1,keepdim=True).clamp_min(1e-12)


def cloud_metric(prediction,truth):
    if len(truth)<8:return dict(chamfer_mm=None,centroid_mm=None)
    d=np.linalg.norm(prediction[:,None]-truth[None],axis=-1)
    return dict(chamfer_mm=float((d.min(0).mean()+d.min(1).mean())*500),
                centroid_mm=float(np.linalg.norm(prediction.mean(0)-truth.mean(0))*1000))


def render_samples(rows):
    model=visual_model(DATA/'scene.xml',object_bodies=NAMES,lighting_scene=DATA/'scene.xml');clip_planes(model)
    model.vis.global_.offwidth=320;model.vis.global_.offheight=240
    state=mujoco.MjData(model);option=mujoco.MjvOption();option.geomgroup[3]=0
    lut=np.zeros(model.ngeom,np.uint8)
    for target,name in enumerate(NAMES[:4],1):lut[model.geom_bodyid==model.body(name).id]=target
    rgb_renderer=mujoco.Renderer(model,height=240,width=320)
    samples=model.vis.quality.offsamples;model.vis.quality.offsamples=0
    seg=mujoco.Renderer(model,height=240,width=320);seg.enable_segmentation_rendering();model.vis.quality.offsamples=samples
    data=[];source_hashes={}
    with rgb_renderer,seg:
        for row in tqdm(rows,desc='Render saved paired states',mininterval=5):
            path=TEST/(row['path'].replace('/','_')+'.npz');source_hashes[str(path)]=sha(path)
            with np.load(path) as z:
                seq=np.concatenate([z['initial_qpos'][None],z['qpos']],0)
                max_frame=min(285,(len(z['qpos'])-1)//15*15)
            for frame in [v for v in (0,75,150,225,285) if v<=max_frame]:
                state.qpos[:]=seq[frame];mujoco.mj_forward(model,state)
                rgb_renderer.update_scene(state,camera='cam_front',scene_option=option);rgb=rgb_renderer.render().copy()
                rgb_renderer.enable_depth_rendering();depth=rgb_renderer.render().copy();rgb_renderer.disable_depth_rendering()
                valid=np.isfinite(depth)&(depth>0)&(depth<=6.5535)
                codes=np.zeros(depth.shape,np.uint16);codes[valid]=np.rint(depth[valid]/.002).astype(np.uint16)
                seg.update_scene(state,camera='cam_front',scene_option=option);labels=seg.render()
                mask=np.zeros(depth.shape,np.uint8);valid_geom=labels[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM)
                mask[valid_geom]=lut[labels[...,0][valid_geom]]
                K,T=camera_calibration(model,state)
                target_mask=binary_erosion(mask==row['target_id']+1,iterations=1)&valid
                if target_mask.any():
                    median=np.median(codes[target_mask])*.002
                    target_mask&=np.abs(codes.astype(np.float64)*.002-median)<=.060
                    smooth=median_filter(codes,size=3,mode='nearest')
                    target_mask&=np.abs(codes.astype(np.int32)-smooth.astype(np.int32))*.002<=.010
                v,u=np.where(target_mask);d=codes[v,u].astype(float)*.002
                local=np.c_[(u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d]
                truth=local@T[:3,:3].T+T[:3,3]
                data.append(dict(row=row,frame=frame,rgb=rgb,depth=codes,K=K,T=T,masks=mask,truth=truth))
    return data,source_hashes


class AttentionCapture:
    """Read-only hooks; leave the model's original attention path untouched."""
    def __init__(self,model):
        self.weights={};self.error=0.;self.handles=[]
        for index,layer in enumerate(model.decoder.layers):
            self.handles.append(layer.multihead_attn.register_forward_hook(self.hook(index)))
    def hook(self,index):
        def capture(module,args,output):
            q,k,v=args[:3];dim=module.embed_dim;heads=module.num_heads;size=dim//heads
            assert module.dropout==0 and module.bias_k is None and module.bias_v is None
            qw,kw,vw=module.in_proj_weight.chunk(3);qb,kb,vb=module.in_proj_bias.chunk(3)
            qs=F.linear(q,qw,qb).reshape(len(q),-1,heads,size).transpose(1,2)
            ks=F.linear(k,kw,kb).reshape(len(k),-1,heads,size).transpose(1,2)
            vs=F.linear(v,vw,vb).reshape(len(v),-1,heads,size).transpose(1,2)
            weights=(qs@ks.transpose(-1,-2)/(size**.5)).softmax(-1)
            reconstructed=F.linear((weights@vs).transpose(1,2).reshape(len(q),-1,dim),module.out_proj.weight,module.out_proj.bias)
            self.error=max(self.error,float((reconstructed-output[0]).abs().max()))
            self.weights[index]=weights[:,:,0].detach()
        return capture
    def remove(self):
        for handle in self.handles:handle.remove()


def tensors(batch):
    rgb=torch.from_numpy(np.stack([d['rgb'] for d in batch])).permute(0,3,1,2).cuda()
    target=torch.tensor([d['row']['target_id'] for d in batch],device='cuda')
    scene={key:torch.as_tensor(np.stack([d[k] for d in batch]),device='cuda',dtype=torch.float32)
           for key,k in [('depth','depth'),('intrinsics','K'),('camera_to_base','T')]}
    return rgb,target,scene


@torch.inference_mode()
def attention_probe(model,data):
    capture=AttentionCapture(model);records=[];features=[];weights=[];predictions=[];parity=0.
    for start in tqdm(range(0,len(data),8),desc='Original decoder attention',mininterval=5):
        batch=data[start:start+8];rgb,target,scene=tensors(batch);f=model.encode(rgb)
        pred=model.predict_cloud(f,target,scene);att=torch.stack([capture.weights[i] for i in range(2)],1)
        features.append(f.cpu());weights.append(att.cpu().numpy());predictions.append(pred.cpu().numpy())
        cover=fractions(np.stack([d['masks'] for d in batch])).numpy();support=cover>0
        for j,d in enumerate(batch):
            t=d['row']['target_id'];layers=[]
            for layer in range(2):
                a=att[j,layer].mean(0).cpu().numpy();mass=(support[j]*a).sum(-1)
                prior=support[j].mean(-1);lift=mass/np.maximum(prior,1e-12)
                layers.append(dict(target_mass=float(mass[t]),target_uniform_lift=float(lift[t]),
                    target_peak_hit=bool(support[j,t,a.argmax()]),selected_object_by_mass=int(mass.argmax()),
                    selected_object_by_lift=int(lift.argmax()),object_masses=mass.tolist(),object_lifts=lift.tolist(),
                    target_peak_hits_per_head=[bool(support[j,t,h.argmax()]) for h in att[j,layer].cpu().numpy()]))
            records.append(dict(path=d['row']['path'],pair=d['row']['pair'],edge=d['row']['edge'],group=d['row']['evaluation_group'],
                target=t,frame=d['frame'],visible_target_pixels=int((d['masks']==t+1).sum()),**cloud_metric(pred[j].cpu().numpy(),d['truth']),layers=layers))
    capture.remove();assert capture.error<1e-5,capture.error
    f=torch.cat(features);pred=np.concatenate(predictions);att=np.concatenate(weights)
    for start in range(0,min(16,len(data)),8):
        _,target,scene=tensors(data[start:start+8]);unhooked=model.predict_cloud(f[start:start+8].cuda(),target,scene).cpu().numpy()
        parity=max(parity,float(abs(unhooked-pred[start:start+8]).max()))
    assert parity==0,parity
    return f,pred,att,records,dict(attention_reconstruction_max_error=capture.error,hook_cloud_parity_max_m=parity)


@torch.inference_mode()
def feature_probe(model,data,features):
    ids=[i for i,d in enumerate(data) if d['frame']==0];home=[data[i] for i in ids]
    covers=fractions(np.stack([d['masks'] for d in home])).cuda()
    f=features[ids].cuda();pooled_final=pooled(f,covers)
    rgb,_,_=tensors(home);square=(model.rgb_mean*255).expand(len(home),3,320,320).clone();square[:,:,40:280]=rgb
    patches=model.vision._process_input((square/255-model.rgb_mean)/model.rgb_std)
    raw_pool=pooled(patches,covers)
    records=[];rolled=[];control_min=1.;raw_min=1.
    for j,d in enumerate(home):
        t=d['row']['target_id']
        rolled_features=[]
        for shift in SHIFTS:
            p=patches[j:j+1].reshape(1,20,20,768).roll(shift,2).reshape(1,400,768)
            w=covers[j:j+1].reshape(1,4,20,20).roll(shift,3).reshape(1,4,400)
            tokens=torch.cat([model.vision.class_token,p],1)
            encoded=model.vision.encoder(tokens)[:,1:]
            position=model.vision.encoder.pos_embedding
            moved=torch.cat([position[:,:1],position[:,1:].reshape(1,20,20,768).roll(shift,2).reshape(1,400,768)],1)
            control=model.vision.encoder.ln(model.vision.encoder.layers(model.vision.encoder.dropout(tokens+moved)))[:,1:]
            fixed=pooled(encoded,w)[0,t];co_moved=pooled(control,w)[0,t];raw=pooled(p,w)[0,t]
            c_fixed=float(cosine(fixed[None],pooled_final[j,t][None])[0]);c_control=float(cosine(co_moved[None],pooled_final[j,t][None])[0])
            c_raw=float(cosine(raw[None],raw_pool[j,t][None])[0]);control_min=min(control_min,c_control);raw_min=min(raw_min,c_raw)
            rolled.append(dict(path=d['row']['path'],pair=d['row']['pair'],group=d['row']['evaluation_group'],shift_patches=shift,
                shift_pixels=shift*16,final_fixed_pos_cosine=c_fixed,final_comoved_pos_cosine=c_control,prepos_cosine=c_raw))
            rolled_features.append(fixed.cpu().numpy())
        negatives=[float(cosine(pooled_final[j,t][None],pooled_final[j,o][None])[0]) for o in range(4) if o!=t]
        records.append(dict(path=d['row']['path'],pair=d['row']['pair'],group=d['row']['evaluation_group'],target=t,
            mean_different_object_cosine=float(np.mean(negatives))))
    paired=[]
    for pair in range(8):
        ia=next(j for j,d in enumerate(home) if d['row']['pair']==pair and d['row']['evaluation_group']=='inside')
        ib=next(j for j,d in enumerate(home) if d['row']['pair']==pair and d['row']['evaluation_group']=='outside')
        t=home[ia]['row']['target_id']
        paired.append(dict(pair=pair,target=t,edge=home[ia]['row']['edge'],
            final_same_object_cosine=float(cosine(pooled_final[ia,t][None],pooled_final[ib,t][None])[0]),
            prepos_same_object_cosine=float(cosine(raw_pool[ia,t][None],raw_pool[ib,t][None])[0]),
            different_object_cosine=float(np.mean([records[ia]['mean_different_object_cosine'],records[ib]['mean_different_object_cosine']]))))
    assert control_min>.99999 and raw_min>.99999,(control_min,raw_min)
    np.savez_compressed(OUT/'features.npz',home_indices=ids,target_pooled_features=pooled_final.cpu().numpy(),
        target_pooled_prepos=raw_pool.cpu().numpy(),native_features=features.numpy())
    return paired,rolled,dict(comoved_position_min_cosine=control_min,prepos_roll_min_cosine=raw_min)


@torch.inference_mode()
def translation_probe(model,data,features,predictions):
    ids=[i for i,d in enumerate(data) if d['frame']==0];records=[];input_error=0.;zero_error=0.;allpred=[]
    for index in tqdm(ids,desc='Fixed RGB / XYZ translations',mininterval=5):
        d=data[index];_,target,scene=tensors([d]);f=features[index:index+1].cuda()
        baseline=model.predict_cloud(f,target,scene);zero_error=max(zero_error,float(abs(baseline.cpu().numpy()-predictions[index:index+1]).max()))
        xyz,valid=backproject(**scene)
        for start in range(0,len(DELTAS),12):
            delta=torch.zeros((len(DELTAS[start:start+12]),3),device='cuda')
            for j,(axis,mm) in enumerate(DELTAS[start:start+12]):delta[j,axis]=mm/1000
            expanded={k:v.expand(len(delta),*v.shape[1:]).clone() for k,v in scene.items()}
            expanded['camera_to_base'][:,:3,3]+=delta
            changed,mask=backproject(**expanded)
            difference=(changed-xyz-delta[:,:,None,None]).masked_select(mask.expand(-1,3,-1,-1))
            input_error=max(input_error,float(difference.abs().max()))
            fixed=f.expand(len(delta),-1,-1);assert torch.equal(fixed[0],f[0])
            cloud=model.predict_cloud(fixed,target.expand(len(delta)),expanded)
            response=cloud.mean(1)-baseline.mean(1);expected=baseline+delta[:,None]
            dist=torch.cdist(cloud,expected);chamfer=(dist.amin(1).mean(1)+dist.amin(2).mean(1))*500
            deform=((cloud-cloud.mean(1,keepdim=True))-(baseline-baseline.mean(1,keepdim=True))).square().sum(-1).mean(-1).sqrt()*1000
            for j,(axis,mm) in enumerate(DELTAS[start:start+12]):
                shift=response[j].cpu().numpy()*1000
                records.append(dict(path=d['row']['path'],pair=d['row']['pair'],group=d['row']['evaluation_group'],axis='XYZ'[axis],input_mm=mm,
                    output_shift_mm=shift.tolist(),gain=float(shift[axis]/mm),
                    off_axis_mm=float(np.linalg.norm(np.delete(shift,axis))),
                    equivariance_chamfer_mm=float(chamfer[j]),centered_shape_rms_mm=float(deform[j])))
            allpred.append(cloud.cpu().numpy())
    assert input_error<3e-6 and zero_error<1e-6,(input_error,zero_error)
    np.savez_compressed(OUT/'translations.npz',predictions=np.concatenate(allpred),input_deltas=np.array(DELTAS),home_indices=ids)
    return records,dict(prescribed_XYZ_translation_max_error_m=input_error,zero_translation_cloud_error_m=zero_error,RGB_features_fixed=True)


def attention_panel(d,weights,record,layer):
    a=weights[layer].mean(0).reshape(20,20)
    heat=np.asarray(Image.fromarray(a).resize((320,320),Image.Resampling.BILINEAR))[40:280]
    intensity=np.sqrt(heat/max(float(a.max()),1e-12));color=(colormaps['inferno'](intensity)[:,:,:3]*255)
    pixels=(d['rgb']*(1-.55*intensity[...,None])+color*(.55*intensity[...,None])).clip(0,255).astype(np.uint8)
    image=Image.new('RGB',(400,370),'white');image.paste(Image.fromarray(pixels).resize((400,300)),(0,36));draw=ImageDraw.Draw(image)
    t=d['row']['target_id'];y,x=np.where(d['masks']==t+1)
    if len(x):draw.rectangle((float(x.min()*1.25),float(y.min()*1.25+36),float((x.max()+1)*1.25),float((y.max()+1)*1.25+36)),outline='#19eea4',width=2)
    score=record['layers'][layer]
    draw.text((6,5),f'{record["group"]} L{layer+1} f{d["frame"]} | {NAMES[t]}',font=font(17,True),fill='#21354a')
    draw.text((6,341),f'mass {score["target_mass"]*100:.1f}% | lift {score["target_uniform_lift"]:.1f}x | Ch {record["chamfer_mm"]:.1f}mm' if record['chamfer_mm'] is not None else 'Target surface <8 pixels',font=font(16),fill='#21354a')
    return image


def figures(data,predictions,attention,records,paired,rolled,translations):
    for edge in ('x_low','x_high'):
        pairs=range(0,4) if edge=='x_low' else range(4,8)
        sheet=Image.new('RGB',(1600,1480),'white')
        history=Image.new('RGB',(1600,1480),'white')
        for row,pair in enumerate(pairs):
            inside=next(i for i,d in enumerate(data) if d['row']['pair']==pair and d['row']['evaluation_group']=='inside' and d['frame']==0)
            outside=next(i for i,d in enumerate(data) if d['row']['pair']==pair and d['row']['evaluation_group']=='outside' and d['frame']==0)
            late=max((i for i,d in enumerate(data) if d['row']['pair']==pair and d['row']['evaluation_group']=='outside'),key=lambda i:data[i]['frame'])
            for column,(index,layer) in enumerate(((inside,0),(inside,1),(outside,0),(outside,1))):
                sheet.paste(attention_panel(data[index],attention[index],records[index],layer),(column*400,row*370))
            for column,(index,layer) in enumerate(((outside,0),(outside,1),(late,0),(late,1))):
                history.paste(attention_panel(data[index],attention[index],records[index],layer),(column*400,row*370))
        sheet.save(OUT/f'attention_{edge}.png');history.save(OUT/f'attention_history_{edge}.png')
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for pair,ax in enumerate(axes.flat):
        for group,color in [('inside','#1673b1'),('outside','#de7626')]:
            i=next(i for i,d in enumerate(data) if d['row']['pair']==pair and d['row']['evaluation_group']==group and d['frame']==0)
            truth=data[i]['truth'];p=predictions[i]
            ax.scatter(truth[:,0]+BASE[0],truth[:,1]+BASE[1],s=6,c=color,alpha=.35)
            ax.scatter(p[:,0]+BASE[0],p[:,1]+BASE[1],s=9,c=color,marker='x',label=f'{group} prediction')
        ax.axvline(.30 if pair<4 else .78,c='gray',ls='--',lw=1);ax.set_aspect('equal',adjustable='datalim')
        ax.set_title(f'{"x_low" if pair<4 else "x_high"} / {NAMES[pair%4]}');ax.set_xlabel('world X (m)');ax.set_ylabel('world Y (m)')
        ax.legend(fontsize=7)
    fig.suptitle('Initial target surface: dots = measured GT-mask cloud; crosses = prediction; dashed = train edge')
    fig.savefig(OUT/'cloud_xy.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    x=np.arange(8)
    axes[0].plot(x,[p['final_same_object_cosine'] for p in paired],'o-',label='Same object, physical inside/outside')
    axes[0].plot(x,[p['different_object_cosine'] for p in paired],'s--',label='Different objects, same image')
    axes[0].plot(x,[p['prepos_same_object_cosine'] for p in paired],'.:',label='Same object, pre-position patches')
    axes[0].set(xlabel='Matched pair (0-3 x_low, 4-7 x_high)',ylabel='Cosine similarity');axes[0].legend(fontsize=8)
    for field,label in [('final_fixed_pos_cosine','Original fixed position'),('final_comoved_pos_cosine','Co-moved position control'),('prepos_cosine','Pre-position patches')]:
        yy=[[r[field] for r in rolled if r['shift_patches']==s] for s in SHIFTS]
        mean=np.array([np.mean(v) for v in yy]);lo=np.array([np.min(v) for v in yy]);hi=np.array([np.max(v) for v in yy])
        axes[1].plot(np.array(SHIFTS)*16,mean,'o-',label=label);axes[1].fill_between(np.array(SHIFTS)*16,lo,hi,alpha=.10)
    axes[1].set(xlabel='Identical content shifted horizontally (pixels)',ylabel='Target RGB feature cosine');axes[1].legend(fontsize=8)
    fig.suptitle('RGB feature probes; shaded range is 16 initial scenes, not a confidence interval')
    fig.savefig(OUT/'rgb_features.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for axis,ax in zip('XYZ',axes):
        for group,color in [('inside','#1673b1'),('outside','#de7626')]:
            delta=sorted({r['input_mm'] for r in translations})
            values=[[r['output_shift_mm']['XYZ'.index(axis)] for r in translations if r['axis']==axis and r['input_mm']==mm and r['group']==group] for mm in delta]
            ax.plot(delta,[np.median(v) for v in values],'o-',color=color,label=group)
            ax.fill_between(delta,[np.percentile(v,25) for v in values],[np.percentile(v,75) for v in values],color=color,alpha=.13)
        ax.plot([-100,100],[-100,100],'k--',lw=1,label='Ideal gain = 1');ax.set(xlabel=f'Input XYZ shift {axis} (mm)',ylabel=f'Output cloud centroid shift {axis} (mm)',title=axis)
        ax.legend(fontsize=8)
    fig.suptitle('Exact RGB features held fixed; only camera-to-base translation changes (median / IQR across 8 scenes)')
    fig.savefig(OUT/'xyz_response.png',dpi=170);plt.close(fig)


def summarize(records,paired,rolled,translations):
    attention={}
    for group in ('inside','outside'):
        attention[group]={}
        for period in ('initial','later'):
            selected=[r for r in records if r['group']==group and ((r['frame']==0)==(period=='initial'))]
            layers=[]
            for layer in range(2):
                layers.append(dict(mean_target_mass=float(np.mean([r['layers'][layer]['target_mass'] for r in selected])),
                    mean_uniform_lift=float(np.mean([r['layers'][layer]['target_uniform_lift'] for r in selected])),
                    peak_hits=sum(r['layers'][layer]['target_peak_hit'] for r in selected),
                    object_rank_hits=sum(r['layers'][layer]['selected_object_by_lift']==r['target'] for r in selected)))
            valid=[r for r in selected if r['chamfer_mm'] is not None]
            attention[group][period]=dict(samples=len(selected),valid_clouds=len(valid),layers=layers,
                mean_chamfer_mm=float(np.mean([r['chamfer_mm'] for r in valid])),mean_surface_centroid_error_mm=float(np.mean([r['centroid_mm'] for r in valid])))
    geometry={}
    for axis in 'XYZ':
        local=[r for r in translations if r['axis']==axis and abs(r['input_mm'])<=20]
        geometry[axis]=dict(local_median_gain=float(np.median([r['gain'] for r in local])),
            local_gain_IQR=np.percentile([r['gain'] for r in local],[25,75]).tolist(),
            local_median_equivariance_chamfer_mm=float(np.median([r['equivariance_chamfer_mm'] for r in local])))
    return dict(status='complete',attention=attention,physical_features=dict(
        mean_same_object_cosine=float(np.mean([p['final_same_object_cosine'] for p in paired])),
        mean_different_object_cosine=float(np.mean([p['different_object_cosine'] for p in paired])),
        mean_prepos_same_object_cosine=float(np.mean([p['prepos_same_object_cosine'] for p in paired]))),
        controlled_features=dict(mean_fixed_position_cosine=float(np.mean([r['final_fixed_pos_cosine'] for r in rolled])),
            mean_comoved_position_cosine=float(np.mean([r['final_comoved_pos_cosine'] for r in rolled]))),
        XYZ_translation=geometry,translation_trials=len(translations))


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=False);shutil.copyfile(NOTE,OUT/'README.md');started=time.perf_counter()
    torch.set_num_threads(4);torch.manual_seed(130);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((TEST/'config.json').read_text());checkpoint_hash=sha(CHECKPOINT)
    assert checkpoint_hash==protocol['checkpoint_sha256']
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);assert saved['step']==23520
    model=PatchFusionPolicy(initialize=False);model.load_state_dict(saved['model'],strict=True);model.requires_grad_(False);model.cuda().eval()
    config=dict(status='running',checkpoint=str(CHECKPOINT),checkpoint_sha256=checkpoint_hash,step=saved['step'],training=saved['config'],
        source_sha256=sha(Path(__file__)),device='CUDA',precision='FP32,TF32 disabled',architecture_changes=False,training_run=False,
        temporal_frames=[0,75,150,225,285],translation_deltas_axis_mm=DELTAS,feature_roll_patch_shifts=SHIFTS,
        probes='read-only decoder attention;physical/controlled RGB features;fixed-feature calibration translations')
    del saved;write(OUT/'config.json',config);before=state_hash(model)
    rows=list(map(json.loads,(TEST/'manifest.jsonl').read_text().splitlines()));assert len(rows)==16
    data,sources=render_samples(rows);features,pred,att,records,checks=attention_probe(model,data)
    write(OUT/'attention_records.json',records)
    paired,rolled,feature_checks=feature_probe(model,data,features);checks.update(feature_checks)
    write(OUT/'feature_pairs.json',paired);write(OUT/'feature_rolls.json',rolled)
    translations,translation_checks=translation_probe(model,data,features,pred);checks.update(translation_checks)
    write(OUT/'translation_records.json',translations)
    figures(data,pred,att,records,paired,rolled,translations)
    np.savez_compressed(OUT/'attention.npz',weights=att,predicted_clouds_base_m=pred,
        rgb=np.stack([d['rgb'] for d in data]),instance=np.stack([d['masks'] for d in data]),
        truth_xyz=np.concatenate([d['truth'] for d in data]),truth_offsets=np.r_[0,np.cumsum([len(d['truth']) for d in data])])
    assert state_hash(model)==before and sha(CHECKPOINT)==checkpoint_hash
    assert all(sha(Path(p))==v for p,v in sources.items())
    checks.update(status='passed',checkpoint_unchanged=True,model_state_unchanged=True,trace_hashes_unchanged=True,
                  no_architecture_changes=True,no_training=True,model_state_sha256=before)
    write(OUT/'verification.json',checks)
    summary=summarize(records,paired,rolled,translations);summary['elapsed_s']=time.perf_counter()-started
    write(OUT/'summary.json',summary);config.update(status='complete',samples=len(data),trace_hashes=sources);write(OUT/'config.json',config)
    text=NOTE.read_text().replace('Status: PLANNED.','Status: COMPLETE.')+'\nResults:\n```json\n'+json.dumps(summary,indent=2)+'\n```\n'
    NOTE.write_text(text);(OUT/'README.md').write_text(text);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
