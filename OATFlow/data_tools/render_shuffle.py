"""Render synchronized RGB, instance masks and query TCOW masks for e96."""
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time

import av
import mujoco
import numpy as np
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import OUTPUT, PILOT, NAMES, write_json
from OATFlow.simulation.render import visual_model
from OATFlow.simulation.io import encoder

VIEWS=('top','left','right')


def worker_init(root):
    global MODEL,STATE,RGB,SEG,OPTION,ONLY,LUT,GEOMS,OUT
    OUT=Path(root)
    MODEL=visual_model(OUT/'scene.xml',object_bodies=NAMES)
    MODEL.vis.global_.offwidth=MODEL.vis.global_.offheight=320
    STATE=mujoco.MjData(MODEL)
    RGB=mujoco.Renderer(MODEL,height=320,width=320,max_geom=1000)
    samples=MODEL.vis.quality.offsamples
    MODEL.vis.quality.offsamples=0
    SEG=mujoco.Renderer(MODEL,height=320,width=320,max_geom=1000)
    MODEL.vis.quality.offsamples=samples
    SEG.enable_segmentation_rendering()
    atexit.register(RGB.close);atexit.register(SEG.close)
    OPTION=mujoco.MjvOption();OPTION.geomgroup[3]=0
    ONLY=mujoco.MjvOption();ONLY.geomgroup[:]=0;ONLY.geomgroup[5]=1
    LUT=np.zeros(MODEL.ngeom,np.uint8)
    GEOMS={}
    for index,name in enumerate(NAMES,1):
        body=MODEL.body(name).id
        ids=np.flatnonzero((MODEL.geom_bodyid==body)&(MODEL.geom_rgba[:,3]>0))
        GEOMS[index]=ids
        LUT[MODEL.geom_bodyid==body]=index
    for g in range(MODEL.ngeom):
        name=MODEL.body(int(MODEL.geom_bodyid[g])).name
        if name not in NAMES and MODEL.geom_bodyid[g]!=0:LUT[g]=7
    LUT[MODEL.geom('table_top').id]=8


def containers():
    owners=np.zeros(4,np.uint8)
    for target in range(4):
        body=MODEL.body(NAMES[target]).id
        p=STATE.xpos[body]
        for cover in (5,6):
            owner=MODEL.body(NAMES[cover-1]).id
            local=STATE.xmat[owner].reshape(3,3).T@(p-STATE.xpos[owner])
            # Same e36 containment convention, including the settled/table tolerance.
            if abs(local[0])<.073 and abs(local[1])<.058 and local[2]-.016>=-.005 and local[2]+.016<=.068:
                assert owners[target]==0
                owners[target]=cover
    return owners


def render(job):
    ref,target=job;directory=OUT/ref
    if (directory/'render_complete.json').exists():
        return json.loads((directory/'render_complete.json').read_text())
    context=target is None
    meta=json.loads((directory/('context.json' if context else 'episode.json')).read_text())
    with np.load(directory/'sim_state.npz') as archive:
        state={k:archive[k] for k in archive.files}
    supervision=np.load(directory/'supervision.npz')
    phases=supervision['phase'];n=len(phases)
    ids=list(range(1,7)) if context else [target+1,5,6]
    targets=list(range(4)) if context else [target]
    amodal=np.empty((n,3,len(ids),320,40),np.uint8)
    instance=np.empty((n,3,320,320),np.uint8)
    front=np.zeros((n,3,len(targets)),np.uint8)
    owner=np.zeros((n,len(targets)),np.uint8)
    fraction=np.zeros((n,3,len(targets)),np.float32)
    entity=np.full((n,len(targets)),2,np.uint8)
    relevance=None if context else np.zeros((n,3,320,320),np.uint8)
    tcow=None if context else np.zeros((n,3,3,320,40),np.uint8)
    hidden=np.zeros(len(targets),bool)
    first_hidden=np.full(len(targets),-1,np.int32)
    writers={view:encoder(directory/f'rgb_{view}.partial.mp4',320,320) for view in VIEWS}
    start=time.perf_counter()
    for frame in tqdm(range(n),desc=f'RGB/masks {ref}',mininterval=10,leave=False):
        STATE.qpos[:],STATE.qvel[:],STATE.ctrl[:]=state['qpos'][frame],state['qvel'][frame],state['ctrl'][frame]
        mujoco.mj_forward(MODEL,STATE)
        owner[frame]=containers()[targets]
        for view_index,view in enumerate(VIEWS):
            camera='cam_'+view
            RGB.update_scene(STATE,camera=camera,scene_option=OPTION)
            writers[view].stdin.write(RGB.render().tobytes())
            SEG.update_scene(STATE,camera=camera,scene_option=OPTION)
            segmentation=SEG.render()
            valid=segmentation[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)
            visible=np.zeros((320,320),np.uint8)
            visible[valid]=LUT[segmentation[:,:,0][valid]]
            instance[frame,view_index]=visible
            masks={}
            for channel,index in enumerate(ids):
                geoms=GEOMS[index]
                previous=MODEL.geom_group[geoms].copy()
                MODEL.geom_group[geoms]=5
                SEG.update_scene(STATE,camera=camera,scene_option=ONLY)
                mask=SEG.render()[:,:,0]>=0
                MODEL.geom_group[geoms]=previous
                masks[index]=mask
                amodal[frame,view_index,channel]=np.packbits(mask,axis=-1,bitorder='little')
            for ti,t in enumerate(targets):
                mask=masks[t+1];area=int(mask.sum())
                assert area>0,(ref,frame,view,'target out of view')
                assert not np.any((visible==t+1)&~mask),(ref,frame,view,'visible mask outside amodal')
                fraction[frame,view_index,ti]=1-int((visible==t+1).sum())/area
                cover_fractions=[np.count_nonzero(mask&(visible==c))/area for c in (5,6)]
                dominant=int(np.argmax(cover_fractions))
                if fraction[frame,view_index,ti]>=.95 and cover_fractions[dominant]>=.475:
                    front[frame,view_index,ti]=5+dominant
                if not context:
                    tcow[frame,view_index,0]=amodal[frame,view_index,0]
                    for channel,index in ((1,front[frame,view_index,ti]),(2,owner[frame,ti])):
                        if index:
                            tcow[frame,view_index,channel]=amodal[frame,view_index,ids.index(int(index))]
        if context:
            for ti,t in enumerate(targets):
                if phases[frame]=='occlude' and not np.any(instance[frame]==t+1):
                    hidden[ti]=True
                    if first_hidden[ti]<0:first_hidden[ti]=frame
                entity[frame,ti]=1 if hidden[ti] else 2
            if phases[frame] in ('shuffle','shuffle_pause','hold'):
                assert not np.any((instance[frame]>=1)&(instance[frame]<=4)),(ref,frame,'object visible during shuffle')
                assert np.all(owner[frame]>0),(ref,frame,'object outside its cover')
        else:
            semantic=0 if phases[frame]=='done' else (2 if str(phases[frame]).startswith(NAMES[target]+'_') else 1)
            entity[frame,0]=semantic
            if semantic:
                label=target+1 if semantic==2 else NAMES.index(meta['selected_cover'])+1
                relevance[frame][instance[frame]==label]=semantic
    for writer in writers.values():writer.stdin.close()
    for view,writer in writers.items():
        assert writer.wait()==0
        path=directory/f'rgb_{view}.partial.mp4'
        with av.open(str(path)) as video:
            video.streams.video[0].thread_count=1
            assert sum(1 for _ in video.decode(video=0))==n
        path.replace(directory/f'rgb_{view}.mp4')
    payload=dict(instance=instance,amodal_packed=amodal,amodal_entity_ids=np.array(ids,np.uint8),
        frontmost_cover_ids=front,container_ids=owner,occlusion_fraction=fraction,
        semantic_entity=entity,target_ids=np.array(targets,np.uint8),timestamp_s=state['timestamp_s'],
        resolution=np.array([320,320]),bitorder=np.array('little'))
    if context:
        assert np.all(first_hidden>=0)
        payload['fully_occluded_frame']=first_hidden
    else:
        # The query TCOW array already contains its target and relevant covers.
        del payload['amodal_packed'],payload['amodal_entity_ids']
        payload.update(tcow_packed=tcow,relevance=relevance)
    # Time-contiguous storage improves temporal lossless compression without
    # changing logical array shapes or the np.load interface.
    np.savez_compressed(directory/'masks.partial.npz',
        **{k:np.asfortranarray(v) if v.ndim>=3 else v for k,v in payload.items()})
    (directory/'masks.partial.npz').replace(directory/'masks.npz')
    if not context:
        write_json(directory/'input.json',dict(target_id=target,views=list(VIEWS),
            rgb_segments={v:[f'../rgb_{v}.mp4',f'rgb_{v}.mp4'] for v in VIEWS},
            context_path='..',expert_path='.',context_frames=meta['context_frames'],expert_frames=n,
            observations=['../observation.npz','observation.npz']))
    result=dict(path=ref,kind='context' if context else 'expert',frames=n,
        elapsed_s=time.perf_counter()-start,bytes=sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
        fully_decoded_views=3,mask_checks='visible subset of amodal; per-view occlusion/containment; context fully hidden')
    write_json(directory/'render_complete.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();root=args.data
    config=json.loads((root/'config.json').read_text())
    assert config['status'] in ('physics_complete','rendering','render_complete')
    rows=[json.loads(l) for l in (root/'manifest.jsonl').read_text().splitlines()]
    contexts=sorted({r['context_path'] for r in rows})
    jobs=[(ref,None) for ref in contexts]+[(r['path'],r['target_id']) for r in rows]
    config['status']='rendering';config['render_workers']=args.workers
    write_json(root/'config.json',config)
    start=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                             initializer=worker_init,initargs=(str(root),)) as pool:
        futures=[pool.submit(render,job) for job in jobs]
        for future in tqdm(as_completed(futures),total=len(futures),desc='Render RGB3 + lossless masks',mininterval=5):
            results.append(future.result())
            write_json(root/'render_progress.json',dict(segments=len(results),total=len(jobs),
                frames=sum(r['frames'] for r in results),elapsed_s=time.perf_counter()-start))
    config['status']='render_complete';config['render_elapsed_s']=time.perf_counter()-start
    config['render_segments']=len(results)
    write_json(root/'config.json',config)
    write_json(root/'render_summary.json',dict(segments=results,elapsed_s=time.perf_counter()-start,
        total_bytes=sum(r['bytes'] for r in results),unique_frames=sum(r['frames'] for r in results)))
    print(json.dumps(config),flush=True)


if __name__=='__main__':main()
