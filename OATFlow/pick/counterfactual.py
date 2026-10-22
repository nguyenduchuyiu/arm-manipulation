"""Four cyclic color permutations and full target-counterfactual expert paths."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import imageio.v3 as iio
import mujoco
import numpy as np
from tqdm import tqdm

from OATFlow.environment.env import MemoryOcclusionEnv
from OATFlow.pick.scene import PERMUTATIONS, execute, setup
from OATFlow.task import TARGETS, normalized, physical_joint_action

TRAIN_PERMUTATIONS = (0, 9, 16, 18)
UNSEEN_PERMUTATIONS = (5, 23, 14, 7)


def restore(env, snapshot):
    mujoco.mj_resetData(env.model, env.data)
    for key in ('qpos', 'qvel', 'ctrl'):
        getattr(env.data, key)[:] = snapshot[key]
    mujoco.mj_forward(env.model, env.data)
    env.initial_positions = env.target_positions()


def write_episode(env, path, scene, permutation, target_id, split, anchor=None):
    if anchor is None:
        setup(env, scene, permutation)
    else:
        restore(env, anchor)
    initial = {key: getattr(env.data, key).copy() for key in ('qpos','qvel','ctrl')}
    target = TARGETS[target_id]
    path.mkdir(parents=True)
    writers = [iio.imopen(path/name,'w',plugin='pyav') for name in ('rgb.mp4','wrist_rgb.mp4')]
    for writer in writers:
        writer.init_video_stream('libx264',fps=25,pixel_format='yuv420p')
    state, actions, phases, valid, qpos, qvel = [], [], [], [], [], []
    def record(phase, action_valid=True):
        for writer, image in zip(writers,(env._overview(),env.wrist_image())):
            writer.write(image,is_batch=False)
        state.append(normalized(env.data.qpos[env.robot_qpos_addresses],env.model.actuator_ctrlrange))
        actions.append(normalized(env.data.ctrl,env.model.actuator_ctrlrange))
        phases.append(phase);valid.append(action_valid)
        qpos.append(env.data.qpos.copy());qvel.append(env.data.qvel.copy())
    try:
        record('pick_start',False)
        expert=execute(env,target,record)
        record('done',False)
    finally:
        for writer in writers:writer.close()
    n=len(phases)
    np.savez_compressed(path/'observation.npz',joint_position=np.array(state,np.float32),timestamp_s=np.arange(n)/25)
    np.savez_compressed(path/'supervision.npz',expert_action=np.array(actions,np.float32),action_valid=np.array(valid),phase=np.array(phases))
    np.savez_compressed(path/'sim_state.npz',initial_qpos=initial['qpos'],initial_qvel=initial['qvel'],initial_ctrl=initial['ctrl'],qpos=np.array(qpos),qvel=np.array(qvel))
    meta=dict(objective='target_lift',seed=scene['seed'],layout_seed=scene['seed'],scene=scene,permutation=permutation,
        slot_objects=PERMUTATIONS[permutation],target_slot=PERMUTATIONS[permutation].index(target),task_id=0,
        task_id_semantics='constant pick task; no slot information',target_id=target_id,target_object_id=target,
        selected_cover=None,frames=n,fps=25,resolution=[320,320],split=split,layout='standard',
        decision_frames=dict(t_occ=1,t_obj=1),action_representation='absolute_joint',success=True,**expert)
    (path/'episode.json').write_text(json.dumps(meta,indent=2)+'\n')
    return dict(path='',task_id=0,target_id=target_id,target_slot=meta['target_slot'],seed=scene['seed'],
        permutation=permutation,split=split,group='standard',frames=n)


def collect_pair(root, reference, layout, permutation, split, evaluation_group, with_cf, cores):
    identity=mp.current_process()._identity
    os.sched_setaffinity(0,{cores[(identity[0]-1) % len(cores)]})
    root,reference=Path(root),Path(reference)
    scene=json.loads((reference/f'g{layout:03d}/p00_t0/episode.json').read_text())['scene']
    env=MemoryOcclusionEnv()
    rows=[];groups=[]
    try:
        for target in range(4):
            relative=Path('nominal')/f'g{layout:03d}'/f'p{permutation:02d}_t{target}'
            row=write_episode(env,root/relative,scene,permutation,target,split)
            row.update(path=str(relative),kind='nominal',layout_index=layout,evaluation_group=evaluation_group)
            rows.append(row)
        if not with_cf:
            return rows,groups
        for source in rows.copy():
            path=root/source['path']
            with np.load(path/'supervision.npz') as z:
                valid=z['action_valid'];phases=z['phase'];actions=z['expert_action']
            with np.load(path/'sim_state.npz') as z:
                qpos=z['qpos'];qvel=z['qvel'];initial_ctrl=z['initial_ctrl']
            anchors=np.flatnonzero(valid & np.char.endswith(phases,'_approach'))[::10]
            for frame in anchors:
                if frame==1 and source['target_id']!=0:
                    continue
                anchor_id=f'g{layout:03d}_p{permutation:02d}_t{source["target_id"]}_f{frame:03d}'
                held_ctrl=initial_ctrl if frame==1 else physical_joint_action(actions[frame-1],env.model.actuator_ctrlrange)
                snapshot=dict(qpos=qpos[frame].copy(),qvel=qvel[frame].copy(),ctrl=held_ctrl.copy())
                restore(env,snapshot)
                np.savez_compressed(root/'anchors'/f'{anchor_id}.npz',**snapshot,
                    overview=env._overview(),wrist=env.wrist_image(),
                    proprio=normalized(env.data.qpos[env.robot_qpos_addresses],env.model.actuator_ctrlrange).astype(np.float32))
                branches=[]
                for target in range(4):
                    if frame==1 or target==source['target_id']:
                        branch_path=next(row['path'] for row in rows if row['kind']=='nominal' and row['target_id']==target)
                        branches.append(dict(target_id=target,kind='nominal_suffix',path=branch_path,frame=int(frame)))
                    else:
                        relative=Path('cf')/anchor_id/f't{target}'
                        row=write_episode(env,root/relative,scene,permutation,target,'train',snapshot)
                        row.update(path=str(relative),kind='counterfactual',layout_index=layout,anchor_id=anchor_id,
                            source_episode=source['path'],source_frame=int(frame),evaluation_group='train')
                        meta=json.loads((root/relative/'episode.json').read_text())
                        meta.update(counterfactual=True,anchor_id=anchor_id,source_episode=source['path'],source_frame=int(frame))
                        (root/relative/'episode.json').write_text(json.dumps(meta,indent=2)+'\n')
                        with np.load(root/relative/'sim_state.npz') as state:
                            np.testing.assert_array_equal(state['qpos'][1],snapshot['qpos'])
                            np.testing.assert_array_equal(state['qvel'][1],snapshot['qvel'])
                        rows.append(row)
                        branches.append(dict(target_id=target,kind='counterfactual',path=str(relative),frame=1))
                groups.append(dict(anchor_id=anchor_id,layout_index=layout,permutation=permutation,
                    source_path=source['path'],source_target=source['target_id'],source_frame=int(frame),branches=branches,
                    initial_reused=bool(frame==1)))
        return rows,groups
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not 1<=args.workers<=8:raise ValueError('workers must be1..8')
    args.output.mkdir(parents=True);(args.output/'anchors').mkdir()
    config=dict(source=str(args.source),reference=str(args.reference),train_permutations=TRAIN_PERMUTATIONS,
        unseen_permutations=UNSEEN_PERMUTATIONS,anchor_stride=10,training_stride=1,horizon=50,
        geometry='exact same slot positions,yaw,initial joints as e68,independent of permutation/query',
        counterfactual='restore full qpos/qvel and previous held command; replan real contact oracle for each new target',
        initial_states='existing four nominal goals reused; no duplicate initial CF trajectories',status='building')
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    readme='# Cyclic-permutation counterfactual data\n\nStatus: building. Four train layouts ×four cyclic permutations ×four targets, plus full CF branches every10approachframes. Every color occupies every slot. Exact geometry/yaw/initialpose held fixed across identity permutations; all new action labels come from actual successful contact-oracle rollouts. Original p0 CF reused unchanged. Training is stride1,H50,absolute6joint commands,25Hz; frozenViT BC initialization/training documented by downstream run. Source e70/e68; no state perturbation.\n\nTest plan: train64cases; spatial32cases; unseen-composition64cases on train geometry; combined32cases on held-out geometry. Four unseen reverse-cycle permutations are disjoint from train. Tests not yet run. Configuration: config.json.\n'
    (args.output/'README.md').write_text(readme)
    rows=[json.loads(line) for line in (args.source/'manifest.jsonl').read_text().splitlines() if line.strip()]
    for row in rows:
        path=args.output/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        path.symlink_to((args.source/row['path']).resolve(),target_is_directory=True)
        row['evaluation_group']='train'
    groups=json.loads((args.source/'groups.json').read_text())
    for group in groups:
        group['permutation']=0
        path=args.output/'anchors'/f'{group["anchor_id"]}.npz'
        path.symlink_to((args.source/'anchors'/path.name).resolve())
    for layout in (4,5):
        for target in range(4):
            relative=Path('nominal')/f'g{layout:03d}'/f'p00_t{target}'
            original=args.reference/f'g{layout:03d}'/f'p00_t{target}'
            path=args.output/relative;path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(original.resolve(),target_is_directory=True)
            meta=json.loads((original/'episode.json').read_text())
            rows.append(dict(path=str(relative),task_id=0,target_id=target,target_slot=target,seed=meta['seed'],
                permutation=0,split='test',group='standard',frames=meta['frames'],layout_index=layout,kind='nominal',evaluation_group='spatial'))
    jobs=[]
    for layout in range(6):
        for permutation in TRAIN_PERMUTATIONS[1:]:
            jobs.append((layout,permutation,'train' if layout<4 else 'test','train' if layout<4 else 'spatial',layout<4))
        for permutation in UNSEEN_PERMUTATIONS:
            jobs.append((layout,permutation,'test','composition' if layout<4 else 'combined',False))
    cores=sorted(os.sched_getaffinity(0))[:8]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(collect_pair,str(args.output),str(args.reference),*job,cores):job for job in jobs}
        for future in tqdm(as_completed(pending),total=len(pending),desc='permutation/CF groups',file=sys.stdout):
            batch,anchors=future.result();rows.extend(batch);groups.extend(anchors)
            (args.output/'progress.json').write_text(json.dumps(dict(completed_groups=len(rows),anchor_groups=len(groups),episodes=len(rows)))+'\n')
    rows.sort(key=lambda row:(row['layout_index'],row['permutation'],row['target_id'],row['kind'],row['path']))
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (args.output/'groups.json').write_text(json.dumps(groups,indent=2)+'\n')
    # Reuse the source normalization to keep the permutation experiment controlled.
    (args.output/'normalization.json').write_text((args.source/'normalization.json').read_text())
    counts={}
    for row in tqdm(rows,desc='audit valid chunks',file=sys.stdout):
        meta=json.loads((args.output/row['path']/'episode.json').read_text())
        assert meta['target_id']==row['target_id']==TARGETS.index(meta['target_object_id'])
        assert row['target_slot']==PERMUTATIONS[row['permutation']].index(meta['target_object_id'])
        with np.load(args.output/row['path']/'supervision.npz') as z:
            n=int(z['action_valid'].sum())
        counts[row['evaluation_group']]=counts.get(row['evaluation_group'],0)+n
    summary=dict(status='complete',episodes=len(rows),train_episodes=sum(row['split']=='train' for row in rows),
        train_chunks=counts['train'],anchor_groups=len(groups),chunks_by_group=counts,
        nominal_train=sum(row['split']=='train' and row['kind']=='nominal' for row in rows),
        counterfactual_train=sum(row['kind']=='counterfactual' for row in rows),all_expert_successful=True)
    (args.output/'generation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    config['status']='complete';(args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (args.output/'README.md').write_text(readme.replace('Status: building.','Status: complete.')+'\n'+json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
