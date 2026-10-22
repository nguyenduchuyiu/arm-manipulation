"""Audit e96 splits, physical replay, annotations and three-view pilot videos."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import av
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import OUTPUT,PILOT,NAMES,BASE,SEEN,UNSEEN,write_json
from OATFlow.data_tools.shuffle_data import arrays,episode,query_masks


def preview(root,rows):
    output=root/'preview'
    output.mkdir(exist_ok=False)
    (output/'README.md').write_text('# e96 three-view pilot\n\nActual recorded context + contact expert; no training/checkpoint. '
        '3RGB320px at25Hz; same context for4target IDs, distinct expert rollouts. '
        'Bottom: TCOW target amodal(yellow), frontmost cover(blue), containing cover(purple). '
        'Exact inputs/settings: ../config.json. No policy prediction or calibration export.\n')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    for row in tqdm(rows,desc='Pilot query previews',mininterval=5):
        data=episode(root,row['path']);mask=query_masks(root,row['path']);meta=data['metadata']
        video_frames=[]
        for view in ('top','left','right'):
            stream_frames=[]
            for path in data['rgb_segments'][view]:
                with av.open(str(path)) as video:
                    video.streams.video[0].thread_count=1
                    stream_frames.extend(frame.to_ndarray(format='rgb24') for frame in video.decode(video=0))
            video_frames.append(stream_frames)
        frames=meta['frames'];assert all(len(v)==frames for v in video_frames)
        path=output/f'query_{meta["target_id"]}.mp4'
        encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo',
            '-pix_fmt','rgb24','-s','960x720','-r','25','-i','pipe:0','-an','-c:v','libx264',
            '-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        for frame in tqdm(range(frames),desc=f'Preview {meta["target_object_id"]}',mininterval=10,leave=False):
            image=Image.new('RGB',(960,720),(19,25,34));draw=ImageDraw.Draw(image)
            for view in range(3):
                rgb=video_frames[view][frame]
                image.paste(Image.fromarray(rgb),(320*view,80))
                overlay=(rgb.astype(np.float32)*.25).astype(np.uint8)
                masks=np.unpackbits(mask['tcow_packed'][frame,view],axis=-1,count=320,bitorder='little').astype(bool)
                for channel,color in ((2,(190,80,235)),(1,(60,150,255)),(0,(255,210,40))):
                    overlay[masks[channel]]=color
                image.paste(Image.fromarray(overlay),(320*view,400))
            draw.text((12,5),f'e96 | query {meta["target_object_id"]} | t={frame/25:.2f}s | {data["supervision"]["phase"][frame]}',font=font,fill='white')
            draw.text((12,31),'Top: RGB top / left / right | Bottom: target amodal / frontmost occluder / container',font=font,fill='white')
            if frame in (0,meta['context_frames']-1,meta['decision_frames']['t_obj'],frames-1):
                image.save(output/f'query_{meta["target_id"]}_f{frame:04d}.jpg')
            encoder.stdin.write(np.asarray(image).tobytes())
        encoder.stdin.close();assert encoder.wait()==0
        with av.open(str(path)) as video:assert sum(1 for _ in video.decode(video=0))==frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args();root=args.data
    config=json.loads((root/'config.json').read_text())
    assert config['status']=='render_complete'
    rows=[json.loads(l) for l in (root/'manifest.jsonl').read_text().splitlines()]
    assert len(rows)==(4 if args.pilot else 3712)
    assert set(SEEN).isdisjoint(UNSEEN) and set(SEEN+UNSEEN)==set(range(24))
    grouped={}
    for row in rows:grouped.setdefault(row['context_path'],[]).append(row)
    assert len(grouped)==(1 if args.pilot else 928)
    if not args.pilot:
        for group in range(40):
            part=[r for r in rows if r['layout_id']==group]
            expected=set(SEEN) if 32<=group<36 else set(range(24))
            assert {r['permutation'] for r in part}==expected
            assert {r['final_permutation'] for r in part}==expected
    model=mujoco.MjModel.from_xml_path(str(root/'scene.xml'))
    checks=[];joint_sum=np.zeros(6);joint_sq=np.zeros(6);joint_count=0
    for context_ref,queries in tqdm(grouped.items(),desc='Audit shared states and episode labels',mininterval=5):
        assert {r['target_id'] for r in queries}==set(range(4))
        assert len({(r['split'],r['evaluation_group']) for r in queries})==1
        context=arrays(root/context_ref/'sim_state.npz')
        with np.load(root/context_ref/'masks.npz') as context_masks:
            assert np.all(context_masks['semantic_entity'][-1]==1)
        reference=None
        for row in queries:
            path=root/row['path'];state=arrays(path/'sim_state.npz');data=episode(root,row['path'])
            inputs=json.loads((path/'input.json').read_text())
            assert not {'state','masks','labels','selected_cover','task_id'}.intersection(inputs)
            meta=data['metadata'];n=row['frames'];nc=row['context_frames']
            initial=np.r_[state['initial_qpos'],state['initial_qvel'],state['initial_ctrl']]
            if reference is None:reference=initial
            else:np.testing.assert_array_equal(reference,initial)
            np.testing.assert_array_equal(state['initial_qpos'],context['qpos'][-1])
            np.testing.assert_array_equal(state['initial_qvel'],context['qvel'][-1])
            obs=data['observation'];sup=data['supervision'];labels=data['labels']
            np.testing.assert_array_equal(obs['timestamp_s'],np.arange(n)/25)
            assert not sup['action_valid'][:nc].any() and sup['action_valid'][nc:-1].all() and not sup['action_valid'][-1]
            assert meta['state_perturbations']==[] and meta['success']
            assert meta['bilateral_lift_streak_frames'][meta['target_object_id']]>=10
            for e,name in enumerate(NAMES):
                a=int(model.joint(name+'_joint').qposadr[0])
                xyz=np.concatenate((context['qpos'][:,a:a+3],state['qpos'][:,a:a+3]))
                np.testing.assert_allclose(labels['entity_xyz_world_m'][:,e],xyz,rtol=0,atol=1e-12)
            np.testing.assert_allclose(labels['entity_xyz_base_m'],labels['entity_xyz_world_m']-BASE,rtol=0,atol=1e-12)
            if row['split']=='train':
                q=obs['joint_position'][sup['action_valid']]
                joint_sum+=q.sum(0);joint_sq+=(q*q).sum(0);joint_count+=len(q)
            # Replay every pilot query and one context per layout in the full audit.
            if args.pilot or (row['permutation']==min(SEEN if 32<=row['layout_id']<36 else range(24))):
                replay=mujoco.MjData(model)
                replay.qpos[:],replay.qvel[:],replay.ctrl[:]=state['initial_qpos'],state['initial_qvel'],state['initial_ctrl']
                worst=0.
                for frame in range(len(state['qpos'])-1):
                    replay.ctrl[:]=state['ctrl'][frame]
                    mujoco.mj_forward(model,replay)
                    worst=max(worst,float(np.abs(replay.qpos-state['qpos'][frame]).max()))
                    mujoco.mj_step(model,replay,20)
                worst=max(worst,float(np.abs(replay.qpos-state['qpos'][-1]).max()))
                assert worst<1e-7,(row['path'],worst)
                checks.append(dict(path=row['path'],max_qpos_error=worst))
    mean=joint_sum/joint_count;std=np.sqrt(np.maximum(joint_sq/joint_count-mean**2,1e-12))
    write_json(root/'normalization.json',dict(applied=False,units=['rad']*5+['m'],
        measured_joint_train_mean=mean.tolist(),measured_joint_train_std=std.tolist(),
        measured_joint_train_frames=joint_count,gripper_fixed_range_m=[-.0255,0],
        gripper_training_formula='-1 - 2*g/.0255; absolute, no delta'))
    size=sum(p.stat().st_size for p in root.glob('g*/p*/**/*') if p.is_file())
    summary=dict(status='passed',episodes=len(rows),contexts=len(grouped),
        episodes_by_group=dict(Counter(r['evaluation_group'] for r in rows)),
        replay_checks=checks,bytes=size,estimated_full_bytes=size*(928 if args.pilot else 1),
        masks='full resolution3views; visible/amodal/occluder/container; all shared query IDs',
        layout_split='train0..31, val32..35, test36..39; composition intentionally uses train geometry',
        permutations_seen=SEEN,permutations_unseen=UNSEEN)
    write_json(root/'audit.json',summary)
    if args.preview:preview(root,rows[:4])
    config['status']='complete';write_json(root/'config.json',config)
    with (root/'README.md').open('a') as handle:
        handle.write(f'\nCompleted: {len(rows)}episodes/{len(grouped)}contexts. '
                     'Audit: audit.json; normalization.json. '
                     'No training. All stored videos fully decoded during rendering.\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
