"""Small live Track-On/context/FM manipulation rollouts, with frozen streaming memory."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import av
import mujoco
import numpy as np
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES
from OATFlow.environment.success import cover_deposited
from OATFlow.shuffle_bc.context_model import FrozenWristViT
from OATFlow.shuffle_bc.data import DATA
from OATFlow.shuffle_bc.evaluate import cases, prepare
from OATFlow.shuffle_bc.evaluate_context_bc import export_video
from OATFlow.shuffle_bc.trackon import FrozenTrackOn
from OATFlow.shuffle_bc.trackon_model import TrackOnContext
from OATFlow.shuffle_bc.train_trackon_xyz import sampled_frames
from OATFlow.simulation.contacts import jaw_contacts
from OATFlow.simulation.control import step as servo_step
from OATFlow.simulation.io import write_json
from OATFlow.simulation.pose import rotation_matrix
from OATFlow.simulation.render import visual_model
from OATFlow.tracking.linear_probe import fingerprint
from OATFlow.tracking.data import read_segment
from OATFlow.tracking.point_data import top_camera


@torch.inference_mode()
def replay_context(rows, scenes, tracker, camera):
    for row, scene in zip(rows, scenes):
        segment = read_segment(DATA, row['context_path'], 10, views=('top',), center_crop=True)
        np.testing.assert_array_equal(segment['raw_index'], sampled_frames(row['context_frames']))
        state = None
        for index in tqdm(range(len(segment['raw_index'])), desc='Causal recorded shuffle memory', mininterval=5, leave=False):
            with torch.autocast('cuda', dtype=torch.float16):
                features = tracker.features(segment['rgb'][0, index:index+1].cuda())
                if state is None:
                    state = tracker.initialize(features, torch.tensor(scene['uv'], device='cuda', dtype=torch.float32))
                hidden, uv, _ = tracker.step(features, state)
            assert torch.isfinite(hidden).all() and torch.isfinite(uv).all()
        scene['tracker_state'] = state
        scene['hidden'] = hidden.half()
        scene['hidden_frame'] = -1
        # The shared video renderer needs only frame0 for its prompt thumbnail.
        scene['images'] = [segment['rgb'][0, 0]]
        del segment


@torch.inference_mode()
def rollout(rows, policy, tracker, wrist, out, title):
    physics = mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
    visual = visual_model(DATA/'scene.xml', object_bodies=NAMES)
    visual.vis.global_.offwidth = visual.vis.global_.offheight = 320
    display = mujoco.MjData(visual); option = mujoco.MjvOption(); option.geomgroup[3] = 0
    scenes = prepare(rows, physics); camera = top_camera(DATA)
    for scene in scenes:
        scene['video_title'] = title
    replay_context(rows, scenes, tracker, camera)
    generator = torch.Generator(device='cuda').manual_seed(0)
    for s in scenes:
        s['predictions'] = []; s['prediction_truth'] = []; s['prediction_frames'] = []
    with mujoco.Renderer(visual, height=320, width=320) as renderer:
        for frame in tqdm(range(900), desc=f'Live policy {rows[0]["evaluation_group"]}', mininterval=10, leave=False):
            active = [s for s in scenes if not s['done']]
            if not active: break
            if frame % 100 == 0:
                write_json(out/'progress.json', dict(status='live rollout', group=rows[0]['evaluation_group'],
                    frame=frame, frame_budget=900, active=len(active)))
            if frame % 3 == 0 or frame % 10 == 0:
                for s in active:
                    display.qpos[:] = s['data'].qpos; mujoco.mj_forward(visual, display)
                    renderer.update_scene(display, camera='cam_top', scene_option=option)
                    s['current_rgb'] = torch.from_numpy(renderer.render()[40:280].copy()).permute(2, 0, 1)
                    renderer.update_scene(display, camera='cam_left', scene_option=option)
                    s['current_left'] = torch.from_numpy(renderer.render().copy()).permute(2, 0, 1)
                    if frame % 3 == 0:
                        with torch.autocast('cuda', dtype=torch.float16):
                            hidden, uv, visible = tracker.step(tracker.features(s['current_rgb'][None].cuda()), s['tracker_state'])
                        assert torch.isfinite(hidden).all() and torch.isfinite(uv).all()
                        s['hidden'] = hidden.half(); s['hidden_frame'] = frame
            if frame % 10 == 0:
                top = torch.stack([s['hidden'] for s in active])
                joints = torch.tensor(np.stack([s['data'].qpos[s['scene'].qadr] for s in active]), dtype=torch.float32, device='cuda')
                noise = policy.flow.sample_noise(1, 'cuda', generator).expand(len(active), -1, -1)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    side = wrist(torch.stack([s['current_left'] for s in active]).cuda()).half()
                    context, kv = policy.context(top, side, joints, with_kv=True)
                    predicted = policy.xyz_head(context).float()*policy.xyz_std+policy.xyz_mean
                    actions = policy.flow.sample_actions(noise, kv, steps=10)
                    actions = torch.cat((actions[..., :9], -.0255*(actions[..., 9:]+1)/2), -1)
                assert torch.isfinite(actions).all() and torch.isfinite(predicted).all()
                for i, s in enumerate(active):
                    s['plan'] = actions[i].float().cpu().numpy(); s['predicted_xyz'] = predicted[i].cpu().numpy()
                    s['anchor'] = s['data'].site_xpos[s['scene'].site].copy()
                    s['anchor_R'] = s['data'].site_xmat[s['scene'].site].reshape(3, 3).copy()
                    s['plans'].append(s['plan'].copy()); s['anchors'].append(s['anchor'].copy())
                    s['anchor_rotations'].append(s['anchor_R'].copy())
                    truth = s['data'].xpos[physics.body(NAMES[s['row']['target_id']]).id]-camera['base']
                    s['predictions'].append(s['predicted_xyz'].copy()); s['prediction_truth'].append(truth.copy())
                    s['prediction_frames'].append(frame)
            for s in active:
                state = s['data']; action = s['plan'][frame % 10]
                if frame % 3 == 0:
                    truth = state.xpos[physics.body(NAMES[s['row']['target_id']]).id]-camera['base']
                    s['video'].append((s['current_rgb'].permute(1, 2, 0).numpy(),
                        s['current_left'].permute(1, 2, 0).numpy(), s['predicted_xyz'].copy(), truth.copy()))
                command = s['ik'].command(s, s['anchor']+action[:3], action[9], rotation_matrix(action[3:9])@s['anchor_R'])
                command = np.clip(command, *physics.actuator_ctrlrange.T)
                servo_step(physics, state, s['held'], command); s['held'] = command.copy()
                objects = np.stack([state.xpos[physics.body(n).id] for n in NAMES])
                contacts = np.array([len(jaw_contacts(physics, state, n)) for n in NAMES])
                lifted = (objects[:, 2]-s['initial_objects'][:, 2] > .04) & (contacts == 2)
                target = s['row']['target_id']; s['streak'] = s['streak']+1 if lifted[target] else 0
                s['longest'] = max(s['longest'], s['streak'])
                s['cover_deposited'] = cover_deposited(physics, state, s['selected_cover'])
                s['other_cover_shift_m'] = float(np.linalg.norm(state.xpos[physics.body(s['other_cover']).id, :2]-s['other_cover_start'][:2]))
                if lifted[:4][np.arange(4) != target].any(): s.update(done=True, reason='wrong_object_lift')
                elif s['streak'] >= 10 and s['cover_deposited'] and s['other_cover_shift_m'] <= .015:
                    s.update(done=True, success=True, reason='success')
                for key in ('qpos', 'qvel', 'ctrl'): s[key].append(getattr(state, key).copy())
                s['commands'].append(command.copy()); s['objects'].append(objects); s['contacts'].append(contacts)
    results = []
    for s in scenes:
        row = s['row']; objects = np.array(s['objects'])
        result = dict(path=row['path'], group=row['evaluation_group'], target=NAMES[row['target_id']],
            success=s['success'], reason=s['reason'], frames=len(s['commands']), cover_deposited=s['cover_deposited'],
            other_cover_shift_m=s['other_cover_shift_m'], lifted_bilateral_streak=s['longest'],
            max_target_lift_m=float((objects[:, row['target_id'], 2]-s['initial_objects'][row['target_id'], 2]).max()),
            max_cover_lift_m=(objects[:, 4:, 2]-s['initial_objects'][4:, 2]).max(0).tolist(),
            query_uv=s['uv'].tolist(), xyz_mean_mm=float(np.linalg.norm(np.array(s['predictions'])-s['prediction_truth'], axis=-1).mean()*1000))
        np.savez_compressed(out/(row['path'].replace('/', '_')+'.npz'), **{key: np.array(s[key]) for key in
            ('initial_qpos', 'initial_qvel', 'initial_ctrl', 'commands', 'qpos', 'qvel', 'ctrl', 'objects', 'contacts',
             'plans', 'anchors', 'anchor_rotations', 'predictions', 'prediction_truth', 'prediction_frames')})
        result['video_frames_verified'] = export_video(s, result, out, camera)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = True
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(2.5 * 2**30 / total)
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e111_rollout_README.md', out/'README.md')
    write_json(out/'progress.json', dict(status='loading fixed checkpoint'))
    shutil.copyfile(args.checkpoint, out/'checkpoint.pt')
    saved = torch.load(out/'checkpoint.pt', map_location='cpu', mmap=True, weights_only=False)
    policy = TrackOnContext(); policy.load_state_dict(saved['model'], strict=True); policy.eval().cuda()
    tracker = FrozenTrackOn().cuda(); wrist = FrozenWristViT().cuda()
    frozen = dict(tracker=fingerprint(tracker), wrist=fingerprint(wrist))
    assert frozen == saved['config']['frozen_sha256']
    all_cases = cases()
    rows = [next(r for r in all_cases if r['evaluation_group'] == group and r['target_id'] == 0)
            for group in ('train', 'composition', 'spatial', 'combined')]
    config = dict(status='running', checkpoint=str(args.checkpoint), checkpoint_snapshot='checkpoint.pt',
        checkpoint_sha256=hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest(),
        checkpoint_epoch=saved['epoch'], checkpoint_step=saved['step'],
        completed_epoch_equivalent=saved['samples_seen']/saved['config']['points_per_epoch'],
        requested_epoch=10, exact_epoch10_available=False, H=25,K=10,flow_steps=10,noise_seed=0,frame_budget=900,
        cases=rows, training=saved['config'], observations='frame0 point + top sequence; current left RGB and measured joints',
        tracker_updates='recorded shuffle stride10+terminal, live rollout stride3; causal hold at intervening K10 replans (age0-2frames)',
        expert_future_loaded=False, GT_or_mask_policy_input=False, actual_wrist_camera=False,
        success='correct cover deposited; target lift>4cm with bilateral contact10frames; other cover shift<=15mm; wrong lift fails',
        video_fps=25,video_capture_stride=3,gpu_allocator_limit_gib=2.5,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),Path('OATFlow/shuffle_bc/evaluate_context_bc.py'),Path('OATFlow/shuffle_bc/evaluate.py'),
            Path('OATFlow/shuffle_bc/trackon.py'),Path('OATFlow/shuffle_bc/trackon_model.py'),
            Path('OATFlow/shuffle_bc/context_model.py'),Path('OATFlow/simulation/pose.py'),Path('OATFlow/simulation/control.py'))})
    write_json(out/'config.json',config)
    for name in config['source_sha256']:
        source=Path(name).resolve();dest=out/'source'/source.relative_to(Path.cwd())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    print(json.dumps(dict(checkpoint_step=saved['step'],epoch=config['completed_epoch_equivalent'],cases=[r['path'] for r in rows])),flush=True)
    title=f'e111 | step {saved["step"]} (~{config["completed_epoch_equivalent"]:.2f} epochs) | Live control'
    del saved
    started=time.perf_counter();results=[]
    for row in tqdm(rows,desc='Four fixed K10 demos',mininterval=5):
        current=rollout([row],policy,tracker,wrist,out,title);results.extend(current)
        with (out/'results.jsonl').open('a') as stream:
            for result in current:stream.write(json.dumps(result)+'\n')
        print(json.dumps(current[0]),flush=True)
    assert frozen==dict(tracker=fingerprint(tracker),wrist=fingerprint(wrist))
    paths=[out/(r['path'].replace('/','_')+'.mp4') for r in rows]
    command=['ffmpeg','-hide_banner','-loglevel','error']
    for path in paths:command.extend(['-i',str(path)])
    command.extend(['-filter_complex','[0:v]scale=640:400[a];[1:v]scale=640:400[b];[2:v]scale=640:400[c];[3:v]scale=640:400[d];[a][b][c][d]xstack=inputs=4:layout=0_0|640_0|0_400|640_400:fill=white[v]',
        '-map','[v]','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'comparison.mp4')])
    subprocess.run(command,check=True)
    with av.open(str(out/'comparison.mp4')) as video:frames=sum(1 for _ in video.decode(video=0))
    assert frames==max(r['frames'] for r in results)
    summary=dict(status='complete',cases=len(results),successes=sum(r['success'] for r in results),
        reasons=dict(Counter(r['reason'] for r in results)),results=results,checkpoint_step=config['checkpoint_step'],
        epoch_equivalent=config['completed_epoch_equivalent'],exact_epoch10=False,elapsed_s=time.perf_counter()-started,
        all_videos_fully_decoded=True,comparison_frames=frames,frozen_vision_unchanged=True,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(out/'summary.json',summary);write_json(out/'progress.json',summary)
    config['status']='complete';write_json(out/'config.json',config)
    with (out/'README.md').open('a') as stream:stream.write('\nActual result: '+json.dumps(summary)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
