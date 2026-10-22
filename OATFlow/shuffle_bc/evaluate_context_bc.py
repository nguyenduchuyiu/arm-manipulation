"""Small K10 closed-loop rollout of e109 with live top/left images and videos."""
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
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES
from OATFlow.environment.success import cover_deposited
from OATFlow.shuffle_bc.context_model import ContextPolicy
from OATFlow.shuffle_bc.context_video import font, marker, GREEN, PINK, ORANGE
from OATFlow.shuffle_bc.data import DATA, select_indices
from OATFlow.shuffle_bc.evaluate import cases, prepare
from OATFlow.shuffle_bc.train_context_bc import load_state
from OATFlow.simulation.contacts import jaw_contacts
from OATFlow.simulation.control import step as servo_step
from OATFlow.simulation.io import encoder, write_json
from OATFlow.simulation.pose import rotation_matrix
from OATFlow.simulation.render import visual_model
from OATFlow.tracking.point_data import project, top_camera
from OATFlow.tracking.linear_probe import fingerprint


def selected_cases():
    result = []
    for group in ('train', 'composition', 'spatial', 'combined'):
        rows = [r for r in cases() if r['evaluation_group'] == group]
        chosen = [r for r in rows if r['context_path'] == rows[0]['context_path']]
        assert len(chosen) == 4 and {r['target_id'] for r in chosen} == {0, 1, 2, 3}
        result.extend(chosen)
    return result


def video_frame(s, frame, sample, result, camera):
    top, left, predicted, gt = sample
    canvas = Image.new('RGB', (1280, 800), '#f5f7fa'); draw = ImageDraw.Draw(canvas)
    row = s['row']; split = dict(train='TRAIN', composition='UNSEEN ORDER', spatial='UNSEEN LAYOUT', combined='BOTH UNSEEN')
    draw.text((30, 18), s.get('video_title', 'e109 | Closed-loop cover opening + pick'), font=font(27, True), fill='#182638')
    draw.text((30, 61), f'{split[row["evaluation_group"]]} | {row["path"]} | target: {NAMES[row["target_id"]]}', font=font(19), fill='#43536a')
    canvas.paste(Image.fromarray(top).resize((800, 600)), (30, 130))
    canvas.paste(Image.fromarray(left).resize((360, 360)), (850, 130))
    draw.text((30, 98), 'LIVE TOP VIEW', font=font(18, True), fill='#182638')
    draw.text((850, 98), 'LIVE LEFT INPUT (surrogate wrist)', font=font(15, True), fill='#182638')
    for xyz, color, cross in ((gt, GREEN, False), (predicted, PINK, True)):
        uv = project(xyz, camera)*2.5+[30, 130]
        if 30 <= uv[0] <= 830 and 130 <= uv[1] <= 730:
            marker(draw, uv, color, cross)
    initial = s['images'][0].permute(1, 2, 0).numpy()
    canvas.paste(Image.fromarray(initial).resize((160, 120)), (850, 535))
    marker(draw, s['uv']*.5+[850, 535], ORANGE, radius=4)
    draw.text((850, 505), 'FRAME0 POINT PROMPT', font=font(16, True), fill='#182638')
    draw.text((1030, 539), f'{frame/25:.2f} s', font=font(24, True), fill='#182638')
    draw.text((1030, 579), 'H25 / K10', font=font(18), fill='#43536a')
    error = np.linalg.norm(predicted-gt)*1000
    draw.text((850, 677), f'XYZ error: {error:.1f} mm', font=font(23, True), fill=PINK)
    status = 'RUNNING' if frame+3 < result['frames'] else result['reason'].upper().replace('_', ' ')
    draw.text((30, 750), status, font=font(20, True), fill=GREEN if result['success'] else '#43536a')
    draw.text((360, 750), 'FM controls robot | green: GT | magenta: predicted center', font=font(19), fill='#43536a')
    return canvas


def export_video(s, result, out, camera):
    name = result['path'].replace('/', '_'); path = out/(name+'.mp4')
    writer = encoder(path, 1280, 800)
    for i, sample in enumerate(tqdm(s['video'], desc=f'Video {name}', mininterval=10, leave=False)):
        image = video_frame(s, i*3, sample, result, camera)
        for _ in range(min(3, result['frames']-i*3)):
            writer.stdin.write(np.asarray(image).tobytes())
        if i in (0, len(s['video'])-1): image.save(out/f'{name}_{"start" if i == 0 else "end"}.png')
    writer.stdin.close(); assert writer.wait() == 0
    with av.open(str(path)) as video: decoded = sum(1 for _ in video.decode(video=0))
    assert decoded == result['frames']
    return decoded


@torch.inference_mode()
def rollout(rows, policy, out):
    physics = mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
    visual = visual_model(DATA/'scene.xml', object_bodies=NAMES)
    visual.vis.global_.offwidth = visual.vis.global_.offheight = 320
    display = mujoco.MjData(visual); option = mujoco.MjvOption(); option.geomgroup[3] = 0
    scenes = prepare(rows, physics); camera = top_camera(DATA)
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
                        s['images'].append(s['current_rgb'])
                        s['image_times'].append(s['row']['context_frames']+frame)
            if frame % 10 == 0:
                rgb = []; joints = []
                for s in active:
                    images = s['images'] if frame % 3 == 0 else s['images']+[s['current_rgb']]
                    times = s['image_times'] if frame % 3 == 0 else s['image_times']+[s['row']['context_frames']+frame]
                    chosen = select_indices(np.array(times), len(images)-1)
                    rgb.append(torch.stack([images[i] for i in chosen]).permute(1, 0, 2, 3)[None])
                    joints.append(s['data'].qpos[s['scene'].qadr])
                noise = policy.flow.sample_noise(1, 'cuda', generator).expand(len(active), -1, -1)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    top, wrist = policy.encode(torch.cat(rgb).cuda(), torch.cat([s['query'] for s in active]).cuda(),
                        torch.stack([s['current_left'] for s in active]).cuda())
                    context, kv = policy.context(top.half(), wrist.half(), torch.tensor(np.array(joints), dtype=torch.float32, device='cuda'), with_kv=True)
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


def montage(rows, out):
    paths = [out/(r['path'].replace('/', '_')+'.mp4') for r in rows if r['target_id'] == 0]
    assert len(paths) == 4
    args = ['ffmpeg', '-loglevel', 'error']
    for path in paths: args += ['-i', str(path)]
    args += ['-filter_complex', '[0:v]scale=640:400[a];[1:v]scale=640:400[b];[2:v]scale=640:400[c];[3:v]scale=640:400[d];[a][b][c][d]xstack=inputs=4:layout=0_0|640_0|0_400|640_400:fill=white[v]',
        '-map', '[v]', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(out/'comparison.mp4')]
    subprocess.run(args, check=True)
    with av.open(str(out/'comparison.mp4')) as video: frames = sum(1 for _ in video.decode(video=0))
    assert frames > 0
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e109_rollout_README.md', out/'README.md')
    write_json(out/'progress.json', dict(status='loading fixed checkpoint'))
    shutil.copy2(args.checkpoint, out/'checkpoint.pt')
    saved = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
    policy = ContextPolicy(); load_state(policy, saved['model'])
    assert dict(tracker=fingerprint(policy.tracker.backbone), wrist=fingerprint(policy.wrist)) == saved['config']['frozen_sha256']
    policy.eval().cuda()
    rows = selected_cases()
    config = dict(status='running', checkpoint=str(args.checkpoint), checkpoint_snapshot='checkpoint.pt',
        checkpoint_sha256=hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest(), checkpoint_epoch=saved['epoch'],
        checkpoint_step=saved['step'], H=25, K=10, flow_steps=10, noise_seed=0, frame_budget=900,
        cases=rows, training=saved['config'], initial_state='recorded post-shuffle expert initial physical state',
        observations='causal top history + frame0 point; live left320 RGB; measured6Djoint state',
        expert_future_loaded=False, GT_or_mask_policy_input=False, actual_wrist_camera=False,
        video_capture_stride=3, video_fps=25,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path('OATFlow/shuffle_bc/context_model.py'), Path('OATFlow/shuffle_bc/evaluate.py'),
             Path('OATFlow/shuffle_bc/data.py'), Path('OATFlow/simulation/pose.py'), Path('OATFlow/simulation/control.py')]})
    write_json(out/'config.json', config)
    for name in config['source_sha256']:
        p = Path(name).resolve(); dest = out/'source'/p.relative_to(Path.cwd())
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    (out/'source'/'README.md').write_text('Exact rollout source; configuration/hashes in ../config.json.\n')
    started = time.perf_counter(); results = []
    for start in tqdm(range(0, len(rows), 4), desc='K10 smoke-test groups', mininterval=10):
        current = rollout(rows[start:start+4], policy, out); results.extend(current)
        with (out/'results.jsonl').open('a') as stream:
            for result in current: stream.write(json.dumps(result)+'\n')
        progress = dict(status='running', cases=len(results), successes=sum(r['success'] for r in results),
            elapsed_s=time.perf_counter()-started)
        write_json(out/'progress.json', progress); print(json.dumps(progress), flush=True)
    frames = montage(rows, out)
    summary = dict(status='complete', cases=len(results), successes=sum(r['success'] for r in results),
        reasons=dict(Counter(r['reason'] for r in results)), elapsed_s=time.perf_counter()-started,
        by_group={name: dict(cases=sum(r['group'] == name for r in results), successes=sum(r['success'] for r in results if r['group'] == name))
            for name in ('train', 'composition', 'spatial', 'combined')}, all_case_videos_verified=True,
        comparison_frames=frames, checkpoint_step=saved['step'], peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(out/'summary.json', summary); write_json(out/'progress.json', summary)
    config['status'] = 'complete'; write_json(out/'config.json', config)
    with (out/'README.md').open('a') as stream:
        stream.write('\nStatus: complete. Actual results: '+json.dumps(summary)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
