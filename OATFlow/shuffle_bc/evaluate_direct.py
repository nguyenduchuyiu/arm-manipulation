"""Closed-loop mask-TCOW / direct-FM rollouts: demos or the full held-out test."""
import argparse
from collections import Counter
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
from OATFlow.shuffle_bc.context_video import font, marker, GREEN, PINK
from OATFlow.shuffle_bc.data import DATA, splits
from OATFlow.shuffle_bc.direct_model import DirectPolicy, pack_top, unpack_top
from OATFlow.shuffle_bc.evaluate import cases, prepare
from OATFlow.shuffle_bc.train_direct import frozen_models
from OATFlow.simulation.contacts import jaw_contacts
from OATFlow.simulation.control import step as servo_step
from OATFlow.simulation.io import encoder, write_json
from OATFlow.simulation.pose import rotation_matrix
from OATFlow.simulation.render import visual_model
from OATFlow.tracking.data import sample_indices
from OATFlow.tracking.linear_probe import fingerprint
from OATFlow.tracking.mask_data import read_segment
from OATFlow.tracking.mask_video import digest
from OATFlow.tracking.point_data import project, top_camera


def paint(rgb, mask):
    pixels = rgb.astype(np.float32).copy()
    pixels[mask] = .55*pixels[mask] + .45*np.array([230, 70, 160])
    return Image.fromarray(pixels.astype(np.uint8))


def export_video(scene, result, out, camera):
    path = out/(result['path'].replace('/', '_')+'.mp4')
    writer = encoder(path, 1280, 800)
    split = dict(train='TRAIN', composition='UNSEEN ORDER', spatial='UNSEEN LAYOUT', combined='BOTH UNSEEN')
    for i, sample in enumerate(tqdm(scene['video'], desc='Encode live demo', mininterval=5, leave=False)):
        frame, top, side, mask, gt, age = sample
        image = Image.new('RGB', (1280, 800), '#f5f7fa'); draw = ImageDraw.Draw(image)
        draw.text((30, 18), scene['video_title'], font=font(26, True), fill='#182638')
        draw.text((30, 61), f'{split[result["group"]]} | {result["path"]} | target: {result["target"]}', font=font(19), fill='#43536a')
        image.paste(paint(top, mask).resize((800, 600)), (30, 130))
        image.paste(Image.fromarray(side).resize((360, 360)), (850, 130))
        draw.text((30, 98), f'LIVE TOP | TCOW mask: last replan, age {age} frames', font=font(17, True), fill='#182638')
        draw.text((850, 98), 'LIVE LEFT (surrogate wrist)', font=font(17, True), fill='#182638')
        uv = project(gt, camera)*2.5+[30, 130]
        if 30 <= uv[0] <= 830 and 130 <= uv[1] <= 730:
            marker(draw, uv, GREEN, False)
        initial = scene['images'][0].permute(1, 2, 0).numpy()
        image.paste(paint(initial, scene['prompt']).resize((160, 120)), (850, 535))
        draw.text((850, 505), 'FRAME 0 MASK PROMPT', font=font(16, True), fill='#182638')
        draw.text((1030, 539), f'{frame/25:.2f} s', font=font(24, True), fill='#182638')
        draw.text((1030, 579), 'H25 / K10', font=font(18), fill='#43536a')
        draw.text((850, 680), 'Pink: TCOW target mask', font=font(18, True), fill=PINK)
        draw.text((850, 712), 'Green: GT center, display only', font=font(17), fill=GREEN)
        status = 'RUNNING' if frame+3 < result['frames'] else result['reason'].upper().replace('_', ' ')
        draw.text((30, 750), status, font=font(20, True), fill=GREEN if result['success'] else '#43536a')
        draw.text((370, 750), 'FM controls robot | no XYZ head or context decoder', font=font(18), fill='#43536a')
        for _ in range(min(3, result['frames']-frame)):
            writer.stdin.write(np.asarray(image).tobytes())
        if i in (0, len(scene['video'])-1):
            image.save(out/f'{path.stem}_{"start" if i == 0 else "end"}.png')
    writer.stdin.close(); assert writer.wait() == 0
    with av.open(str(path)) as video:
        decoded = sum(1 for _ in video.decode(video=0))
    assert decoded == result['frames']
    return decoded


@torch.inference_mode()
def rollout(row, policy, tracker, wrist, out, title, video=True, progress=None, frame_budget=900):
    physics = mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
    visual = visual_model(DATA/'scene.xml', object_bodies=NAMES)
    visual.vis.global_.offwidth = visual.vis.global_.offheight = 320
    display = mujoco.MjData(visual); option = mujoco.MjvOption(); option.geomgroup[3] = 0
    scene = prepare([row], physics)[0]
    scene['video_title'] = title
    context = read_segment(DATA, row['context_path'], True)
    expected = np.unique(np.r_[np.arange(0, row['context_frames'], 3), row['context_frames']-1])
    np.testing.assert_array_equal(context['raw_index'], expected)
    scene['images'] = list(context['rgb'].unbind())
    scene['image_times'] = context['raw_index'].tolist()
    scene['prompt'] = context['prompt'][row['target_id']].copy()
    scene['query'].zero_(); scene['query'][0, 0, 0] = torch.from_numpy(scene['prompt'])
    camera = top_camera(DATA)
    del context
    generator = torch.Generator(device='cuda').manual_seed(0)
    masks, histories, replan_frames = [], [], []
    with mujoco.Renderer(visual, height=320, width=320) as renderer:
        for frame in tqdm(range(frame_budget), desc=f'Live {row["evaluation_group"]}', mininterval=5, leave=False):
            if scene['done']: break
            if frame%100 == 0:
                write_json(out/'progress.json', dict(**(progress or {}), status='live rollout', case=row['path'], frame=frame, frame_budget=frame_budget))
            if frame%3 == 0 or frame%10 == 0:
                display.qpos[:] = scene['data'].qpos; mujoco.mj_forward(visual, display)
                renderer.update_scene(display, camera='cam_top', scene_option=option)
                top = torch.from_numpy(renderer.render()[40:280].copy()).permute(2, 0, 1)
                renderer.update_scene(display, camera='cam_left', scene_option=option)
                side = torch.from_numpy(renderer.render().copy()).permute(2, 0, 1)
                if frame%3 == 0:
                    scene['images'].append(top)
                    scene['image_times'].append(row['context_frames']+frame)
            if frame%10 == 0:
                images = scene['images'] if frame%3 == 0 else scene['images']+[top]
                times = scene['image_times'] if frame%3 == 0 else scene['image_times']+[row['context_frames']+frame]
                chosen = sample_indices(len(images)-1, 30)
                assert chosen[0] == 0 and chosen[-1] == len(images)-1
                chosen_times = np.asarray(times)[chosen]
                assert chosen_times.max() == row['context_frames']+frame
                rgb = torch.stack([images[i] for i in chosen]).permute(1, 0, 2, 3)[None].cuda()
                noise = policy.flow.sample_noise(1, 'cuda', generator)
                joints = torch.tensor(scene['data'].qpos[scene['scene'].qadr][None], dtype=torch.float32, device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    raw = tracker.features(rgb, scene['query'].cuda())[:, -1].contiguous()
                    codes, scales = pack_top(raw)
                    hidden = unpack_top(codes, scales)
                    aux = wrist(side[None].cuda()).half()
                    actions = policy.sample_features(hidden, aux, joints, noise, steps=10)
                    logits = tracker.masks_from_features(hidden[:, None])
                assert torch.isfinite(actions).all() and torch.isfinite(logits).all()
                scene['mask'] = (logits[0, 0] > 0).cpu().numpy()
                masks.append(np.packbits(scene['mask'], axis=-1, bitorder='little'))
                histories.append(chosen_times); replan_frames.append(frame)
                scene['plan'] = actions[0].float().cpu().numpy()
                scene['anchor'] = scene['data'].site_xpos[scene['scene'].site].copy()
                scene['anchor_R'] = scene['data'].site_xmat[scene['scene'].site].reshape(3, 3).copy()
                scene['plans'].append(scene['plan'].copy()); scene['anchors'].append(scene['anchor'].copy())
                scene['anchor_rotations'].append(scene['anchor_R'].copy())
            state = scene['data']; action = scene['plan'][frame%10]
            if video and frame%3 == 0:
                gt = state.xpos[physics.body(NAMES[row['target_id']]).id]-camera['base']
                scene['video'].append((frame, top.permute(1, 2, 0).numpy(), side.permute(1, 2, 0).numpy(), scene['mask'].copy(), gt.copy(), frame%10))
            command = scene['ik'].command(scene, scene['anchor']+action[:3], action[9], rotation_matrix(action[3:9])@scene['anchor_R'])
            command = np.clip(command, *physics.actuator_ctrlrange.T)
            servo_step(physics, state, scene['held'], command); scene['held'] = command.copy()
            objects = np.stack([state.xpos[physics.body(n).id] for n in NAMES])
            contacts = np.array([len(jaw_contacts(physics, state, n)) for n in NAMES])
            lifted = (objects[:, 2]-scene['initial_objects'][:, 2] > .04) & (contacts == 2)
            target = row['target_id']; scene['streak'] = scene['streak']+1 if lifted[target] else 0
            scene['longest'] = max(scene['longest'], scene['streak'])
            scene['cover_deposited'] = cover_deposited(physics, state, scene['selected_cover'])
            scene['other_cover_shift_m'] = float(np.linalg.norm(state.xpos[physics.body(scene['other_cover']).id, :2]-scene['other_cover_start'][:2]))
            if lifted[:4][np.arange(4) != target].any():
                scene.update(done=True, reason='wrong_object_lift')
            elif scene['streak'] >= 10 and scene['cover_deposited'] and scene['other_cover_shift_m'] <= .015:
                scene.update(done=True, success=True, reason='success')
            for key in ('qpos', 'qvel', 'ctrl'): scene[key].append(getattr(state, key).copy())
            scene['commands'].append(command.copy()); scene['objects'].append(objects); scene['contacts'].append(contacts)
    objects = np.array(scene['objects'])
    result = dict(path=row['path'], group=row['evaluation_group'], target=NAMES[row['target_id']], success=scene['success'],
        reason=scene['reason'], frames=len(scene['commands']), cover_deposited=scene['cover_deposited'],
        other_cover_shift_m=scene['other_cover_shift_m'], lifted_bilateral_streak=scene['longest'],
        max_target_lift_m=float((objects[:, row['target_id'], 2]-scene['initial_objects'][row['target_id'], 2]).max()),
        max_cover_lift_m=(objects[:, 4:, 2]-scene['initial_objects'][4:, 2]).max(0).tolist())
    np.savez_compressed(out/(row['path'].replace('/', '_')+'.npz'), **{k:np.asarray(scene[k]) for k in
        ('initial_qpos','initial_qvel','initial_ctrl','commands','qpos','qvel','ctrl','objects','contacts','plans','anchors','anchor_rotations')},
        initial_mask=scene['prompt'], mask_packed=np.asarray(masks), history_times=np.asarray(histories), replan_frames=np.asarray(replan_frames))
    result['video_frames_verified'] = export_video(scene, result, out, camera) if video else 0
    return result


def montage(rows, out, name='comparison.mp4'):
    pending = out/(Path(name).stem+'.tmp.mp4')
    if pending.exists(): pending.unlink()
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    for row in rows: command.extend(['-i', str(out/(row['path'].replace('/', '_')+'.mp4'))])
    command.extend(['-filter_complex', '[0:v]scale=640:400[a];[1:v]scale=640:400[b];[2:v]scale=640:400[c];[3:v]scale=640:400[d];[a][b][c][d]xstack=inputs=4:layout=0_0|640_0|0_400|640_400:fill=white[v]',
        '-map','[v]','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(pending)])
    subprocess.run(command, check=True)
    with av.open(str(pending)) as video: frames = sum(1 for _ in video.decode(video=0))
    pending.replace(out/name)
    return frames


def counts(results):
    def metric(rows):
        n = len(rows); success = sum(r['success'] for r in rows)
        return dict(cases=n, successes=success, success_rate=success/n if n else None,
                    cover_deposits=sum(r['cover_deposited'] for r in rows))
    return dict(**metric(results), reasons=dict(Counter(r['reason'] for r in results)),
        by_group={g:metric([r for r in results if r['group']==g]) for g in ('composition','spatial','combined')},
        by_target={target:metric([r for r in results if r['target']==target]) for target in NAMES[:4]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--readme', type=Path, default=Path('docs/research/e115_rollout_README.md'))
    parser.add_argument('--full-test', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Continue a full test from its atomic results.json')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    out = args.output
    previous_config = json.loads((out/'config.json').read_text()) if args.resume else None
    if args.resume:
        assert args.full_test and previous_config['full_test'] and previous_config['status'] != 'complete'
        assert digest(args.checkpoint) == previous_config['checkpoint_sha256'] == digest(out/'checkpoint.pt')
    else:
        out.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(args.readme, out/'README.md')
        write_json(out/'progress.json', dict(status='loading checkpoint'))
        shutil.copyfile(args.checkpoint, out/'checkpoint.pt')
    saved = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
    assert saved['epoch'] > 0 and saved['step'] > 0
    policy = DirectPolicy(); policy.load_state_dict(saved['model'], strict=True); policy.eval().requires_grad_(False).cuda()
    tracker, wrist = frozen_models()
    frozen = dict(TCOW=fingerprint(tracker), auxiliary_ViT=fingerprint(wrist))
    assert frozen == json.loads((args.checkpoint.parent/'frozen_hashes.json').read_text())
    assert digest(Path(saved['config']['TCOW_checkpoint'])) == saved['config']['TCOW_sha256']
    if args.full_test:
        groups = splits()
        rows = [r for g in ('composition','spatial','combined') for r in groups[g]]
        assert [len(groups[g]) for g in ('composition','spatial','combined')] == [1024,256,128]
        assert len(rows) == len({r['path'] for r in rows}) == 1408
        assert Counter(r['target_id'] for r in rows) == {i:352 for i in range(4)}
        previews = [r for g in ('composition','spatial','combined') for r in groups[g]
                    if r['context_path']==groups[g][0]['context_path']]
        assert len(previews) == 12
        assert all({r['target_id'] for r in previews if r['evaluation_group']==g} == {0,1,2,3}
                   for g in ('composition','spatial','combined'))
        preview_paths = {r['path'] for r in previews}
        rows = previews+[r for r in rows if r['path'] not in preview_paths]
    else:
        all_cases = cases()
        rows = [next(r for r in all_cases if r['evaluation_group']==group and r['target_id']==0)
                for group in ('train','composition','spatial','combined')]
        preview_paths = {r['path'] for r in rows}
    sources = [Path(__file__),*[Path('OATFlow/shuffle_bc')/name for name in
        ('direct_model.py','direct_data.py','evaluate.py','train_direct.py','context_model.py')],
        *[Path('OATFlow/tracking')/name for name in ('mask.py','backbone.py','mask_data.py','data.py')],
        *[Path('OATFlow/simulation')/name for name in ('pose.py','control.py')]]
    config = dict(status='running', checkpoint=str(args.checkpoint), checkpoint_snapshot='checkpoint.pt',
        checkpoint_sha256=digest(out/'checkpoint.pt'), checkpoint_epoch=saved['epoch'], checkpoint_step=saved['step'],
        H=25,K=10,flow_steps=10,noise_seed=0,frame_budget=900,policy_hz=25,cases=rows,training=saved['config'],
        full_test=args.full_test,video_cases=sorted(preview_paths),
        scene_sha256=digest(DATA/'scene.xml'),manifest_sha256=digest(DATA/'manifest.jsonl'),
        initial_state='recorded post-shuffle expert initial physical state; future expert observations/actions never read',
        history='context stride3+terminal, live stride3, current replan frame appended temporarily if off cadence; native30uniform prefix sampling',
        prompt='visible target mask at context frame0 only', later_GT_policy_input=False, actual_wrist_camera=False,
        numeric_boundaries='BF16 native encoders, block16 int8 TCOW roundtrip, FP16 auxiliary tokens, as training cache',
        video='25FPS, captures every3frames with causal hold; predicted mask updates every10frames; GT center display only',
        success='correct cover deposited; target lift>4cm with bilateral contacts10frames; other cover shift<=15mm; wrong-object lift fails',
        frozen_sha256=frozen,source_sha256={str(p):digest(p) for p in sources})
    if args.resume:
        for key in ('checkpoint_sha256','cases','H','K','flow_steps','noise_seed','frame_budget','history',
                    'numeric_boundaries','success','source_sha256','video_cases','frozen_sha256','scene_sha256','manifest_sha256'):
            assert previous_config[key] == config[key], key
    write_json(out/'config.json', config)
    if not args.resume: write_json(out/'results.json',[])
    for source in sources:
        dest = out/'source'/source.resolve().relative_to(Path.cwd())
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
    title = f'E115 | Frozen mask TCOW + direct FM | epoch {saved["epoch"]}'
    del saved
    print('CONFIG', json.dumps({k:v for k,v in config.items() if k not in ('cases','training','source_sha256')}), flush=True)
    results = json.loads((out/'results.json').read_text()) if args.resume else []
    done = {r['path'] for r in results}
    assert len(done) == len(results) and done <= {r['path'] for r in rows}
    elapsed_before = sum(r['elapsed_case_seconds'] for r in results) if args.resume else 0.
    if args.resume:
        (out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    started = time.perf_counter()
    for row in tqdm(rows, desc='Full held-out K10 test' if args.full_test else 'Four fixed K10 rollouts', mininterval=5):
        if row['path'] in done: continue
        if args.resume:
            stem = row['path'].replace('/','_')
            for suffix in ('.npz','.mp4','_start.png','_end.png'):
                path = out/(stem+suffix)
                if path.exists(): path.unlink()
        elapsed = elapsed_before+time.perf_counter()-started
        progress = dict(cases_done=len(results),cases_total=len(rows),successes=sum(r['success'] for r in results),
                        elapsed_seconds=elapsed,eta_seconds=elapsed/max(len(results),1)*(len(rows)-len(results)))
        tick = time.perf_counter()
        result = rollout(row, policy, tracker, wrist, out, title, video=row['path'] in preview_paths,progress=progress)
        result['elapsed_case_seconds'] = time.perf_counter()-tick
        results.append(result)
        write_json(out/'results.tmp',results);(out/'results.tmp').replace(out/'results.json')
        with (out/'results.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
        print('RESULT', json.dumps(result), flush=True)
        elapsed = elapsed_before+time.perf_counter()-started
        write_json(out/'progress.json',dict(status='running',**counts(results),cases_done=len(results),cases_total=len(rows),
            elapsed_seconds=elapsed,eta_seconds=elapsed/len(results)*(len(rows)-len(results))))
        if args.full_test:
            for group in ('composition','spatial','combined'):
                selected = [r for r in rows if r['evaluation_group']==group and r['path'] in preview_paths]
                if all(r['path'] in {v['path'] for v in results} for r in selected) and not (out/f'{group}.mp4').exists():
                    montage(selected,out,f'{group}.mp4')
    assert frozen == dict(TCOW=fingerprint(tracker), auxiliary_ViT=fingerprint(wrist))
    assert digest(args.checkpoint) == config['checkpoint_sha256']
    assert len(results) == len(rows) and {r['path'] for r in results} == {r['path'] for r in rows}
    if args.full_test:
        frames = None
    else:
        frames = montage(rows, out)
        assert frames == max(r['frames'] for r in results)
    summary = dict(status='complete', **counts(results), results=results, elapsed_seconds=elapsed_before+time.perf_counter()-started,
        all_videos_fully_decoded=True,comparison_frames=frames,frozen_vision_unchanged=True,
        checkpoint_unchanged=True,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(out/'summary.json',summary); write_json(out/'progress.json',summary)
    config['status']='complete'; write_json(out/'config.json',config)
    brief = {k:v for k,v in summary.items() if k!='results'}
    with (out/'README.md').open('a') as stream: stream.write('\nActual results: '+json.dumps(brief)+'\n')
    print('COMPLETE',json.dumps(brief),flush=True)


if __name__ == '__main__': main()
