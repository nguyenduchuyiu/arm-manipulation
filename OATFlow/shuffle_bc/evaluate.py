"""Post-shuffle closed-loop cover opening and pick, with a causal top-camera policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import Scene, NAMES, SEEN, UNSEEN
from OATFlow.environment.success import cover_deposited
from OATFlow.shuffle_bc.data import DATA, ROOT, splits, read_context, select_indices
from OATFlow.shuffle_bc.model import ShufflePolicy
from OATFlow.simulation.contacts import jaw_contacts
from OATFlow.simulation.control import step as servo_step
from OATFlow.simulation.io import encoder
from OATFlow.simulation.pose import PoseIK, rotation_matrix
from OATFlow.simulation.render import visual_model


def cases():
    groups = splits()
    result = []
    for group, layouts, perms in [('train', (0, 31), SEEN[::4]),
            ('composition', (0, 31), UNSEEN[::2]), ('spatial', (36, 39), SEEN[::4]),
            ('combined', (36, 39), UNSEEN[::2])]:
        result.extend(r for r in groups[group] if r['layout_id'] in layouts and r['permutation'] in perms)
    assert len(result) == 128
    return result


def prepare(rows, physics):
    # Read only recorded context, never the expert's future images or object states.
    from OATFlow.tracking.point_data import top_camera, project
    camera = top_camera(DATA)
    contexts = {}
    scenes = []
    for row in rows:
        ref = row['context_path']
        if ref not in contexts:
            contexts[ref] = read_context(ref)
        context = contexts[ref]
        scene = Scene(physics)
        with np.load(DATA / row['path'] / 'sim_state.npz') as z:
            for key in ('qpos', 'qvel', 'ctrl'):
                getattr(scene.data, key)[:] = z['initial_' + key]
        mujoco.mj_forward(physics, scene.data)
        # Query annotation uses only the first revealed frame, never the post-shuffle state.
        uv = project(context['xyz'][0, row['target_id']], camera)
        x, y = np.rint(uv).astype(int)
        assert 3 <= x < 317 and 3 <= y < 237
        query = torch.zeros(1, 1, 30, 240, 320)
        yy, xx = torch.meshgrid(torch.arange(y-3, y+4), torch.arange(x-3, x+4), indexing='ij')
        query[0, 0, 0, yy, xx] = ((xx-x).square() + (yy-y).square() <= 9).float()
        state = scene.data
        selected = row['selected_cover']
        other = 'cover_b' if selected == 'cover_a' else 'cover_a'
        scenes.append(dict(row=row, scene=scene, data=state, model=physics,
            selected_cover=selected, other_cover=other,
            other_cover_start=state.xpos[physics.body(other).id].copy(),
            held=state.ctrl.copy(), ik=PoseIK((physics, scene.qadr, scene.dofs[:5], scene.site)),
            images=[image.clone() for image in context['rgb'][0]],
            image_times=context['raw_index'].tolist(), query=query, uv=uv,
            initial_qpos=state.qpos.copy(), initial_qvel=state.qvel.copy(), initial_ctrl=state.ctrl.copy(),
            initial_objects=np.stack([state.xpos[physics.body(n).id] for n in NAMES]),
            qpos=[state.qpos.copy()], qvel=[state.qvel.copy()], ctrl=[state.ctrl.copy()],
            commands=[], objects=[], contacts=[], plans=[], anchors=[], anchor_rotations=[],
            video=[], success=False, done=False, reason='timeout', streak=0, longest=0,
            cover_deposited=False, other_cover_shift_m=0.))
    return scenes


@torch.no_grad()
def rollout(rows, policy, out, frame_budget=900, oracle=False):
    physics = mujoco.MjModel.from_xml_path(str(DATA / 'scene.xml'))
    visual = visual_model(DATA / 'scene.xml', object_bodies=NAMES)
    visual.vis.global_.offwidth = visual.vis.global_.offheight = 320
    display = mujoco.MjData(visual)
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    scenes = prepare(rows, physics)
    if oracle:
        for s in scenes:
            path = DATA / s['row']['path']
            with np.load(path / 'observation.npz') as z:
                s['oracle_ee'] = z['ee_position_base_m'] + [.539369022999, .0639727628473, 0.]
                s['oracle_R'] = rotation_matrix(z['ee_rotation_6d'])
            with np.load(path / 'supervision.npz') as z:
                s['oracle_jaw'] = z['joint_command'][:, 5]
    generator = torch.Generator(device='cuda').manual_seed(0)
    with mujoco.Renderer(visual, height=320, width=320) as renderer:
        for frame in tqdm(range(frame_budget), desc='Full cover + pick rollout', mininterval=10, leave=False):
            active = [s for s in scenes if not s['done']]
            if not active:
                break
            if frame % 3 == 0 or frame % 10 == 0:
                for s in active:
                    display.qpos[:] = s['data'].qpos; mujoco.mj_forward(visual, display)
                    renderer.update_scene(display, camera='cam_top', scene_option=option)
                    s['current_rgb'] = torch.from_numpy(renderer.render()[40:280].copy()).permute(2, 0, 1)
                    if frame % 3 == 0:
                        s['images'].append(s['current_rgb'])
                        s['image_times'].append(s['row']['context_frames'] + frame)
                        if s['row']['target_id'] == 0:
                            s['video'].append(s['current_rgb'].permute(1, 2, 0).numpy())
            if frame % 10 == 0 and not oracle:
                rgb = []; joints = []; queries = []
                for s in active:
                    images = s['images'] if frame % 3 == 0 else s['images'] + [s['current_rgb']]
                    times = s['image_times'] if frame % 3 == 0 else s['image_times'] + [s['row']['context_frames'] + frame]
                    chosen = select_indices(np.array(times), len(images)-1)
                    rgb.append(torch.stack([images[i] for i in chosen]).permute(1, 0, 2, 3)[None])
                    queries.append(s['query'])
                    joints.append(s['data'].qpos[s['scene'].qadr])
                noise = policy.flow.sample_noise(1, 'cuda', generator).expand(len(active), -1, -1)
                # Microbatch1 bounds tracker activation memory during long rollouts.
                for i, s in enumerate(active):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        plan = policy.sample(rgb[i].cuda(), queries[i].cuda(),
                            torch.tensor(np.array(joints[i:i+1]), dtype=torch.float32, device='cuda'), noise[i:i+1], 10)
                    s['plan'] = plan[0].float().cpu().numpy()
                    s['anchor'] = s['data'].site_xpos[s['scene'].site].copy()
                    s['anchor_R'] = s['data'].site_xmat[s['scene'].site].reshape(3, 3).copy()
                    s['plans'].append(s['plan'].copy()); s['anchors'].append(s['anchor'].copy())
                    s['anchor_rotations'].append(s['anchor_R'].copy())
            for s in active:
                state = s['data']; site = s['scene'].site
                if oracle:
                    i = min(frame + 1, len(s['oracle_ee']) - 1)
                    position, rotation = s['oracle_ee'][i], s['oracle_R'][i]
                    jaw = s['oracle_jaw'][min(frame, len(s['oracle_jaw'])-1)]
                else:
                    action = s['plan'][frame % 10]
                    position = s['anchor'] + action[:3]
                    rotation = rotation_matrix(action[3:9]) @ s['anchor_R']; jaw = action[9]
                command = s['ik'].command(s, position, jaw, rotation)
                command = np.clip(command, *physics.actuator_ctrlrange.T)
                servo_step(physics, state, s['held'], command); s['held'] = command.copy()
                objects = np.stack([state.xpos[physics.body(n).id] for n in NAMES])
                contacts = np.array([len(jaw_contacts(physics, state, n)) for n in NAMES])
                lifted = (objects[:, 2] - s['initial_objects'][:, 2] > .04) & (contacts == 2)
                target = s['row']['target_id']
                s['streak'] = s['streak'] + 1 if lifted[target] else 0
                s['longest'] = max(s['longest'], s['streak'])
                s['cover_deposited'] = cover_deposited(physics, state, s['selected_cover'])
                s['other_cover_shift_m'] = float(np.linalg.norm(
                    state.xpos[physics.body(s['other_cover']).id, :2] - s['other_cover_start'][:2]))
                if lifted[:4][np.arange(4) != target].any():
                    s.update(done=True, reason='wrong_object_lift')
                elif s['streak'] >= 10 and s['cover_deposited'] and s['other_cover_shift_m'] <= .015:
                    s.update(done=True, success=True, reason='success')
                for key in ('qpos', 'qvel', 'ctrl'):
                    s[key].append(getattr(state, key).copy())
                s['commands'].append(command.copy()); s['objects'].append(objects); s['contacts'].append(contacts)
    results = []
    for s in scenes:
        row = s['row']; name = row['path'].replace('/', '_')
        objects = np.array(s['objects'])
        record = dict(path=row['path'], group=row['evaluation_group'], target=NAMES[row['target_id']],
            success=s['success'], reason=s['reason'], frames=len(s['commands']),
            cover_deposited=s['cover_deposited'], other_cover_shift_m=s['other_cover_shift_m'],
            lifted_bilateral_streak=s['longest'], max_target_lift_m=float((objects[:, row['target_id'], 2] - s['initial_objects'][row['target_id'], 2]).max()),
            max_cover_lift_m=(objects[:, 4:, 2] - s['initial_objects'][4:, 2]).max(0).tolist(), query_uv=s['uv'].tolist())
        np.savez_compressed(out / (name + '.npz'), **{k: np.array(s[k]) for k in
            ('initial_qpos', 'initial_qvel', 'initial_ctrl', 'commands', 'qpos', 'qvel', 'ctrl',
             'objects', 'contacts', 'plans', 'anchors', 'anchor_rotations')})
        if row['target_id'] == 0:
            video = encoder(out / (name + '.mp4'), 640, 480)
            for pixels in s['video']:
                image = Image.fromarray(pixels).resize((640, 480))
                draw = ImageDraw.Draw(image)
                draw.text((8, 8), f'{row["evaluation_group"]} {row["path"]} | {record["reason"]}', fill='white', stroke_fill='black', stroke_width=1)
                for _ in range(3):
                    video.stdin.write(np.asarray(image).tobytes())
            video.stdin.close(); assert video.wait() == 0
        results.append(record)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'train_2/final.pt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu-memory-gib', type=float,
                        help='Cap this evaluator CUDA allocator when sharing a GPU with training')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    if args.gpu_memory_gib is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_gib * 1024**3 / total)
    out = args.output or ROOT / ('oracle_pose' if args.oracle else 'test_k10'); out.mkdir(exist_ok=False)
    checkpoint = args.checkpoint
    policy = None; saved = None
    if not args.oracle:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        policy = ShufflePolicy(pretrained=False).cuda().eval()
        policy.load_state_dict(saved['model'], strict=True)
    rows = cases() if not args.oracle else cases()[:4]
    config = dict(status='running', data=str(DATA), checkpoint=str(checkpoint) if saved else 'measured EE oracle',
        checkpoint_step=saved['step'] if saved else None, cases=len(rows), H=25 if saved else 1, K=10 if saved else 1, flow_steps=10 if saved else None,
        noise_seed=0, frame_budget=900, policy_hz=25, servo_hz=50, physics_hz=500,
        initial_state='recorded initial expert state, immediately after shuffle',
        observations='only past/current top RGB, initial radius3 click, measured joints/jaw',
        learned='entire cover opening/deposit and target grasp/lift', oracle=args.oracle,
        success='correct cover deposited in drop zone; target lift>4cm with bilateral jaw contacts for10frames; other cover displacement<=15mm;wrong cube lift fails',
        training=saved['config'] if saved else 'none; controller validation',
        gpu_memory_gib=args.gpu_memory_gib)
    write = lambda p, value: p.write_text(json.dumps(value, indent=2) + '\n')
    write(out / 'config.json', config)
    readme = '# e105_tcow ' + out.name + '\n\nStatus: running. Exact train/test config: config.json.\n'
    (out / 'README.md').write_text(readme)
    results = []; started = time.perf_counter()
    for offset in tqdm(range(0, len(rows), 4), desc='Full shuffle BC cases', mininterval=5):
        current = rollout(rows[offset:offset+4], policy, out, oracle=args.oracle); results.extend(current)
        with (out / 'results.jsonl').open('a') as stream:
            for row in current: stream.write(json.dumps(row) + '\n')
        print(json.dumps(dict(cases=len(results), successes=sum(r['success'] for r in results))), flush=True)
    summary = dict(cases=len(results), successes=sum(r['success'] for r in results),
        reasons=dict(Counter(r['reason'] for r in results)), elapsed_s=time.perf_counter()-started,
        by_group={group: dict(cases=sum(r['group']==group for r in results),
                             successes=sum(r['group']==group and r['success'] for r in results))
                  for group in sorted({r['group'] for r in results})})
    write(out / 'summary.json', summary); config['status'] = 'complete'; write(out / 'config.json', config)
    (out / 'README.md').write_text(readme.replace('Status: running.', 'Status: complete.') + json.dumps(summary) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
