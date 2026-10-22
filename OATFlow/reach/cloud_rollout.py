"""Snapshot the current best E118 policy, test fixed held-out goals and render videos."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

from OATFlow.reach.cloud_data import ROOT, DATA, sha, write
from OATFlow.reach.cloud_eval import rollouts
from OATFlow.reach.pointcloud import PointCloudReach
from OATFlow.simulation.render import visual_model, add_marker, path_overlay


def font(size, bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf', size)


def wire_cube(scene, center):
    for axis in range(3):
        other = [j for j in range(3) if j != axis]
        for a in (-1, 1):
            for b in (-1, 1):
                line = np.tile(center, (2, 1)); line[:, axis] += [-.01, .01]
                line[:, other] += [a*.01, b*.01]
                path_overlay(scene, line, (248, 156, 45), .00065)


def input_plot(cloud, target_base, path):
    fig = plt.figure(figsize=(3.0, 2.9), dpi=100, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*cloud.T, c='#148fbb', s=11, alpha=.9)
    ax.scatter(*target_base, c='#ed8a28', marker='+', s=55)
    for axis in range(3):
        getattr(ax, 'set_'+('x', 'y', 'z')[axis]+'lim')(target_base[axis]-.014, target_base[axis]+.014)
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(2)); axis.label.set_fontsize(7)
    ax.tick_params(labelsize=6, pad=0); ax.set_box_aspect([1,1,1]); ax.view_init(22, -70)
    fig.subplots_adjust(left=0, right=.89, top=1, bottom=.02)
    fig.savefig(path); plt.close(fig)
    return Image.open(path).convert('RGB')


def encode(path, size, fps):
    return subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast',
        '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)


def validate_video(path, count):
    with av.open(str(path)) as video:
        decoded = sum(1 for _ in video.decode(video=0))
    assert decoded == count, (str(path), decoded, count)
    return dict(path=path.name, frames=decoded, bytes=path.stat().st_size)


def render(out, arrays, result, clouds, checkpoint_step):
    model = visual_model(DATA.parent/'scene.xml',
        lighting_scene='/workspace/robotics_assets/robot/nexarm/visual_reference.xml')
    model.vis.headlight.ambient[:] = .4
    model.vis.headlight.diffuse[:] = .65
    model.vis.global_.offwidth = 960; model.vis.global_.offheight = 600
    state = mujoco.MjData(model)
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    base = np.array(json.loads((ROOT/'clouds/config.json').read_text())['base_origin_world_m'])
    view = mujoco.MjvCamera(); view.lookat[:] = base+[0, 0, .23]
    view.distance = 1.12; view.azimuth = 135; view.elevation = -25
    indices = np.arange(0, 301, 2)  # Preserve 12 seconds at12.5FPS, including final state.
    posters = []; checks = []
    with mujoco.Renderer(model, height=600, width=960, max_geom=2200) as renderer:
        for case in tqdm(range(6), desc='Render six fixed held-out goals', mininterval=5):
            record = result['records'][case]; goal = arrays['target'][case]
            target_base = goal-base; trajectory = int(arrays['trajectory_ids'][case])
            view.lookat[:] = base + [.3*target_base[0], .3*target_base[1], .25]
            cloud = clouds[trajectory]
            inset = input_plot(cloud, target_base, out/f'case_{case+1:02d}_input.png')
            path = out/f'case_{case+1:02d}.mp4'; writer = encode(path, (1280, 800), 12.5)
            final_status = 'PASS @5mm' if record['success_5mm'] else ('PASS @10mm' if record['success_10mm'] else 'OUTSIDE SUCCESS CRITERION')
            try:
                for frame in tqdm(indices, desc=f'Goal {case+1}: RGB replay', leave=False, mininterval=5):
                    state.qpos[:] = arrays['qpos'][case, frame]; mujoco.mj_forward(model, state)
                    np.testing.assert_allclose(state.site_xpos[model.site('grasp_site').id],
                        arrays['ee'][case, frame], rtol=0, atol=2e-6)
                    renderer.update_scene(state, camera=view, scene_option=option)
                    wire_cube(renderer.scene, goal)
                    for point in np.unique(cloud, axis=0):
                        add_marker(renderer.scene, point+base, [.05,.65,.87,.85], radius=.0013)
                    add_marker(renderer.scene, goal, [1., .45, .06, 1.], radius=.002)
                    path_overlay(renderer.scene, arrays['ee'][case, :frame+1:2], (55, 204, 136), .0015)
                    add_marker(renderer.scene, arrays['ee'][case, frame], [.15,.85,.50,1.], radius=.003)
                    image = Image.new('RGB', (1280, 800), '#ffffff')
                    image.paste(Image.fromarray(renderer.render()), (0, 95))
                    image.paste(inset, (970, 145)); draw = ImageDraw.Draw(image)
                    error = float(np.linalg.norm(arrays['ee'][case, frame]-goal)*1000)
                    draw.text((22, 15), f'E118 | depth cloud -> FM reach | best step {checkpoint_step:,}', font=font(25, True), fill='#17324d')
                    draw.text((22, 56), f'Held-out goal {case+1:02d} | voxel {record["voxel_id"]} | H25 / K15 / flow10 / noise0', font=font(17), fill='#607587')
                    # A projected ring locates the diagnostic target even behind the gripper.
                    cameras = renderer.scene.camera
                    eye = (cameras[0].pos+cameras[1].pos)*.5
                    forward = cameras[0].forward/np.linalg.norm(cameras[0].forward)
                    up = cameras[0].up/np.linalg.norm(cameras[0].up)
                    right = np.cross(forward, up)
                    delta = goal-eye; depth = float(delta@forward)
                    focal = 600*cameras[0].frustum_near/(cameras[0].frustum_top-cameras[0].frustum_bottom)
                    x = 480+focal*float(delta@right)/depth
                    y = 395-focal*float(delta@up)/depth
                    draw.ellipse((x-9, y-9, x+9, y+9), outline='#ffad43', width=2)
                    draw.text((x+13, y-12), 'target', font=font(14, True), fill='#ffad43')
                    draw.text((980, 110), 'POLICY INPUT CLOUD', font=font(18, True), fill='#148fbb')
                    draw.text((980, 445), '128 base-frame points', font=font(16), fill='#17324d')
                    draw.text((980, 475), 'Orange +: GT target', font=font(16), fill='#db8124')
                    draw.text((980, 505), '(overlay only)', font=font(14), fill='#607587')
                    draw.text((980, 552), f't = {frame/25:5.2f} s', font=font(20, True), fill='#17324d')
                    draw.text((980, 592), f'EE error: {error:6.2f} mm', font=font(19, True), fill='#17324d')
                    draw.text((22, 720), 'Orange: 2 cm target voxel | Cyan: input depth cloud | Green: measured EE path', font=font(18), fill='#17324d')
                    draw.text((22, 756), 'Reach + close in air. Voxel/cloud are diagnostic overlays; no physical object is grasped.', font=font(16), fill='#607587')
                    if frame == 300:
                        draw.text((22, 105), final_status, font=font(21, True), fill='#178761' if record['success_5mm'] else '#ad6425')
                        image.save(out/f'case_{case+1:02d}_final.png')
                        posters.append(image.copy())
                    writer.stdin.write(np.asarray(image).tobytes())
            finally:
                writer.stdin.close()
                assert writer.wait() == 0
            checks.append(validate_video(path, len(indices)))
    # Same first six cases, synchronized; outcomes never select the displayed cases.
    streams = [av.open(str(out/f'case_{i+1:02d}.mp4')) for i in range(6)]
    writer = encode(out/'overview.mp4', (1920, 800), 12.5)
    try:
        decoders = [iter(stream.decode(video=0)) for stream in streams]
        for _ in tqdm(indices, desc='Encode synchronized six-goal overview', mininterval=5):
            image = Image.new('RGB', (1920, 800), 'white')
            for i, decoder in enumerate(decoders):
                pixels = next(decoder).to_image().resize((640, 400), Image.Resampling.LANCZOS)
                image.paste(pixels, ((i%3)*640, (i//3)*400))
            writer.stdin.write(np.asarray(image).tobytes())
    finally:
        for stream in streams: stream.close()
        writer.stdin.close(); assert writer.wait() == 0
    checks.append(validate_video(out/'overview.mp4', len(indices)))
    poster = Image.new('RGB', (1920, 800), 'white')
    for i, image in enumerate(posters):
        poster.paste(image.resize((640, 400), Image.Resampling.LANCZOS), ((i%3)*640, (i//3)*400))
    poster.save(out/'poster.png')
    write(out/'video_checks.json', dict(status='passed', videos=checks,
        replay_EE_tolerance_m=2e-6, video_fps=12.5, source_policy_hz=25,
        annotations='GT voxel and center shown for diagnosis only; not extra policy input/physics geometry'))


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e118_rollout_README.md', out/'README.md')
    # Atomic trainer replacement leaves this linked snapshot unchanged.
    os.link(ROOT/'train/best.pt', out/'checkpoint.pt')
    saved = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
    checkpoint_step = saved['step']; checkpoint_hash = sha(out/'checkpoint.pt')
    assert saved['config']['clouds_sha256'] == sha(ROOT/'clouds/clouds.npy')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    model = PointCloudReach().cuda(); model.load_state_dict(saved['model'], strict=True); model.eval()
    with np.load(ROOT/'train/evaluation_indices.npz') as z:
        ids = z['test_episodes'][:16].copy()
    with np.load(ROOT/'train/split.npz') as z:
        assert set(ids.tolist()).issubset(z['test_trajectories'].tolist())
        assert set(ids.tolist()).isdisjoint(z['train_trajectories'].tolist())
    rows = [json.loads(line) for line in (DATA/'trajectories.jsonl').read_text().splitlines()]
    clouds = np.load(ROOT/'clouds/clouds.npy', mmap_mode='r')
    config = dict(status='running', checkpoint_step=checkpoint_step, checkpoint_sha256=checkpoint_hash,
        checkpoint_source=str(ROOT/'train/best.pt'), snapshot=str(out/'checkpoint.pt'),
        checkpoint_selection=saved['selection'], trajectory_ids=ids.tolist(),
        selection='first16 of original128 fixed held-out goals; videos first6 before outcomes',
        horizon=25, execute=15, flow_steps=10, noise_seed=0, policy_frames=300, policy_hz=25,
        camera_cloud='unchanged cached front depth ->128base-frame points;2cm isolated voxel',
        input='cloud + measured joints/jaw; explicit target center is never policy input',
        source_sha256=sha(Path(__file__)), rollout_sha256=sha(Path(__file__).with_name('cloud_eval.py')))
    write(out/'config.json', config); del saved
    started = time.perf_counter()
    result, physical = rollouts(model, clouds, ids, rows)
    result.update(checkpoint_step=checkpoint_step, checkpoint_sha256=checkpoint_hash,
                  rollout_elapsed_s=time.perf_counter()-started, subset_not_full_test=True)
    write(out/'rollout.json', result); np.savez_compressed(out/'rollout.npz', **physical)
    del model; torch.cuda.empty_cache()
    render(out, physical, result, clouds, checkpoint_step)
    config.update(status='complete', elapsed_s=time.perf_counter()-started)
    write(out/'config.json', config)
    with (out/'README.md').open('a') as f:
        f.write(f'\nComplete. Actual checkpoint:{checkpoint_step}; results:rollout.json, media:video_checks.json.\n'
                f'\nSummary:{json.dumps(result["summary"])}\n')
    print(json.dumps(dict(status='complete', step=checkpoint_step, summary=result['summary'], output=str(out))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'test_k15')
    main(parser.parse_args().output)
