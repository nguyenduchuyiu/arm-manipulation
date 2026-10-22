"""Validate a free-XY pick pilot and compare its real initial images/spawns with E102."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES, write_json
from OATFlow.top_pick import render as rendering
from OATFlow.top_pick.wide import ROOT, LOW, HIGH
from OATFlow.tracking.mask_video import font


OLD = Path('/workspace/robotics_assets/experiments/e102_top/data')


def arrays(path):
    with np.load(path) as z: return {k:z[k] for k in z.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'pilot')
    parser.add_argument('--output', type=Path, default=ROOT/'samples')
    args = parser.parse_args(); root, out = args.data, args.output
    assert json.loads((root/'config.json').read_text())['status'] == 'physics_complete'
    out.mkdir(exist_ok=False)
    shutil.copyfile('docs/research/e117_README.md', out/'README.md')
    rows = [json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
    model = mujoco.MjModel.from_xml_path(str(root/'scene.xml')); state = mujoco.MjData(model)
    rendering.init(str(root))
    jaws = {model.geom(f'link_6_{side}_jaw_collision_0').id for side in ('left', 'right')}
    initial = {}; checked = []; representative = {}
    for row in tqdm(rows, desc='Verify physical picks and top visibility', mininterval=5):
        directory = root/row['path']; stored = arrays(directory/'sim_state.npz')
        key = (row['layout_id'], row['permutation'])
        if key in initial:
            for k in ('initial_qpos', 'initial_qvel', 'initial_ctrl'):
                np.testing.assert_array_equal(stored[k], initial[key][k])
        else: initial[key] = {k:stored[k] for k in ('initial_qpos', 'initial_qvel', 'initial_ctrl')}
        body = model.body(row['target_object_id']).id
        positions = arrays(directory/'labels.npz')['entity_xyz_world_m']
        start = positions[0, :4, 2]; longest = run = 0
        for i, qpos in enumerate(stored['qpos']):
            state.qpos[:] = qpos; state.qvel[:] = stored['qvel'][i]
            mujoco.mj_forward(model, state)
            target_z = state.xpos[body, 2]-start[row['target_id']]
            touching = {j for c in state.contact for j in jaws if
                        (c.geom1 == j and model.geom_bodyid[c.geom2] == body) or
                        (c.geom2 == j and model.geom_bodyid[c.geom1] == body)}
            run = run+1 if target_z > .04 and touching == jaws else 0
            longest = max(longest, run)
            others = [j for j in range(4) if j != row['target_id']]
            assert np.all(positions[i, others, 2]-start[others] <= .04)
        assert longest >= 10 and positions[-1, row['target_id'], 2]-start[row['target_id']] > .04
        rendering.STATE.qpos[:] = stored['initial_qpos']
        mujoco.mj_forward(rendering.MODEL, rendering.STATE)
        rendering.SEG.update_scene(rendering.STATE, camera='cam_top', scene_option=rendering.OPTION)
        ids = rendering.SEG.render()[..., 0]
        visible = {name:int(np.isin(ids, np.flatnonzero(rendering.MODEL.geom_bodyid == rendering.MODEL.body(name).id)).sum()) for name in NAMES[:4]}
        assert min(visible.values()) >= 5
        checked.append(dict(path=row['path'], bilateral_streak=longest, visibility=visible))
        if row['permutation'] == 0 and row['target_id'] == 0:
            rendering.RGB.update_scene(rendering.STATE, camera='cam_top', scene_option=rendering.OPTION)
            image = Image.fromarray(rendering.RGB.render().copy())
            image.save(out/f'new_g{row["layout_id"]:03d}.png'); representative[row['layout_id']] = image
    for g in representative:
        old_state = arrays(OLD/f'g{g:03d}/p00/t0/sim_state.npz')
        rendering.STATE.qpos[:] = old_state['initial_qpos']
        mujoco.mj_forward(rendering.MODEL, rendering.STATE)
        rendering.RGB.update_scene(rendering.STATE, camera='cam_top', scene_option=rendering.OPTION)
        Image.fromarray(rendering.RGB.render().copy()).save(out/f'old_g{g:03d}.png')
    # Rerender old and new states with the identical current visual model and camera.
    sheet = Image.new('RGB', (1280, 860), '#f5f7fa'); draw = ImageDraw.Draw(sheet)
    draw.text((20, 10), 'Pick spawn comparison | identical camera, objects and home start', font=font(24, True), fill='#21354a')
    for col, g in enumerate((0, 36)):
        for row, (image, label) in enumerate([
                (Image.open(out/f'old_g{g:03d}.png'), 'E102: four fixed spawn regions'),
                (representative[g], 'E117 trial: four free XY points')]):
            x, y = col*640, 45+row*400
            draw.text((x+18, y), f'{label} | layout {g}', font=font(20, True), fill='#21354a')
            sheet.paste(image.resize((360, 360)), (x+140, y+30))
    draw.text((20, 845), 'Actual expert-data initial images; two accepted new layouts, 16 verified demonstrations. No learned-policy results.', font=font(14), fill='#52647b')
    sheet.save(out/'comparison.png')
    old = np.array([json.loads((OLD/f'g{g:03d}/layout.json').read_text())['positions'] for g in range(32)])
    groups = sorted(representative)
    new = np.array([json.loads((root/f'g{g:03d}/layout.json').read_text())['positions'] for g in groups])
    # The nearest-layout metric matches E102: average XY displacement over four ordered points.
    distance = np.linalg.norm(new[:, None]-old[None], axis=-1).mean(-1)
    nearest = distance.min(-1)*1000
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.scatter(*old.reshape(-1, 2).T, s=24, color='#427cad', alpha=.4, label='E102: 32 training layouts')
    colors = ['#d85c2c', '#8052a0']
    for g, points, color in zip(groups, new, colors):
        ax.scatter(*points.T, marker='X', s=100, color=color, edgecolors='white', linewidths=.6, label=f'New accepted layout {g}')
    ax.plot([LOW[0], HIGH[0], HIGH[0], LOW[0], LOW[0]], [LOW[1], LOW[1], HIGH[1], HIGH[1], LOW[1]], '--', color='#8595a3', label='Proposed sampling bounds')
    ax.set(xlabel='Table X (m)', ylabel='Table Y (m)', title='Wider free-position sampling: proposals vs physically accepted examples',
           xlim=(.28, .80), ylim=(-.44, -.08), aspect='equal')
    ax.grid(alpha=.2); ax.legend(fontsize=9, loc='lower right')
    fig.savefig(out/'spawn_positions.png', dpi=180); fig.savefig(out/'spawn_positions.svg'); plt.close(fig)
    failures = json.loads((root/'audit.json').read_text())['rejected']
    categories = Counter('separation' if 'overlapping' in r['error'] else 'IK' if r['error'].startswith('Oracle IK') else 'physical pick' for r in failures)
    report = dict(status='passed', episodes=len(rows), accepted_layouts=len(groups), initial_states_query_independent=True,
        verified_physical_picks=checked, proposed_xy_low_m=LOW.tolist(), proposed_xy_high_m=HIGH.tolist(),
        accepted_xy_min_m=new.min((0, 1)).tolist(), accepted_xy_max_m=new.max((0, 1)).tolist(),
        nearest_old_training_layout_mm={str(g):float(d) for g, d in zip(groups, nearest)},
        rejected_layouts=len(failures), rejection_categories=dict(categories),
        interpretation='Two accepted trial layouts, not a success-rate benchmark or proof that every proposed position is feasible.')
    write_json(out/'verification.json', report)
    rendering.RGB.close(); rendering.SEG.close()
    print(json.dumps({k:v for k, v in report.items() if k != 'verified_physical_picks'}), flush=True)


if __name__ == '__main__': main()
