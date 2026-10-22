"""Audit dynamic shuffle states, real contact replay, front RGB-D and fisheye wrist RGB."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import BASE, NAMES, SEEN, UNSEEN, write_json
from OATFlow.tracking.mask_video import digest
from OATFlow.data_tools.wrist_camera import INTRINSICS


STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def arrays(path):
    with np.load(path) as z: return {k:z[k] for k in z.files}


def replay(model, stored, context):
    data = mujoco.MjData(model)
    initial = stored['integration_state'][0] if context else stored['initial_integration_state']
    mujoco.mj_setState(model, data, initial, STATE)
    qpos_error = qvel_error = penetration = 0.
    n = len(stored['qpos'])
    for i in range(n):
        data.ctrl[:] = stored['ctrl'][i]; data.eq_active[:] = stored['eq_active'][i]
        data.mocap_quat[:] = stored['mocap_quat'][i]
        mujoco.mj_forward(model, data)
        qpos_error = max(qpos_error, float(np.abs(data.qpos-stored['qpos'][i]).max()))
        qvel_error = max(qvel_error, float(np.abs(data.qvel-stored['qvel'][i]).max()))
        penetration = max(penetration, max((-c.dist for c in data.contact), default=0.))
        if context:
            begin = data.mocap_pos.copy(); end = stored['mocap_target'][i]
            for step in range(20):
                u = (step+1)/20; data.mocap_pos[:] = begin*(1-u)+end*u
                mujoco.mj_step(model, data)
        elif i+1 < n:
            mujoco.mj_step(model, data, 20)
    if context:
        terminal = np.empty(mujoco.mj_stateSize(model, STATE)); mujoco.mj_getState(model, data, terminal, STATE)
        terminal_error = float(np.abs(terminal-stored['terminal_integration_state']).max())
    else: terminal_error = 0.
    assert qpos_error < 1e-7 and qvel_error < 1e-6, (qpos_error, qvel_error)
    return dict(max_qpos_error=qpos_error, max_qvel_error=qvel_error,
                terminal_full_state_error=terminal_error, max_sampled_penetration_m=penetration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args(); root = args.data
    config = json.loads((root/'config.json').read_text())
    assert config['status'] in ('render_complete', 'complete')
    assert config['depth_views'] == ['front']
    rows = [json.loads(l) for l in (root/'manifest.jsonl').read_text().splitlines()]
    assert len(rows) == config['expected_episodes']
    grouped = {}
    for row in rows: grouped.setdefault(row['context_path'], []).append(row)
    assert len(grouped) == config['expected_contexts']
    model = mujoco.MjModel.from_xml_path(str(root/'scene.xml'))
    assert model.ncam == 2 and {model.camera(i).name for i in range(model.ncam)} == {'cam_front', 'wrist'}
    assert set(SEEN).isdisjoint(UNSEEN)
    checks = []; contacts = []; relative = []; wrist_variations = []
    total_frames = 0; start = time.perf_counter(); joints = []
    first_by_layout = {}
    for ref in sorted(grouped): first_by_layout.setdefault(grouped[ref][0]['layout_id'], ref)
    for ref, queries in tqdm(sorted(grouped.items()), desc='Audit dynamics and paired RGB-D', mininterval=5):
        assert {r['target_id'] for r in queries} == {0, 1, 2, 3}
        meta = json.loads((root/ref/'context.json').read_text())
        contacts.append(meta['dynamics']['cover_object_contact_substeps'])
        relative.append(meta['dynamics']['max_relative_translation_m'])
        assert all(v > 0 for v in contacts[-1]) and all(v > .001 for v in relative[-1])
        ctx = arrays(root/ref/'sim_state.npz')
        context_masks = arrays(root/ref/'masks.npz')
        do_replay = config['pilot'] or ref == first_by_layout[queries[0]['layout_id']]
        if do_replay: checks.append(dict(path=ref, kind='context', **replay(model, ctx, True)))
        for segment_ref, segment in [(ref, ctx)]+[(r['path'], arrays(root/r['path']/'sim_state.npz')) for r in queries]:
            directory = root/segment_ref; n = len(segment['qpos']); total_frames += n
            render = json.loads((directory/'render_complete.json').read_text())
            assert set(render['depth_checks']) == {'front'}
            assert render['depth_checks']['front']['frames'] == n and render['depth_checks']['front']['codec_lossless']
            assert not (directory/'depth_wrist.mkv').exists()
            calibration = arrays(directory/'calibration.npz')
            assert calibration['views'].tolist() == ['front', 'wrist'] and float(calibration['depth_unit_m']) == .002
            assert calibration['projection_model'].tolist() == ['pinhole', 'fisheye_equidistant']
            assert calibration['depth_views'].tolist() == ['front']
            np.testing.assert_allclose(calibration['intrinsics'][1], INTRINSICS)
            np.testing.assert_array_equal(calibration['distortion_coefficients'], np.zeros((2, 4)))
            np.testing.assert_array_equal(calibration['visualization_rotation_deg'], [0, 180])
            assert np.isclose(calibration['render_clip_near_m'], .001)
            assert np.isclose(calibration['render_clip_far_m'], 10.)
            np.testing.assert_array_equal(calibration['timestamp_s'], segment['timestamp_s'])
            world = calibration['camera_to_world']; base = calibration['camera_to_base']
            np.testing.assert_allclose(world[..., :3, 3]-BASE, base[..., :3, 3], atol=1e-12)
            np.testing.assert_allclose(world[:, 0], np.broadcast_to(world[0, 0], world[:, 0].shape), atol=1e-12)
            wrist_variations.append(float(np.linalg.norm(np.ptp(world[:, 1, :3, 3], axis=0))))
            data = mujoco.MjData(model)
            for frame in (0, n//2, n-1):
                data.qpos[:] = segment['qpos'][frame]; mujoco.mj_forward(model, data)
                for vi, cam in enumerate(('cam_front', 'wrist')):
                    cid = model.camera(cam).id
                    np.testing.assert_allclose(world[frame, vi, :3, 3], data.cam_xpos[cid], atol=1e-12)
                    np.testing.assert_allclose(world[frame, vi, :3, :3], data.cam_xmat[cid].reshape(3, 3)@np.diag([1., -1., -1.]), atol=1e-12)
            masks = arrays(directory/'masks.npz')
            assert masks['instance'].shape == (n, 2, 240, 320)
            np.testing.assert_array_equal(masks['timestamp_s'], segment['timestamp_s'])
            assert masks['target_in_fov'].shape[:2] == (n, 2)
        for row in queries:
            prompt = context_masks['instance'][0, 0] == row['target_id']+1
            assert prompt.sum() > 20
            np.savez_compressed(root/row['path']/'prompt.npz', query_mask=prompt,
                                view=np.array('front'), frame=0, timestamp_s=ctx['timestamp_s'][0])
            expert = arrays(root/row['path']/'sim_state.npz')
            np.testing.assert_array_equal(expert['initial_integration_state'], ctx['terminal_integration_state'])
            sup = arrays(root/row['path']/'supervision.npz')
            assert sup['action_valid'][:-1].all() and not sup['action_valid'][-1]
            assert row['success'] and row['bilateral_lift_streak_frames'][row['target_object_id']] >= 10
            assert row['target_wrist'] is not None
            if do_replay: checks.append(dict(path=row['path'], kind='expert', **replay(model, expert, False)))
            if row['split'] == 'train':
                obs = arrays(root/row['path']/'observation.npz'); joints.append(obs['joint_position'][sup['action_valid']])
    assert max(wrist_variations) > .05, 'wrist camera did not actually move'
    if not config['pilot']:
        assert Counter(r['evaluation_group'] for r in rows) == dict(train=2048, val=256, composition=1024, spatial=256, combined=128)
    joint = np.concatenate(joints)
    write_json(root/'normalization.json', dict(applied=False, measured_joint_train_mean=joint.mean(0).tolist(),
        measured_joint_train_std=joint.std(0).tolist(), measured_joint_train_frames=len(joint),
        gripper_fixed_range_m=[-.0255, 0], gripper_training_formula='-1 - 2*g/.0255'))
    size = sum(p.stat().st_size for p in root.glob('g*/p*/**/*') if p.is_file())
    result = dict(status='passed', episodes=len(rows), contexts=len(grouped), unique_frames=total_frames,
        groups=dict(Counter(r['evaluation_group'] for r in rows)), bytes=size,
        estimated_full_bytes=size*(928/len(grouped)), replay=checks,
        minimum_contact_substeps=int(np.min(contacts)), maximum_relative_translation_m=float(np.max(relative)),
        wrist_max_position_range_m=max(wrist_variations), elapsed_s=time.perf_counter()-start,
        audit_source_sha256=digest(Path(__file__)))
    write_json(root/'audit.json', result)
    config['status'] = 'complete'; write_json(root/'config.json', config)
    with (root/'README.md').open('a') as stream:
        stream.write(f'\nActual status: COMPLETE. {len(rows)}episodes/{len(grouped)}contexts; '
                     f'{total_frames}unique frames; {size/1e9:.3f}GB. See audit.json and render_summary.json. '
                     'Front RGB-D + fisheye wrist RGB, both calibrated; front depth2mm FFV1 verified lossless; prompts are frame0-only.\n')
    print('AUDIT', json.dumps(result), flush=True)


if __name__ == '__main__': main()
