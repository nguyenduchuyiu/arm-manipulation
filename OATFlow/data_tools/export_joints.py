"""Export measured chunk-start joints aligned with the existing e86 EE labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm

from OATFlow.simulation.pose import ARM_JOINTS

CHUNKS = Path('/workspace/robotics_assets/datasets/reach_voxel_grid/ee_delta_25hz_h25_s2')
OUTPUT = CHUNKS / 'joint_states'
SOURCE = {}


def initialize(root, addresses):
    SOURCE.update(root=Path(root), addresses=addresses)


def measured_chunk_states(row):
    with np.load(SOURCE['root']/row['source_trajectory_file']) as z:
        total = row['source_frames_50hz']
        assert len(z['action_absolute']) == total
        starts = np.arange(0,(total+1)//2,2,dtype=np.int32)
        indices = 2*starts
        joints = z['state_joint_position'][indices]
        assert joints.shape == (row['chunk_end']-row['chunk_begin'],6)
        assert np.isfinite(joints).all()
        np.testing.assert_array_equal(joints,z['state_full_qpos'][indices][:,SOURCE['addresses']].astype(np.float32))
        return joints,starts,z['ee_position_world_m'][indices]


def main():
    global CHUNKS, OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks',type=Path,required=True)
    CHUNKS=parser.parse_args().chunks; OUTPUT=CHUNKS/'joint_states'
    config = json.loads((CHUNKS/'config.json').read_text())
    assert config['status'] == 'complete'
    assert json.loads((CHUNKS/'audit.json').read_text())['status'] == 'passed'
    root = CHUNKS.parent
    physics = mujoco.MjModel.from_xml_path(str(CHUNKS/'scene.xml'))
    names = [*ARM_JOINTS,'right_jaw_slide_joint']
    addresses = np.array([physics.jnt_qposadr[physics.joint(n).id] for n in names])
    rows = [json.loads(line) for line in (CHUNKS/'trajectories.jsonl').read_text().splitlines()]
    anchors = np.load(CHUNKS/'anchor_ee.npy',mmap_mode='r')
    states = np.load(CHUNKS/'state_ee_pose_base.npy',mmap_mode='r')
    starts = np.load(CHUNKS/'chunk_start_25hz.npy',mmap_mode='r')
    episodes = np.load(CHUNKS/'chunk_trajectory_index.npy',mmap_mode='r')
    OUTPUT.mkdir(exist_ok=False)
    description = ('# e86 chunk joint input states\n\nStatus: exporting.\n\n'
        'Measured five arm joints(rad) and right-jaw slide(m) at source state2*chunk_start. '
        'Same chunk ordering and unchanged EE output labels in e86_grid/ee_delta_25hz_h25_s2. '
        'No action arrays copied. Gripper0=open,-0.0255=closed. Raw physical units on disk; '
        'train-only per-joint mean/std and fixed-range gripper transform inside policy. '
        'All chunks audited against measured full qpos, source EE anchors and jaw observations. '
        'No training/checkpoint/epochs/batch/LR or learned evaluation applied during export. '
        'Exact provenance: config.json; checks: audit.json.\n')
    (OUTPUT/'README.md').write_text(description)
    record = dict(status='exporting',source_chunks=str(CHUNKS),source_dataset=str(root),
        chunks=len(anchors),episodes=len(rows),state_dim=6,state_representation='measured_arm_joints_right_jaw',
        joint_names=names,state_units=['rad']*5+['m'],normalization_applied=False,
        anchor='measured source state2*chunk_start_25hz',workers=8,
        source_hashes={n:hashlib.sha256((CHUNKS/n).read_bytes()).hexdigest()
                       for n in ('config.json','audit.json','trajectories.jsonl')})
    (OUTPUT/'config.json').write_text(json.dumps(record,indent=2)+'\n')
    output = np.lib.format.open_memmap(OUTPUT/'state_joint_position.npy',mode='w+',dtype=np.float32,
                                      shape=(len(anchors),6))
    with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(root,addresses)) as pool:
        values = pool.map(measured_chunk_states,rows,chunksize=16)
        for ep,(row,(joint,chunk_starts,ee)) in enumerate(tqdm(zip(rows,values),total=len(rows),
                desc='Export/audit measured joint inputs',mininterval=5)):
            lo,hi = row['chunk_begin'],row['chunk_end']
            np.testing.assert_array_equal(chunk_starts,starts[lo:hi])
            np.testing.assert_array_equal(episodes[lo:hi],np.full(hi-lo,ep))
            np.testing.assert_array_equal(ee,anchors[lo:hi])
            np.testing.assert_array_equal(joint[:,-1],states[lo:hi,-1])
            output[lo:hi] = joint
    output.flush()
    record['status'] = 'complete'
    (OUTPUT/'config.json').write_text(json.dumps(record,indent=2)+'\n')
    audit = dict(status='passed',audited_episodes=len(rows),audited_chunks=len(anchors),
        checks=['measured full qpos','chunk starts/order','trajectory membership','EE anchors','measured jaw'],
        state_sha256=hashlib.sha256((OUTPUT/'state_joint_position.npy').read_bytes()).hexdigest())
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (OUTPUT/'README.md').write_text(description.replace('Status: exporting.','Status: complete; all chunks audited.'))
    print(json.dumps(audit),flush=True)


if __name__ == '__main__':
    main()
