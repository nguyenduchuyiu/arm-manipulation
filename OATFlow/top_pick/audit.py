"""Verify actual query pairing, home state, physical command replay and balance."""
from collections import Counter
import json

import mujoco
import numpy as np
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import Scene,NAMES
from OATFlow.top_pick.collect import DATA,ROOT


def main():
    rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    model=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
    scene=Scene(model);reference={};home=None;worst=0.;replayed=[]
    replay_paths={r['path'] for group in ('train','val','composition','spatial','combined')
                  for r in [next(r for r in rows if r['evaluation_group']==group)]}
    for row in tqdm(rows,desc='Audit home/query state and balance',mininterval=5):
        path=DATA/row['path']
        with np.load(path/'sim_state.npz') as z:
            qpos,qvel,ctrl=(z['initial_'+key] for key in ('qpos','qvel','ctrl'))
            full=np.r_[qpos,qvel,ctrl];key=(row['layout_id'],row['permutation'])
            if key in reference:np.testing.assert_array_equal(reference[key],full)
            else:reference[key]=full
            robot=qpos[scene.qadr]
            if home is None:home=robot
            np.testing.assert_allclose(robot,home,atol=1e-10,rtol=0)
            for slot,color in enumerate(row['slot_object_ids']):
                expected=json.loads((DATA/f'g{row["layout_id"]:03d}/layout.json').read_text())['positions'][slot]
                np.testing.assert_allclose(qpos[scene.target_qadr[NAMES[color]]:scene.target_qadr[NAMES[color]]+2],expected,atol=.003,rtol=0)
            if row['path'] in replay_paths:
                saved_qpos=z['qpos'];saved_qvel=z['qvel']
        if row['path'] in replay_paths:
            with np.load(path/'supervision.npz') as z:commands=z['joint_command'];valid=z['action_valid']
            state=mujoco.MjData(model);state.qpos[:]=qpos;state.qvel[:]=qvel;state.ctrl[:]=ctrl
            mujoco.mj_forward(model,state)
            for i in range(len(commands)-1):
                state.ctrl[:]=commands[i]
                mujoco.mj_step(model,state,20);mujoco.mj_forward(model,state)
                worst=max(worst,float(np.max(np.abs(state.qpos-saved_qpos[i+1]))))
            assert worst<1e-8,worst
            replayed.append(row['path'])
    balance={}
    for group in ('train','val','composition','spatial','combined'):
        cells=Counter((r['target_id'],r['target_slot']) for r in rows if r['evaluation_group']==group)
        assert len(cells)==16 and len(set(cells.values()))==1
        balance[group]=dict(episodes=sum(cells.values()),count_per_color_slot=next(iter(cells.values())))
    result=dict(status='passed',episodes=len(rows),paired_scenes=len(reference),query_initial_state='bit-identical',
                initial_robot='common settled home for all episodes',color_slot_balance=balance,
                physical_replays=replayed,max_replay_qpos_error=worst)
    (DATA/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
