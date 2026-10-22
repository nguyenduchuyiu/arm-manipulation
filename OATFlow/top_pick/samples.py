"""Contact sheets and measured train/test spatial and permutation separation."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from OATFlow.top_pick.collect import DATA,ROOT


def main():
    rows=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
    out=ROOT/'samples'
    if (out/'diversity.json').exists():return
    out.mkdir(exist_ok=True)
    import mujoco
    from OATFlow.top_pick.render import init
    import OATFlow.top_pick.render as rendering
    init(str(DATA))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    all_slots={}
    for row in rows:
        with np.load(DATA/row['path']/'labels.npz') as z:
            positions=z['entity_xyz_world_m'][0,:4,:2]
        slots=positions[np.array(row['slot_object_ids'])]
        all_slots.setdefault(row['evaluation_group'],[]).append(slots)
    for group in ('train','composition','spatial','combined'):
        candidates=[r for r in rows if r['evaluation_group']==group and r['target_id']==0]
        chosen=[candidates[i] for i in np.linspace(0,len(candidates)-1,4,dtype=int)]
        sheet=Image.new('RGB',(1280,390),'#14202b');draw=ImageDraw.Draw(sheet)
        for col,row in enumerate(chosen):
            start_file=DATA/row['path']/'start.png'
            if not start_file.exists():
                with np.load(DATA/row['path']/'sim_state.npz') as z:rendering.STATE.qpos[:]=z['qpos'][0]
                mujoco.mj_forward(rendering.MODEL,rendering.STATE)
                rendering.RGB.update_scene(rendering.STATE,camera='cam_top',scene_option=rendering.OPTION)
                Image.fromarray(rendering.RGB.render().copy()).save(start_file)
            sheet.paste(Image.open(start_file),(320*col,70))
            order=' '.join('RBGY'[i] for i in row['slot_object_ids'])
            draw.text((320*col+5,7),f'{group}: {row["path"]}\nslots: {order} | target RED\nHOME | all 4 visible',font=font,fill='white')
        sheet.save(out/f'{group}.png')
    train=np.unique(np.round(np.stack(all_slots['train']).reshape(-1,8),6),axis=0)
    metrics={}
    for group in ('val','composition','spatial','combined'):
        test=np.unique(np.round(np.stack(all_slots[group]).reshape(-1,8),6),axis=0)
        # Average Euclidean slot displacement to the nearest training layout.
        distances=np.linalg.norm((test[:,None]-train[None]).reshape(len(test),len(train),4,2),axis=-1).mean(-1)
        closest=distances.min(-1)*1000
        metrics[group]=dict(unique_layouts=len(test),nearest_train_layout_mean_mm=float(closest.mean()),
                            nearest_train_layout_min_mm=float(closest.min()),nearest_train_layout_max_mm=float(closest.max()))
    permutations={group:sorted({r['permutation'] for r in rows if r['evaluation_group']==group}) for group in all_slots}
    assert set(permutations['train']).isdisjoint(permutations['composition'])
    assert set(permutations['train']).isdisjoint(permutations['combined'])
    report=dict(by_group=Counter(r['evaluation_group'] for r in rows),permutations=permutations,
                spatial=metrics,spatial_distribution='same randomization range; new draws, not out-of-range extrapolation')
    (out/'diversity.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'README.md').write_text('# e102_top samples\n\nActual initial top RGB for four RedCube queries per split. '
        'All start at settled home; four cubes visible, covers parked. Permutation labels describe left-to-right nominal slots. '
        'Composition uses train geometry/unseen permutations; spatial uses new geometry/seen permutations; combined uses both new. '
        'Spatial test draws use the same range as train. diversity.json quantifies nearest training layout distance. '
        'Images show expert dataset initial states, not learned policy rollouts. No training in this folder.\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
