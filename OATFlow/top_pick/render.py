"""Render only top RGB; camera/lighting/robot remain from the original scene."""
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES
from OATFlow.simulation.render import visual_model
from OATFlow.simulation.io import encoder
from OATFlow.top_pick.collect import DATA


def init(root):
    global ROOT,MODEL,STATE,RGB,SEG,OPTION
    ROOT=Path(root);MODEL=visual_model(ROOT/'scene.xml',object_bodies=NAMES)
    MODEL.vis.global_.offwidth=MODEL.vis.global_.offheight=320
    STATE=mujoco.MjData(MODEL);RGB=mujoco.Renderer(MODEL,height=320,width=320)
    OPTION=mujoco.MjvOption();OPTION.geomgroup[3]=0
    SEG=mujoco.Renderer(MODEL,height=320,width=320);SEG.enable_segmentation_rendering()
    atexit.register(RGB.close);atexit.register(SEG.close)


def render(row):
    path=ROOT/row['path']
    with np.load(path/'sim_state.npz') as z:qpos=z['qpos']
    writer=encoder(path/'rgb_top.mp4',320,320)
    for i,q in enumerate(qpos):
        STATE.qpos[:]=q;mujoco.mj_forward(MODEL,STATE)
        RGB.update_scene(STATE,camera='cam_top',scene_option=OPTION);rgb=RGB.render().copy()
        if i==0:
            Image.fromarray(rgb).save(path/'start.png')
            SEG.update_scene(STATE,camera='cam_top',scene_option=OPTION);ids=SEG.render()[:,:,0]
            visible={n:int(np.isin(ids,np.flatnonzero(MODEL.geom_bodyid==MODEL.body(n).id)).sum()) for n in NAMES[:4]}
            assert min(visible.values())>=5,(row['path'],visible)
            (path/'visibility.json').write_text(json.dumps(visible)+'\n')
        writer.stdin.write(rgb.tobytes())
    writer.stdin.close();assert writer.wait()==0
    return len(qpos)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=DATA)
    args=p.parse_args();rows=[json.loads(s) for s in (args.data/'manifest.jsonl').read_text().splitlines()]
    from OATFlow.top_pick.samples import main as samples
    if args.data==DATA:samples()
    total=0
    with ProcessPoolExecutor(8,mp_context=mp.get_context('spawn'),initializer=init,initargs=(str(args.data),)) as pool:
        for frames in tqdm(pool.map(render,rows,chunksize=4),total=len(rows),desc='Top RGB',mininterval=5):total+=frames
    (args.data/'render_summary.json').write_text(json.dumps(dict(status='complete',episodes=len(rows),frames=total,views=['top']))+'\n')
    r=args.data/'README.md';r.write_text(r.read_text().replace('physics complete; RGB pending','complete'))


if __name__=='__main__':main()
