"""Draw the actual E118/E88 voxel split, using the existing real-arm web mesh."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/workspace/robotics_assets/experiments/e118_cloud')
DATA = Path('/workspace/robotics_assets/datasets/reach_voxel_grid/ee_delta_25hz_h25_s2')
PACK = REPO/'docs/research/manipulation_slides/assets3d'
OUT = ROOT/'voxel_split'
COLORS = ['#2f79ba', '#159b86', '#ee8732']
NAMES = ['Train', 'Validation', 'Test']


def main():
    OUT.mkdir(exist_ok=False)
    points = np.load(DATA/'target_base_m.npy')
    labels = np.full(len(points), -1, np.int8)
    with np.load(ROOT/'train/split.npz') as z:
        ids = [z[name+'_trajectories'].copy() for name in ('train', 'val', 'test')]
    for label, selected in enumerate(ids):
        assert len(np.unique(selected)) == len(selected)
        assert (labels[selected] == -1).all()
        labels[selected] = label
    assert (labels >= 0).all() and len(points) == 32017
    assert len(np.unique(points, axis=0)) == len(points)
    distances = cKDTree(points[ids[0]]).query(points[ids[2]])[0]
    stats = dict(status='complete', goals=len(points), edge_m=.02,
        counts={name: len(i) for name, i in zip(NAMES, ids)},
        split='voxel-disjoint random holdout; spatially interleaved, not separated workspace regions',
        source_split=str(ROOT/'train/split.npz'),
        split_sha256=hashlib.sha256((ROOT/'train/split.npz').read_bytes()).hexdigest(),
        base_bounds_m=[points.min(0).tolist(), points.max(0).tolist()],
        test_nearest_train_mm=dict(min=float(distances.min()*1000), median=float(np.median(distances)*1000),
            max=float(distances.max()*1000), fraction_at20mm=float(np.mean(np.isclose(distances, .02)))),
        robot='Existing original visual meshes at recorded settled home; no new policy/reachability result')
    (OUT/'verification.json').write_text(json.dumps(stats, indent=2)+'\n')
    (OUT/'split.json').write_text(json.dumps(dict(points=points.round(7).tolist(), labels=labels.tolist(),
        counts=stats['counts']), separators=(',', ':')))
    np.savez_compressed(OUT/'split.npz', centers_base_m=points, labels=labels,
        train_trajectories=ids[0], val_trajectories=ids[1], test_trajectories=ids[2], edge_m=.02)
    shutil.copytree(PACK/'vendor', OUT/'vendor')
    shutil.copyfile(PACK/'models/arm_home.glb', OUT/'arm_home.glb')
    slices(points, labels, ids)
    (OUT/'index.html').write_text(HTML)
    (OUT/'README.md').write_text('''# E118 reach voxel split visualization

Complete visualization only; no new training or evaluation. Exact source split:
../train/split.npz. Train25613, validation3202, test3202; disjoint voxel centers,
same interleaved feasible workspace. This is not a held-out workspace-region test.

voxel_split_3d.png: real home-pose robot mesh and 2cm cubes in three separate panels.
voxel_slices.png/.svg: exact horizontal/vertical categorical grid slices.
index.html: offline interactive viewer; serve this folder with a local HTTP server.
Toggle faint train context in validation/test, opacity, and orbit/zoom each view.
split.npz/.json preserve all centers and split labels. verification.json records
counts, bounds, split hash and nearest-training-voxel distances. The arm GLB and
vendored Three.js modules are copied unchanged from the existing slide assets.
Reproduce: .venv/bin/python -m OATFlow.reach.visualize_split (fresh output folder).
Training source weights/epochs/batch/LR: unchanged E118; this visualization does
not load policy weights, run inference, or change the running trainer.
''')
    print(json.dumps(stats), flush=True)


def slices(points, labels, ids):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.6), facecolor='white')
    fig.suptitle('Reach workspace: the actual train / validation / test voxel split',
                 x=.06, ha='left', y=.98, fontsize=18, fontweight='bold', color='#17324d')
    fig.text(.06, .927, 'Held-out voxel centers are interleaved with training goals. Cell width: 2 cm.',
             color='#536778', fontsize=11)
    cases = [(2, .11), (2, .25), (2, .43), (1, -.20), (1, 0.), (1, .20)]
    coordinate = ['X', 'Y', 'Z']
    cmap = ListedColormap(COLORS); cmap.set_bad('#eef1f4')
    for ax, (fixed, value) in zip(axes.flat, cases):
        horizontal, vertical = [k for k in range(3) if k != fixed]
        available = np.unique(points[:, fixed]); actual = available[np.argmin(abs(available-value))]
        keep = np.isclose(points[:, fixed], actual, atol=1e-7)
        a, b = np.unique(points[:, horizontal]), np.unique(points[:, vertical])
        cells = np.full((len(b), len(a)), np.nan)
        ia = np.rint((points[keep, horizontal]-a.min())/.02).astype(int)
        ib = np.rint((points[keep, vertical]-b.min())/.02).astype(int)
        cells[ib, ia] = labels[keep]
        ax.imshow(np.ma.masked_invalid(cells), origin='lower', interpolation='nearest',
            extent=[(a.min()-.01)*100,(a.max()+.01)*100,(b.min()-.01)*100,(b.max()+.01)*100],
            cmap=cmap, vmin=-.5, vmax=2.5, aspect='equal')
        counts = np.bincount(labels[keep], minlength=3)
        ax.set_title(f'{coordinate[fixed]} = {actual*100:.0f} cm   ·   {keep.sum():,} voxels',
                     loc='left', fontsize=11, pad=8, fontweight='bold')
        ax.set_xlabel(f'Robot-base {coordinate[horizontal]} (cm)')
        ax.set_ylabel(f'Robot-base {coordinate[vertical]} (cm)')
        ax.tick_params(labelsize=9); ax.set_facecolor('#eef1f4')
    legend = [Patch(facecolor=color, label=f'{name}  {len(i):,} ({len(i)/len(points):.0%})')
              for color, name, i in zip(COLORS, NAMES, ids)]
    legend.append(Patch(facecolor='#eef1f4', label='No validated goal'))
    fig.legend(handles=legend, ncol=4, loc='lower center', frameon=False,
               bbox_to_anchor=(.5, .03), fontsize=10)
    fig.subplots_adjust(left=.065, right=.98, top=.86, bottom=.13, hspace=.43, wspace=.32)
    fig.savefig(OUT/'voxel_slices.png', dpi=190)
    fig.savefig(OUT/'voxel_slices.svg'); plt.close(fig)


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reach voxel split</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f7fafb;color:#17324d;font:15px system-ui,sans-serif}
header{padding:30px 35px 15px}h1{font-size:29px;letter-spacing:-.03em;margin:6px 0 10px}
p{color:#607587;margin:6px 0}.small{font-size:12px}.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;padding:0 20px}
section{position:relative;min-width:0}h2{font-size:20px;margin:15px 18px 0}.note{margin:5px 18px;color:#607587;font-size:13px}
.viewport{width:100%;height:65vh;min-height:400px}canvas{display:block}
footer{padding:10px 35px;color:#607587;font-size:13px;display:flex;gap:25px;align-items:center;flex-wrap:wrap}
input{accent-color:#2f79ba}button{border:1px solid #d5e0e7;background:white;border-radius:7px;padding:7px 13px;color:#17324d;cursor:pointer}
@media(max-width:850px){.panels{grid-template-columns:1fr}.viewport{height:60vh}}
</style><script type="importmap">{"imports":{"three":"./vendor/three.module.js"}}</script>
<header><p class="small">E118 / REACH WORKSPACE / 2 cm VOXELS</p>
<h1>Train and test sample the same feasible volume</h1>
<p>32,017 verified goals · disjoint voxel centers · random spatial holdout</p></header>
<div class="panels">
<section><h2 style="color:#2f79ba">Train · 25,613</h2><div class="note">80% of validated targets</div><div class="viewport" id="v0"></div></section>
<section><h2 style="color:#159b86">Validation · 3,202</h2><div class="note">10% · held-out voxel centers</div><div class="viewport" id="v1"></div></section>
<section><h2 style="color:#ee8732">Test · 3,202</h2><div class="note">10% · held-out voxel centers</div><div class="viewport" id="v2"></div></section>
</div><footer>
<label>Voxel opacity <input id="alpha" type="range" min=".02" max=".9" step=".01" value=".16"></label>
<label><input type="checkbox" id="context"> Faint train context in validation/test</label>
<label><input type="checkbox" id="robot" checked> Robot mesh</label>
<button id="reset">Reset views</button><span>Drag to orbit · scroll to zoom · actual 2 cm cubes</span>
</footer><footer>Original arm visuals at settled home. The split tests interpolation within the feasible volume, rather than a separate unseen workspace region.</footer>
<script type="module">
import * as THREE from 'three';
import {GLTFLoader} from './vendor/GLTFLoader.js';
import {OrbitControls} from './vendor/OrbitControls.js';
const data=await (await fetch('./split.json')).json();
const asset=await new GLTFLoader().loadAsync('./arm_home.glb');
const palette=['#2f79ba','#159b86','#ee8732'];const views=[];
const geometry=new THREE.BoxGeometry(.02,.02,.02);
function cubes(label,color,opacity){
 const selected=data.points.filter((p,i)=>data.labels[i]===label);
 const mat=new THREE.MeshStandardMaterial({color,transparent:true,opacity,depthWrite:false,roughness:.7});
 const mesh=new THREE.InstancedMesh(geometry,mat,selected.length);const matrix=new THREE.Matrix4();
 selected.forEach((p,i)=>{matrix.makeTranslation(p[0],p[2],-p[1]);mesh.setMatrixAt(i,matrix)});
 mesh.instanceMatrix.needsUpdate=true;mesh.computeBoundingSphere();return mesh;
}
function reset(v){v.camera.position.set(1.23,.94,1.4);v.controls.target.set(0,.22,0);v.controls.update();v.render()}
for(let index=0;index<3;index++){
 const element=document.getElementById('v'+index);
 const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
 renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
 renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.12;
 element.append(renderer.domElement);const scene=new THREE.Scene();scene.background=new THREE.Color('#f7fafb');
 const camera=new THREE.PerspectiveCamera(34,1,.01,15);const controls=new OrbitControls(camera,renderer.domElement);
 controls.minDistance=.8;controls.maxDistance=4;controls.maxPolarAngle=Math.PI*.495;
 scene.add(new THREE.HemisphereLight(0xe2eff9,0xcbd4d8,2.4));
 const light=new THREE.DirectionalLight(0xfff2e3,3.2);light.position.set(-1,2,1.5);scene.add(light);
 const fill=new THREE.DirectionalLight(0xe3edf9,1.5);fill.position.set(1,1,-1);scene.add(fill);
 const grid=new THREE.GridHelper(1.15,23,0xc1cfd6,0xe1e8ec);grid.position.y=-.002;scene.add(grid);
 const robot=asset.scene.clone(true);scene.add(robot);
 const cloud=cubes(index,palette[index],.16);cloud.renderOrder=2;scene.add(cloud);
 const context=index>0?cubes(0,'#73869a',.012):null;if(context){context.visible=false;scene.add(context)}
 const v={renderer,scene,camera,controls,robot,cloud,context,render:()=>renderer.render(scene,camera)};
 function resize(){const w=element.clientWidth,h=element.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();v.render()}
 controls.addEventListener('change',v.render);addEventListener('resize',resize);views.push(v);resize();reset(v);
}
document.getElementById('alpha').oninput=e=>{for(const v of views){v.cloud.material.opacity=Number(e.target.value);v.render()}};
document.getElementById('context').onchange=e=>{for(const v of views){if(v.context)v.context.visible=e.target.checked;v.render()}};
document.getElementById('robot').onchange=e=>{for(const v of views){v.robot.visible=e.target.checked;v.render()}};
document.getElementById('reset').onclick=()=>views.forEach(reset);
window.voxelViews=views;window.voxelSplitReady=true;
</script></html>'''


if __name__ == '__main__':
    main()
