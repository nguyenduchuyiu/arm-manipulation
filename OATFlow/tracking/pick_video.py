"""Three held-out e96 tracking -> e88 reach/grip rollouts and IK lift probes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

from OATFlow.simulation.render import visual_model, add_marker, path_overlay
from OATFlow.simulation.pose import ARM_JOINTS, PoseIK, rotation_matrix
from OATFlow.simulation.control import step
from OATFlow.reach import ReachPolicy
from OATFlow.simulation.contacts import jaw_contacts
from OATFlow.tracking.data import sample_indices, subset
from OATFlow.tracking.point import PointXYZ
from OATFlow.tracking.point_data import project, top_camera

DATA = Path('/workspace/robotics_assets/datasets/shuffle_pick_multiview')
TRACKER = Path('/workspace/robotics_assets/models/point_tracking_local/best.pt')
FM = Path('/workspace/robotics_assets/models/reach_joint_delta_ee/best.pt')
OUT = None
NAMES = ('RedCube', 'BlueCube', 'GreenCube', 'YellowCube', 'cover_a', 'cover_b')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(row, model, camera):
    end = row['decision_frames']['t_obj']
    arrays = []
    images, indices = [], []
    for ref, offset, stop in ((row['context_path'], 0, row['context_frames'] - 1),
                              (row['path'], row['context_frames'], end - row['context_frames'])):
        with np.load(DATA / ref / 'sim_state.npz') as z:
            arrays.append({key: z[key][:stop + 1].copy() for key in ('qpos', 'qvel', 'ctrl')})
        wanted = set(np.r_[np.arange(0, stop + 1, 3), stop].tolist())
        with av.open(str(DATA / ref / 'rgb_top.mp4')) as video:
            for i, frame in enumerate(video.decode(video=0)):
                if i > stop:
                    break
                if i in wanted:
                    images.append(frame.to_ndarray(format='rgb24')[40:280].copy())
                    indices.append(offset + i)
    past = {key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
    assert len(past['qpos']) == end + 1 and indices[-1] == end
    initial = mujoco.MjData(model); initial.qpos[:] = past['qpos'][0]
    mujoco.mj_forward(model, initial)
    uv = project(initial.xpos[model.body(NAMES[row['target_id']]).id] - camera['base'], camera)
    query = torch.zeros(1, 1, 30, 240, 320)
    x, y = np.rint(uv).astype(int)
    yy, xx = torch.meshgrid(torch.arange(max(0, y-3), min(240, y+4)),
                            torch.arange(max(0, x-3), min(320, x+4)), indexing='ij')
    query[0, 0, 0, yy, xx] = ((xx-x).square() + (yy-y).square() <= 9).float()
    assert query.sum() > 0
    return dict(row=row, past=past, images=images, image_indices=indices, query=query,
                uv=uv, tracking=[], live=[], commands=[], plans=[], held=None)


@torch.no_grad()
def predict(tracker, case, ends):
    clips = [np.stack([case['images'][i] for i in sample_indices(end)]) for end in ends]
    rgb = torch.from_numpy(np.stack(clips)).permute(0, 4, 1, 2, 3).cuda()
    query = case['query'].expand(len(ends), -1, -1, -1, -1).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        values = tracker.predict(rgb, query).cpu().numpy()
    for end, value in zip(ends, values):
        case['tracking'].append((case['image_indices'][end], value.copy()))
    return values


def snapshot(model, state, case, phase):
    case['live'].append(dict(qpos=state.qpos.copy(), qvel=state.qvel.copy(), ctrl=state.ctrl.copy(),
        ee=state.site_xpos[model.site('grasp_site').id].copy(),
        objects=np.stack([state.xpos[model.body(n).id] for n in NAMES[:4]]),
        contacts=np.array([len(jaw_contacts(model, state, n)) for n in NAMES[:4]]), phase=phase))


@torch.no_grad()
def rollout(tracker, policy, physics, case, renderer, option, base):
    model, qadr, _, site = physics
    for start in tqdm(range(0, len(case['images']), 4), desc='Causal recorded-prefix tracking', mininterval=5):
        predict(tracker, case, list(range(start, min(start+4, len(case['images'])))))
    case['opening_goal'] = case['tracking'][-1][1].copy()
    state = mujoco.MjData(model)
    for key in ('qpos', 'qvel', 'ctrl'):
        getattr(state, key)[:] = case['past'][key][-1]
    with np.load('/workspace/robotics_assets/datasets/reach_voxel_grid/trajectories/v002043.npz') as home:
        state.qpos[:8] = home['state_full_qpos'][0]
        state.qvel[:8] = home['state_full_qvel'][0]
        state.ctrl[:] = home['initial_ctrl']
    mujoco.mj_forward(model, state)
    for body in NAMES:
        adr = model.jnt_qposadr[model.body_jntadr[model.body(body).id]]
        np.testing.assert_array_equal(state.qpos[adr:adr+7], case['past']['qpos'][-1, adr:adr+7])
    case['data'] = state; case['held'] = state.ctrl.copy()
    snapshot(model, state, case, 'FM reach/grip')
    ik = PoseIK(physics)
    generator = torch.Generator(device='cuda').manual_seed(0)
    for frame in tqdm(range(400), desc='FM300 + separate IK lift100', mininterval=5):
        if frame % 3 == 0:
            renderer.update_scene(state, camera='cam_top', scene_option=option)
            pixels = renderer.render()[40:280].copy()
            case['images'].append(pixels)
            case['image_indices'].append(len(case['past']['qpos']) + frame)
        if frame % 15 == 0:
            case['goal'] = predict(tracker, case, [len(case['images'])-1])[0]
        if frame < 300:
            if frame % 15 == 0:
                goal = torch.as_tensor(case['goal'][None], device='cuda')
                joints = torch.tensor(state.qpos[qadr][None], dtype=torch.float32, device='cuda')
                noise = policy.flow.sample_noise(1, 'cuda', generator)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    plan = policy.sample(goal, joints, noise, steps=10).float().cpu().numpy()[0]
                assert np.isfinite(plan).all()
                anchor = state.site_xpos[site].copy(); rotation = state.site_xmat[site].reshape(3,3).copy()
                case['plans'].append(dict(frame=frame, anchor=anchor.copy(), rotation=rotation.copy(),
                                          delta=plan.copy(), goal_base_m=case['goal'].copy()))
            action = plan[frame % 15]
            command = ik.command(case, anchor + action[:3], action[9], rotation_matrix(action[3:9]) @ rotation)
            phase = 'FM reach/grip'
        else:
            if frame == 300:
                lift_start = state.site_xpos[site].copy()
                lift_rotation = state.site_xmat[site].reshape(3,3).copy()
                lift_jaw = float(case['held'][-1])
            fraction = min((frame-299)/75, 1.)
            target = lift_start + [0, 0, .08 * fraction**2 * (3-2*fraction)]
            command = ik.command(case, target, lift_jaw, lift_rotation)
            phase = 'IK lift probe' if frame < 375 else 'IK hold probe'
        command = np.clip(command, *model.actuator_ctrlrange.T)
        step(model, state, case['held'], command)
        case['held'] = command.copy(); case['commands'].append(command.copy())
        assert np.isfinite(state.qpos).all()
        snapshot(model, state, case, phase)


def finish(physics, case, number, base):
    live = case['live']; target = case['row']['target_id']
    ee = np.stack([x['ee'] for x in live]); objects = np.stack([x['objects'] for x in live])
    contacts = np.stack([x['contacts'] for x in live]); lift = objects[:,target,2] - objects[0,target,2]
    lifted = (lift > .04) & (contacts[:,target] == 2)
    longest = count = 0
    for value in lifted:
        count = count+1 if value else 0; longest = max(longest,count)
    errors = np.linalg.norm(ee-objects[:,target],axis=-1)*1000
    record = dict(case=number,source=case['row']['path'],group=case['row']['evaluation_group'],target=NAMES[target],
        opening_vision_error_mm=float(np.linalg.norm(case['opening_goal']-(objects[0,target]-base))*1000),
        initial_live_vision_error_mm=float(np.linalg.norm(case['plans'][0]['goal_base_m']-(objects[0,target]-base))*1000),
        FM_final_true_error_mm=float(errors[300]),FM_min_true_error_mm=float(errors[:301].min()),
        FM_final_target_jaw_contacts=int(contacts[300,target]),
        max_probe_lift_mm=float(lift[300:].max()*1000),final_probe_lift_mm=float(lift[-1]*1000),
        lifted_bilateral_streak=longest,grasp_retention_success=longest>=10,
        reached_object_shifts_mm=(np.linalg.norm(objects[300]-objects[0],axis=-1)*1000).tolist(),
        FM_ik_position_max_mm=float(np.max(case['ik_errors'][:300])*1000),
        probe_ik_position_max_mm=float(np.max(case['ik_errors'][300:])*1000))
    saved = dict(qpos=np.stack([x['qpos'] for x in live]),qvel=np.stack([x['qvel'] for x in live]),
        ctrl=np.stack([x['ctrl'] for x in live]),commands=np.stack(case['commands']),ee_world_m=ee,
        objects_world_m=objects,jaw_contacts=contacts,phase=np.array([x['phase'] for x in live]),
        tracking_frame=np.array([x[0] for x in case['tracking']]),
        tracking_xyz_base_m=np.stack([x[1] for x in case['tracking']]),
        plan_delta=np.stack([p['delta'] for p in case['plans']]),
        plan_anchor=np.stack([p['anchor'] for p in case['plans']]),
        plan_anchor_rotation=np.stack([p['rotation'] for p in case['plans']]),
        FM_goal_base_m=np.stack([p['goal_base_m'] for p in case['plans']]),
        query_uv=case['uv'],ik_targets=case['ik_targets'],ik_errors=case['ik_errors'])
    np.savez_compressed(OUT/f'case_{number:02d}.npz',**saved)
    state = mujoco.MjData(physics[0])
    for key in ('qpos','qvel','ctrl'):
        getattr(state,key)[:] = saved[key][0]
    mujoco.mj_forward(physics[0],state); held=state.ctrl.copy(); worst=0.
    for i,command in enumerate(tqdm(case['commands'],desc='Verify physical command replay',mininterval=5)):
        step(physics[0],state,held,command);held=command
        worst=max(worst,float(np.max(np.abs(state.qpos-saved['qpos'][i+1]))))
    assert worst < 1e-10, worst
    record['replay_max_qpos_difference']=worst
    return record


def render(model, case, record, number, renderer, option, camera):
    past = case['past']; boundary=len(past['qpos']); total=boundary+len(case['live'])
    display=mujoco.MjData(model)
    view=mujoco.MjvCamera();view.lookat[:]=(.54,-.14,.14)
    view.distance,view.azimuth,view.elevation=1.03,135,-28
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    frames=np.unique(np.r_[np.arange(0,boundary,3),boundary-1,
                           boundary+np.arange(0,len(case['live']),3),total-1])
    track_frames=np.array([x[0] for x in case['tracking']])
    assert (np.diff(track_frames)>0).all()
    target=case['row']['target_id']
    path=OUT/f'case_{number:02d}.mp4'
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x720','-r','25/3','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    with mujoco.Renderer(model,height=560,width=960,max_geom=2400) as overview:
        for frame in tqdm(frames,desc=f'Render tracking -> FM video {number}',mininterval=5):
            index=frame-boundary
            source={key:past[key][frame] for key in ('qpos','qvel','ctrl')} if index<0 else case['live'][index]
            for key in ('qpos','qvel','ctrl'):getattr(display,key)[:]=source[key]
            mujoco.mj_forward(model,display)
            prediction=case['tracking'][max(0,int(np.searchsorted(track_frames,frame,side='right')-1))][1]
            truth=display.xpos[model.body(NAMES[target]).id]-camera['base']
            # Render clean top input before adding any diagnostic geometry.
            renderer.update_scene(display,camera='cam_top',scene_option=option)
            top=Image.fromarray(renderer.render()[40:280])
            draw=ImageDraw.Draw(top)
            for point,color in ((truth,'#55ee88'),(prediction,'#ee66dd')):
                x,y=project(point,camera);draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
            if frame==0:
                x,y=case['uv'];draw.ellipse((x-3,y-3,x+3,y+3),fill='orange')
            overview.update_scene(display,camera=view,scene_option=option)
            add_marker(overview.scene,prediction+camera['base'],[.95,.2,.65,1.])
            add_marker(overview.scene,truth+camera['base'],[.2,.9,.5,1.],radius=.003)
            if index>=0:
                path_overlay(overview.scene,[x['ee'] for x in case['live'][:index+1]],(64,238,145),.0018)
                plan=case['plans'][min(index//15,19)]
                if index<300:
                    path_overlay(overview.scene,np.vstack([plan['anchor'],plan['anchor']+plan['delta'][:,:3]]),(255,180,65),.0013)
            image=Image.new('RGB',(1280,720),(19,25,34));image.paste(Image.fromarray(overview.render()),(0,80))
            image.paste(top,(960,80));draw=ImageDraw.Draw(image)
            phase='Recorded shuffle + expert cover opening' if index<0 else case['live'][index]['phase']
            if index==0:phase='HOME RESET -> FM reach/grip'
            draw.text((12,8),f'Point tracker -> joint-input FM | {case["row"]["evaluation_group"]} {case["row"]["path"]} | {NAMES[target]}',font=font,fill='white')
            draw.text((12,36),f'{phase} | source t={frame/25:.2f}s'+(f' | FM/probe t={index/25:.2f}s' if index>=0 else ''),font=small,fill='white')
            draw.text((972,340),f'Tracking error: {np.linalg.norm(prediction-truth)*1000:.1f}mm',font=small,fill='white')
            draw.text((972,375),'GT XYZ (base m)',font=small,fill='#55ee88')
            draw.text((972,400),' '.join(f'{x:.3f}' for x in truth),font=small,fill='white')
            draw.text((972,435),'Predicted XYZ (base m)',font=small,fill='#ee66dd')
            draw.text((972,460),' '.join(f'{x:.3f}' for x in prediction),font=small,fill='white')
            if index>=0:
                distance=np.linalg.norm(case['live'][index]['ee']-(truth+camera['base']))*1000
                draw.text((972,505),f'EE -> target: {distance:.1f}mm',font=small,fill='white')
                draw.text((972,535),f'Jaw contacts: {case["live"][index]["contacts"][target]}/2',font=small,fill='white')
            draw.text((12,650),'Magenta: tracker center | Green: GT center / actual EE path | Orange: FM predicted action chunk',font=small,fill='white')
            draw.text((12,677),'Recorded cover opening -> robot home reset -> live visual FM reach/grip (H25/K15, flow10) -> separate IK lift probe',font=small,fill='white')
            encoder.stdin.write(np.asarray(image).tobytes())
            if frame in (0,boundary,total-1):image.save(OUT/f'case_{number:02d}_{frame:04d}.jpg')
    encoder.stdin.close();assert encoder.wait()==0
    with av.open(str(path)) as video:decoded=sum(1 for _ in video.decode(video=0))
    assert decoded==len(frames)
    record['video_frames']=decoded


def main():
    global OUT, TRACKER, FM
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tracker',type=Path,default=TRACKER)
    parser.add_argument('--reach',type=Path,default=FM)
    args=parser.parse_args();OUT=args.output;TRACKER=args.tracker;FM=args.reach
    readme=('# Causal tracking -> reach/grip evaluation\n\nStatus: running.\n'
            'Inference only; selected tracker/FM and fixed held-out cases in config.json. '
            'Recorded shuffle/expert cover opening -> home reset -> live tracker/FM reach-grip '
            '(H25/K15,flow10,seed0,300actions) -> separate IK lift probe100actions. '
            'No GT goals or forced jaw command. Exact data/model/normalization in parent assets.\n')
    torch.set_num_threads(4);torch.manual_seed(101)
    torch.backends.cuda.matmul.allow_tf32=True
    OUT.mkdir(exist_ok=False);(OUT/'README.md').write_text(readme);shutil.copyfile(DATA/'scene.xml',OUT/'scene.xml')
    model=visual_model(DATA/'scene.xml',object_bodies=NAMES)
    model.vis.global_.offwidth,model.vis.global_.offheight=1280,720
    base_model=mujoco.MjModel.from_xml_path('/workspace/robotics_assets/datasets/reach_voxel_grid/scene.xml')
    for i in range(base_model.nbody):
        name=base_model.body(i).name
        if name:
            for field in ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat'):
                np.testing.assert_array_equal(getattr(model,field)[model.body(name).id],getattr(base_model,field)[i])
    for field in ('dof_damping','dof_frictionloss','dof_armature'):
        np.testing.assert_array_equal(getattr(model,field)[:base_model.nv],getattr(base_model,field))
    for field in ('actuator_ctrlrange','actuator_forcerange','actuator_gainprm','actuator_biasprm','eq_data'):
        np.testing.assert_array_equal(getattr(model,field),getattr(base_model,field))
    qadr=np.array([model.jnt_qposadr[model.joint(n).id] for n in (*ARM_JOINTS,'right_jaw_slide_joint')])
    dofs=np.array([model.jnt_dofadr[model.joint(n).id] for n in ARM_JOINTS])
    physics=(model,qadr,dofs,model.site('grasp_site').id)
    tracker_state=torch.load(TRACKER,map_location='cpu',weights_only=False,mmap=True)
    tracker=PointXYZ(**tracker_state['model_config']).cuda().eval()
    tracker.load_state_dict(tracker_state['model'],strict=True)
    policy_state=torch.load(FM,map_location='cpu',weights_only=False,mmap=True)
    policy=ReachPolicy().cuda().eval();policy.load_state_dict(policy_state['model'],strict=True)
    camera=top_camera(DATA);base=camera['base']
    np.testing.assert_allclose(policy_state['config']['base_origin_world_m'],base,rtol=0,atol=1e-12)
    splits=subset(DATA)
    rows=[next(row for row in splits[group] if row['target_id']==target)
          for group,target in (('composition',0),('spatial',1),('combined',2))]
    config=dict(status='running',tracker_checkpoint=str(TRACKER),tracker_step=tracker_state['step'],
        FM_checkpoint=str(FM),FM_step=policy_state['step'],checkpoint_sha256=dict(tracker=sha(TRACKER),FM=sha(FM)),
        source_data=str(DATA),cases=rows,training='none; inference only',views=['top'],query_frame=0,
        query='radius3 2D initial dot; no ID/GT XYZ/masks as model input',past_frames=30,
        image_stride=3,tracker_goal_update_actions=15,live_goal_updates=True,horizon=25,execute=15,
        flow_steps=10,noise_seed=0,FM_actions=300,IK_probe_actions=100,IK_probe_lift_m=.08,
        forced_gripper=False,probe_gripper='hold last FM gripper command',policy_hz=25,servo_hz=50,physics_hz=500,
        initial_state='recorded t_obj object state + exact e86 settled robot home',
        prelude='recorded shuffle and expert cover removal; not learned cover selection/control',
        action_representation='measured chunk-start deltaXYZ3 + relative rotation6D + absolute gripper1',
        state_representation='five measured arm joint radians + right jaw metres',base_origin_world_m=base.tolist(),
        source_sha256=sha(Path(__file__)))
    write_json(OUT/'config.json',config)
    del tracker_state,policy_state
    option=mujoco.MjvOption();option.geomgroup[3]=0
    results=[]
    with mujoco.Renderer(model,height=320,width=320,max_geom=1000) as renderer:
        for number,row in enumerate(tqdm(rows,desc='Three held-out end-to-end cases',mininterval=5),1):
            write_json(OUT/'progress.json',dict(status='rollout',case=number,total=3))
            case=prepare(row,model,camera)
            rollout(tracker,policy,physics,case,renderer,option,base)
            record=finish(physics,case,number,base)
            write_json(OUT/'progress.json',dict(status='rendering',case=number,total=3))
            render(model,case,record,number,renderer,option,camera)
            results.append(record);write_json(OUT/'metrics.json',results);print(json.dumps(record),flush=True)
    config['status']='complete';write_json(OUT/'config.json',config)
    write_json(OUT/'progress.json',dict(status='complete',cases=3))
    text=readme.replace('Status: running.','Status: complete; all3videos fully decoded.')
    text+='\nActual results: metrics.json. Secure lift successes: '+str(sum(r['grasp_retention_success'] for r in results))+'/3.\n'
    (OUT/'README.md').write_text(text)



if __name__=='__main__':
    main()
