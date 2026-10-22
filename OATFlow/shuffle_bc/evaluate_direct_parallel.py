"""Full E115 test with independent single-case processes and no video export."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import shutil
import time

import numpy as np
import mujoco
import torch
from tqdm import tqdm

from OATFlow.shuffle_bc.data import DATA, splits
from OATFlow.shuffle_bc.direct_model import DirectPolicy
from OATFlow.shuffle_bc.evaluate_direct import counts, rollout
from OATFlow.shuffle_bc.train_direct import frozen_models
from OATFlow.shuffle_bc.truncate_direct import truncate
from OATFlow.simulation.io import write_json
from OATFlow.tracking.linear_probe import fingerprint
from OATFlow.tracking.mask_video import digest

ROOT = Path('/workspace/robotics_assets/experiments/e115_direct')


def worker(number, out, tasks, messages, start, frozen, frame_budget):
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    saved = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
    policy = DirectPolicy(); policy.load_state_dict(saved['model'], strict=True)
    policy.eval().requires_grad_(False).cuda()
    tracker, camera = frozen_models()
    assert dict(TCOW=fingerprint(tracker), auxiliary_ViT=fingerprint(camera)) == frozen
    directory = out/f'worker_{number}'
    title = f'E115 | epoch {saved["epoch"]} | full test'
    del saved
    messages.put(('ready',number,None)); start.wait()
    while True:
        row = tasks.get()
        if row is None: break
        messages.put(('start',number,row['path']))
        tick = time.perf_counter()
        result = rollout(row,policy,tracker,camera,directory,title,video=False,progress=dict(worker=number),frame_budget=frame_budget)
        result.update(elapsed_case_seconds=time.perf_counter()-tick,
                      trace=f'worker_{number}/'+row['path'].replace('/','_')+'.npz',frame_budget=frame_budget)
        messages.put(('result',number,result))
    assert dict(TCOW=fingerprint(tracker), auxiliary_ViT=fingerprint(camera)) == frozen
    messages.put(('done',number,None))


def save_results(out, results):
    write_json(out/'results.tmp',results)
    (out/'results.tmp').replace(out/'results.json')


def verify_trace(out, result, row, budget):
    name = result['path'].replace('/','_')+'.npz'
    records=json.loads((ROOT/'test_k10_2/summary.json').read_text())['results']
    old=next(r for r in records if r['path']==result['path'])
    model=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
    expected,reference=truncate(ROOT/'test_k10_2'/name,old,row,budget,model)
    with np.load(out/result['trace']) as actual:
        assert set(actual.files)==set(reference)
        for key in actual.files:
            np.testing.assert_array_equal(actual[key],reference[key],err_msg=result['path']+':'+key)
    for key in ('success','reason','frames','cover_deposited','other_cover_shift_m','lifted_bilateral_streak','max_target_lift_m','max_cover_lift_m'):
        assert result[key]==expected[key],key
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4,choices=(2,4))
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--frame-budget',type=int,default=900,choices=(400,900))
    parser.add_argument('--import-results',type=Path,help='Committed parallel test to import and re-score at this horizon')
    args = parser.parse_args()
    torch.set_num_threads(2)
    reference = json.loads((ROOT/'test_k10_2/config.json').read_text())
    source = ROOT/'train_2/best.pt'
    saved = torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    assert (saved['epoch'],saved['step']) == (32,29420)
    assert digest(source) == reference['checkpoint_sha256']
    if args.preflight:
        rows = reference['cases']
    else:
        groups = splits(); names = ('composition','spatial','combined')
        assert [len(groups[n]) for n in names] == [1024,256,128]
        rows = [groups[g][i] for i in range(1024) for g in names if i<len(groups[g])]
        assert len(rows)==len({r['path'] for r in rows})==1408
        assert all(sum(r['target_id']==i for r in rows)==352 for i in range(4))
    protocol_keys = ('H','K','flow_steps','noise_seed','frame_budget','policy_hz','history','prompt',
                     'numeric_boundaries','success','frozen_sha256')
    sources = [Path(__file__),Path('OATFlow/shuffle_bc/truncate_direct.py'),*[Path(name) for name in reference['source_sha256']]]
    config = dict(status='running',workers=args.workers,preflight=args.preflight,video=False,
        checkpoint=str(source),checkpoint_sha256=digest(source),checkpoint_epoch=32,checkpoint_step=29420,
        cases=rows,training=saved['config'],protocol={k:reference[k] for k in protocol_keys},
        scene_sha256=digest(DATA/'scene.xml'),manifest_sha256=digest(DATA/'manifest.jsonl'),
        source_sha256={str(p):digest(p) for p in sources},
        parallelism='independent processes; single-case tensor shapes and noise0 reset per episode; original camera schedules')
    config['protocol']['frame_budget']=args.frame_budget
    config['import_results']=str(args.import_results) if args.import_results else None
    del saved
    out = args.output
    if args.resume:
        previous = json.loads((out/'config.json').read_text())
        assert previous['status']!='complete'
        for key in ('workers','preflight','checkpoint_sha256','cases','protocol','scene_sha256','manifest_sha256','source_sha256','import_results'):
            assert previous[key]==config[key],key
        assert digest(out/'checkpoint.pt')==config['checkpoint_sha256']
        results=json.loads((out/'results.json').read_text())
    else:
        out.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(source,out/'checkpoint.pt')
        readme='docs/research/e115_400_README.md' if args.frame_budget==400 else 'docs/research/e115_parallel_README.md'
        shutil.copyfile(readme,out/'README.md')
        results=[]
        if not args.preflight:
            assert args.import_results is not None
            old_config=json.loads((args.import_results/'config.json').read_text())
            assert old_config['checkpoint_sha256']==config['checkpoint_sha256']
            for key in protocol_keys:
                if key!='frame_budget':assert old_config['protocol'][key]==config['protocol'][key],key
            assert args.frame_budget<=old_config['protocol']['frame_budget']
            assert old_config['scene_sha256']==config['scene_sha256']
            assert old_config['manifest_sha256']==config['manifest_sha256']
            for path,expected in old_config['source_sha256'].items():
                if Path(path).name not in ('evaluate_direct.py','evaluate_direct_parallel.py'):
                    assert digest(Path(path))==expected,path
            prior=json.loads((args.import_results/'results.json').read_text())
            by_path={r['path']:r for r in rows}
            physics=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
            imported=out/'imported';imported.mkdir()
            (imported/'README.md').write_text(f'Committed traces from {args.import_results}, clipped and physically re-scored at {args.frame_budget}frames. Full config/provenance in ../config.json. No videos.\n')
            for old in tqdm(prior,desc='Re-score committed prefixes',mininterval=5):
                name=old['path'].replace('/','_')+'.npz'
                result,arrays=truncate(args.import_results/old['trace'],old,by_path[old['path']],args.frame_budget,physics)
                np.savez_compressed(imported/name,**arrays)
                results.append(dict(result,trace='imported/'+name,imported_from=str(args.import_results)))
        save_results(out,results)
    assert len({r['path'] for r in results})==len(results)
    assert {r['path'] for r in results}<={r['path'] for r in rows}
    write_json(out/'config.json',config)
    for path in sources:
        dest=out/'source'/path.resolve().relative_to(Path.cwd())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    (out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    done={r['path'] for r in results};pending=[r for r in rows if r['path'] not in done]
    imported_count=len(results)
    context=mp.get_context('spawn');tasks=context.Queue();messages=context.Queue();start=context.Event()
    for row in pending:tasks.put(row)
    for _ in range(args.workers):tasks.put(None)
    processes=[]
    for i in range(args.workers):
        folder=out/f'worker_{i}';folder.mkdir(exist_ok=True)
        (folder/'README.md').write_text(f'E115 full-test worker{i}; no videos. Same epoch32/step29420, H25/K10/Euler10/noise0/{args.frame_budget}frames. Full training/test configuration in ../config.json; aggregate completion in ../results.json.\n')
        process=context.Process(target=worker,args=(i,out,tasks,messages,start,config['protocol']['frozen_sha256'],args.frame_budget),daemon=True)
        process.start();processes.append(process)
    ready=set();finished=set();active={};compute_started=None;started=time.perf_counter()
    bar=tqdm(total=len(rows),initial=len(results),desc='Parallel full test',mininterval=5)
    try:
        while len(finished)<args.workers:
            for process in processes:
                if process.exitcode not in (None,0):raise RuntimeError(f'worker {process.pid} exited {process.exitcode}')
            try:
                event,number,value=messages.get(timeout=1)
            except Empty:
                continue
            if event=='ready':
                ready.add(number)
                print('READY',number,flush=True)
                if len(ready)==args.workers:
                    compute_started=time.perf_counter();start.set()
            elif event=='start':active[number]=value
            elif event=='done':finished.add(number);active.pop(number,None)
            elif event=='result':
                if args.preflight:value['trace_matches_reference']=verify_trace(out,value,next(r for r in rows if r['path']==value['path']),args.frame_budget)
                assert value['path'] not in done
                results.append(value);done.add(value['path']);active.pop(number,None)
                save_results(out,results)
                with (out/'results.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
                bar.update(1)
                print('RESULT',json.dumps(dict(worker=number,**value)),flush=True)
            elapsed=time.perf_counter()-compute_started if compute_started is not None else 0.
            completed=len(results)-imported_count
            progress=dict(status='running' if compute_started else 'loading workers',**counts(results),
                cases_done=len(results),cases_total=len(rows),frame_budget=args.frame_budget,workers_ready=len(ready),active_cases=active,
                elapsed_seconds=time.perf_counter()-started,compute_seconds=elapsed,
                eta_seconds=elapsed/completed*(len(rows)-len(results)) if completed else None)
            write_json(out/'progress.json',progress)
        for process in processes:
            process.join();assert process.exitcode==0
        assert {r['path'] for r in results}=={r['path'] for r in rows}
        assert digest(source)==config['checkpoint_sha256']
        summary=dict(status='complete',**counts(results),workers=args.workers,frame_budget=args.frame_budget,imported_cases=imported_count,
            elapsed_seconds=time.perf_counter()-started,compute_seconds=time.perf_counter()-compute_started,
            frozen_unchanged=True,checkpoint_unchanged=True,video_exported=False,results=results)
        write_json(out/'summary.json',summary);write_json(out/'progress.json',{k:v for k,v in summary.items() if k!='results'})
        config['status']='complete';write_json(out/'config.json',config)
        print('COMPLETE',json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)
    finally:
        bar.close()
        for process in processes:
            if process.is_alive():process.terminate()
            process.join()
        tasks.cancel_join_thread();tasks.close()
        messages.cancel_join_thread();messages.close()


if __name__=='__main__':main()
