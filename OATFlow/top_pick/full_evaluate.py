"""Evaluate E122 final23520 on the original1408held-out pick episodes."""
import argparse
from collections import Counter
import json
from pathlib import Path
from OATFlow.top_pick.cloud_evaluate import main
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.full_train import ROOT
from OATFlow.top_pick.patch_fusion import PatchFusionPolicy


def compare(k):
    assert k in (8,15)
    runs={value:ROOT/f'test_k{value}' for value in (k,10)}
    configs={k:json.loads((p/'config.json').read_text()) for k,p in runs.items()}
    for key in ('checkpoint_sha256','checkpoint_step','test_manifest_sha256','cases','H','flow_steps',
                'noise_seed','frame_budget','policy_hz','servo_hz','physics_hz','success'):
        assert configs[k][key]==configs[10][key],key
    assert configs[k]['status']==configs[10]['status']=='complete'
    results={value:[json.loads(s) for s in (p/'results.jsonl').read_text().splitlines()] for value,p in runs.items()}
    assert [r['path'] for r in results[k]]==[r['path'] for r in results[10]]
    assert len(results[k])==len({r['path'] for r in results[k]})==1408
    summaries={value:json.loads((p/'summary.json').read_text()) for value,p in runs.items()}
    paired=Counter('both_success' if a['success'] and b['success'] else
                   f'K{k}_only' if a['success'] else 'K10_only' if b['success'] else 'both_failure'
                   for a,b in zip(results[k],results[10]))
    comparison=dict(status='complete',checkpoint_step=23520,cases=1408,K10=summaries[10],
        delta_percentage_points=100*(summaries[k]['success_rate']-summaries[10]['success_rate']),
        paired_outcomes=dict(paired),same_checkpoint_and_manifest=True,
        noise='same seed0 and generator protocol;different cadence samples a different number of plans',
        changed_setting=f'execute{k}instead of10actions per chunk;H25/Euler10/frame budget300 unchanged')
    comparison[f'K{k}']=summaries[k]
    write(runs[k]/'comparison_k10.json',comparison)
    print(json.dumps(comparison),flush=True)
    note=f'\nCOMPLETE: K{k} {summaries[k]["successes"]}/1408 ({100*summaries[k]["success_rate"]:.3f}%) versus K10 {summaries[10]["successes"]}/1408 ({100*summaries[10]["success_rate"]:.3f}%),delta{comparison["delta_percentage_points"]:+.3f}percentage points. Same checkpoint/manifest/protocol verified. Paired outcomes:{dict(paired)}. Exact comparison:test_k{k}/comparison_k10.json.\n'
    local=Path(f'docs/research/e122_k{k}_README.md');text=local.read_text()
    status=next(line for line in text.splitlines() if line.startswith('Status:'))
    text=text.replace(status,f'Status: COMPLETE — all1408K{k}episodes evaluated.',1)+note
    local.write_text(text);(runs[k]/'README.md').write_text(text)
    with Path('docs/research/REPORT.md').open('a') as f:f.write(f'\nE122 K{k} versus K10:\n'+note)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--k',type=int,choices=(8,10,15),default=10)
    args=parser.parse_args()
    main(args.resume,root=ROOT,policy_class=PatchFusionPolicy,
         readme=f'docs/research/e122_k{args.k}_README.md' if args.k!=10 else 'docs/research/e122_test_README.md',
         expected_cases=1408,k=args.k)
    if args.k!=10:compare(args.k)
