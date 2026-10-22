"""Probe cover identity from e114 target masks on every held-out episode."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.tracking.data import sample_indices
from OATFlow.tracking.mask import TargetMaskTCOW
from OATFlow.tracking.mask_data import read_segment, splits
from OATFlow.tracking.mask_video import digest
from OATFlow.tracking.utils import write_json


DATA = Path('/workspace/robotics_assets/datasets/shuffle_pick_multiview')
TRAIN = Path('/workspace/robotics_assets/experiments/e114_mask/train_3')
GROUPS = ('composition', 'spatial', 'combined')


def context(ref):
    segment = read_segment(DATA, ref, True)
    with np.load(DATA / ref / 'masks.npz') as labels:
        index = segment['raw_index']
        segment['covers'] = labels['amodal_packed'][index, 0, 4:6, 40:280].copy()
        segment['container'] = labels['container_ids'][index].copy()
    return segment


def cover_assignment(mask, covers):
    area = int(mask.sum())
    overlap = (covers & mask[None]).sum((-1, -2))
    winner = int(overlap.argmax())
    # Require majority support and a unique cover; no nearest-cover fallback.
    confident = bool(area > 0 and overlap[winner] * 2 > area and overlap[winner] > overlap[1-winner])
    return (winner + 5 if confident else 0), overlap.tolist(), (float(overlap[winner]/area) if area else 0.)


def score(records):
    def count(rows):
        n = len(rows)
        return dict(count=n, correct_cover=sum(r['correct_cover'] for r in rows),
            cover_accuracy=sum(r['correct_cover'] for r in rows)/n if n else None,
            wrong_cover=sum(r['predicted_cover_id'] not in (0, r['true_cover_id']) for r in rows),
            unresolved=sum(r['predicted_cover_id'] == 0 for r in rows),
            GT_mask_cover_assignment_correct=sum(r['oracle_cover_id'] == r['true_cover_id'] for r in rows),
            object_mask_binding_correct=sum(r['correct_object'] for r in rows))
    return dict(all=count(records), post_shuffle=count([r for r in records if r['phase']=='post_shuffle']),
        shuffle_probes=count([r for r in records if r['phase']=='shuffle']),
        by_group={group: {phase: count([r for r in records if r['group']==group and r['phase']==phase])
                          for phase in ('post_shuffle', 'shuffle')} for group in GROUPS},
        previous160_post_shuffle=count([r for r in records if r['previous_subset'] and r['phase']=='post_shuffle']))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile('docs/research/e114_cover_README.md', out / 'README.md')
    checkpoint = TRAIN / 'best.pt'
    checkpoint_hash = digest(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    model = TargetMaskTCOW()
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False).cuda()
    rows = [json.loads(line) for line in (DATA / 'manifest.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['evaluation_group'] in GROUPS]
    previous = {r['path'] for group in GROUPS for r in splits(DATA)[group]}
    rows.sort(key=lambda row: (row['path'] not in previous, row['evaluation_group'], row['path']))
    assert len(rows) == 1408 and len(previous) == 160
    config = dict(status='running', checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        epoch=saved['epoch'], step=saved['step'], episodes=len(rows),
        group_counts={g: sum(r['evaluation_group']==g for r in rows) for g in GROUPS},
        previous_subset=160, threshold=.5, frames_per_prediction=30, cache_stride=3,
        cover_rule='unique highest binary-mask overlap with GT amodal cover; >50% of predicted area inside winner, otherwise unresolved',
        GT='physical container_ids, independently checked against selected_cover after shuffle',
        GT_cover_masks='evaluation assignment only; never model input',
        probes='post-shuffle final context frame + midpoint of each shuffle run; not every hidden frame',
        inputs='top RGB causal prefix + frame0 visible mask only', action_control_test=False,
        source_sha256=digest(Path(__file__)))
    write_json(out / 'config.json', config)
    segments = {}
    refs = sorted({r['context_path'] for r in rows})
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(context, ref): ref for ref in refs}
        for job in tqdm(as_completed(jobs), total=len(jobs), desc='Cache held-out context RGB/masks', mininterval=5):
            segments[jobs[job]] = job.result()
    points = []
    for phase in ('post_shuffle', 'shuffle'):
        for row in rows:
            segment = segments[row['context_path']]
            if phase == 'post_shuffle':
                ends = [len(segment['raw_index'])-1]
            else:
                ids = np.flatnonzero(segment['phase']=='shuffle')
                runs = np.split(ids, np.flatnonzero(np.diff(ids)>1)+1)
                ends = [int(run[len(run)//2]) for run in runs if len(run)]
            points.extend((row, end, phase) for end in ends)
    records, predictions = [], []
    replay = {(r['path'], r['raw_frame']): r['iou'] for group in GROUPS
              for r in json.loads((TRAIN/f'test_{group}.json').read_text())['records']
              if r['phase']=='post_shuffle'}
    differences = []
    for start in tqdm(range(0, len(points), 4), desc='Cover identity probes', mininterval=5):
        chosen = points[start:start+4]
        rgb, query = [], []
        for row, end, _ in chosen:
            segment = segments[row['context_path']]
            index = sample_indices(end, 30)
            assert segment['raw_index'][index].max() <= segment['raw_index'][end]
            rgb.append(segment['rgb'][index].permute(1, 0, 2, 3))
            prompt = torch.zeros(1, 30, 240, 320, dtype=torch.bool)
            prompt[0, 0] = torch.from_numpy(segment['prompt'][row['target_id']])
            query.append(prompt)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(torch.stack(rgb).cuda(), torch.stack(query).cuda())[:, -1]
        assert torch.isfinite(logits).all()
        binary = (logits > 0).cpu().numpy()
        predictions.append(np.packbits(binary, axis=-1, bitorder='little'))
        for (row, end, phase), predicted in zip(chosen, binary):
            segment = segments[row['context_path']]
            target = row['target_id']
            assert segment['hidden'][end, target]
            truth = np.unpackbits(segment['packed'][end], axis=-1, count=320, bitorder='little').astype(bool)
            covers = np.unpackbits(segment['covers'][end], axis=-1, count=320, bitorder='little').astype(bool)
            true_cover = int(segment['container'][end, target])
            assert true_cover in (5, 6)
            if phase == 'post_shuffle':
                assert true_cover == {'cover_a': 5, 'cover_b': 6}[row['selected_cover']]
            predicted_cover, overlap, support = cover_assignment(predicted, covers)
            oracle_cover, _, _ = cover_assignment(truth[target], covers)
            intersection = (truth & predicted[None]).sum((-1, -2))
            union = (truth | predicted[None]).sum((-1, -2))
            ious = intersection/union.clip(1)
            winner = int(ious.argmax())
            object_id = winner if ious[winner] > 0 and (ious == ious[winner]).sum()==1 else -1
            raw = int(segment['raw_index'][end])
            record = dict(path=row['path'], group=row['evaluation_group'], phase=phase, raw_frame=raw,
                previous_subset=row['path'] in previous, true_cover_id=true_cover, predicted_cover_id=predicted_cover,
                correct_cover=predicted_cover==true_cover, oracle_cover_id=oracle_cover,
                cover_overlap_pixels=overlap, winner_support_fraction=support, predicted_area=int(predicted.sum()),
                target_iou=float(ious[target]), object_ious=ious.tolist(), target_id=target,
                predicted_object_id=object_id, correct_object=object_id==target)
            records.append(record)
            if phase=='post_shuffle' and (row['path'], raw) in replay:
                differences.append(abs(record['target_iou'] - replay[row['path'], raw]))
        if len(records)%80==0 or start+4 >= len(points):
            result = score(records)
            write_json(out / 'progress.json', dict(status='running', completed=len(records), total=len(points), metrics=result))
            print('PROGRESS', json.dumps(dict(completed=len(records), total=len(points), metrics=result)), flush=True)
    assert len(differences)==160 and max(differences)<1e-7
    assert digest(checkpoint)==checkpoint_hash
    result = dict(status='complete', **score(records), checkpoint_unchanged=True,
                  previous160_post_shuffle_IoU_replay_max_difference=max(differences),
                  all_heldout_episodes=1408, every_hidden_frame_tested=False)
    write_json(out / 'records.json', records)
    write_json(out / 'summary.json', result)
    write_json(out / 'progress.json', result)
    np.savez_compressed(out / 'predictions.npz', packed=np.concatenate(predictions), bitorder='little')
    config['status']='complete'
    write_json(out / 'config.json', config)
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
