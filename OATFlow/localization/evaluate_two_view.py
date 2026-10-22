"""Evaluate a finished two-view XYZ checkpoint without further optimizer updates."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.localization.model import WEIGHTS, state_hash
from OATFlow.localization.train_two_view import DATASET, evaluate, save_examples, sha
from OATFlow.localization.two_view import TwoViewXYZ
from OATFlow.simulation.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); out = args.run
    torch.set_num_threads(4); torch.manual_seed(113)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    config = json.loads((out / 'config.json').read_text())
    history = json.loads((out / 'history.json').read_text())
    total_epochs = config['epochs']
    assert len(history) == total_epochs and history[-1]['epoch'] == total_epochs
    checkpoint_hashes = {p.name: sha(p) for p in (out / 'best.pt', out / 'latest.pt')}
    saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    net = TwoViewXYZ().eval().cuda()
    missing, unexpected = net.load_state_dict(saved['model'], strict=False)
    assert not unexpected and all(k.startswith('encoder.') for k in missing)
    before = state_hash(net.encoder); assert before == config['encoder_state_sha256']
    with np.load(DATASET / 'labels.npz') as z:
        labels = z['xyz_base_m'].copy(); voxels = z['voxel_id'].copy()
    with np.load(out / 'split.npz') as z:
        train, val, test = (z[k].copy() for k in ('train', 'val', 'test'))
    mean, std = saved['mean'], saved['std']; n = len(labels)
    rgb = np.load(DATASET / 'rgb.npy', mmap_mode='r')[:n]
    packed = np.load(DATASET / 'masks_packed.npy', mmap_mode='r')[:n]
    masks = torch.empty((n, 2, 320, 320), dtype=torch.bool)
    for start in tqdm(range(0, n, 256), desc='Load masks for verification', mininterval=5):
        masks[start:start+256] = torch.from_numpy(np.unpackbits(packed[start:start+256], axis=-1, count=320,
                                                               bitorder='little').astype(bool))
    features = torch.empty((n, 2, 400, 768), dtype=torch.float16)
    started = time.perf_counter()
    with torch.no_grad():
        for start in tqdm(range(0, n, 16), desc='Rebuild verification cache to RAM', mininterval=5):
            images = torch.from_numpy(np.array(rgb[start:start+16], copy=True)).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                value = net.encode(images)
            features[start:start+len(value)] = value.cpu()
            if start % 256 == 0:
                write_json(out / 'progress.json', dict(status='verifying completed training', epochs=total_epochs,
                    cached_samples=start+len(value), samples=n, elapsed_s=time.perf_counter()-started))
    ids = test[:8]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        images = torch.from_numpy(np.array(rgb[ids], copy=True)).cuda()
        raw_encoder_dtype = str(net.encoder(images.flatten(0, 1)).dtype)
        online = net(images, masks[ids].cuda())
        cached = net.from_features(features[ids].cuda(), masks[ids].cuda())
    parity = np.linalg.norm((online.cpu().numpy()-cached.cpu().numpy())*std, axis=-1)*1000
    assert float(parity.max()) < 1., 'Online/cache XYZ discrepancy exceeds 1mm.'
    test_metrics, test_prediction = evaluate(net, features, masks, labels, test, mean, std, 'Test best')
    train_metrics, train_prediction = evaluate(net, features, masks, labels, train, mean, std, 'Train best')
    val_metrics, _ = evaluate(net, features, masks, labels, val, mean, std, 'Validation best')
    with np.load(out / 'test_predictions.npz') as z:
        np.testing.assert_array_equal(z['indices'], test)
        previous_difference = np.linalg.norm(test_prediction-z['prediction_base_m'], axis=-1)*1000
    assert float(previous_difference.max()) < 1., 'Replayed test differs from original cached test.'
    assert before == state_hash(net.encoder) and sha(WEIGHTS) == config['encoder_file_sha256']
    assert all(not p.requires_grad and p.grad is None for p in net.encoder.parameters())
    assert checkpoint_hashes == {p.name: sha(p) for p in (out / 'best.pt', out / 'latest.pt')}
    save_examples(out, rgb, masks, labels, test, test_prediction, 'test')
    save_examples(out, rgb, masks, labels, train, train_prediction, 'train')
    result = dict(status='complete', epochs=total_epochs, steps=history[-1]['step'], samples=n,
        split_counts=config['split_counts'], best_epoch=saved['epoch'], best_step=saved['step'],
        train=train_metrics, validation=val_metrics, test=test_metrics,
        training_and_validation_seconds=history[-1]['elapsed_s'], cache_seconds=config['cache_seconds'],
        render_seconds=config['dataset_config']['render_seconds'], verification_seconds=time.perf_counter()-started,
        encoder_frozen_unchanged=True, checkpoint_weights_unchanged_during_verification=True,
        checkpoint_sha256=checkpoint_hashes, online_cache_mean_difference_mm=float(parity.mean()),
        online_cache_max_difference_mm=float(parity.max()), original_test_replay_max_difference_mm=float(previous_difference.max()),
        raw_encoder_dtype_under_autocast=raw_encoder_dtype, decoder_input_dtype='torch.float16 for online and cache',
        precision_fix='Cast frozen ViT outputs to FP16 in encode(), matching the features used throughout training.',
        verification_source_sha256={str(p): sha(p) for p in (Path(__file__), Path('OATFlow/localization/two_view.py'))},
        cache_persisted=False, optimizer_updates_during_evaluation=0, action_control=False,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(out / 'summary.json', result); write_json(out / 'progress.json', result)
    config.update(status='complete', inference_precision='FP16 frozen-ViT output, identical to training cache')
    write_json(out / 'config.json', config)
    (out / 'source_inference_two_view.py').write_text(Path('OATFlow/localization/two_view.py').read_text())
    (out / 'source_evaluate_two_view.py').write_text(Path(__file__).read_text())
    with (out / 'README.md').open('a') as f:
        f.write(f'\nCompleted{total_epochs}epochs; best epoch{saved["epoch"]}, test{test_metrics["mean_mm"]:.3f}mm. '
            'Frozen ViT output is FP16 consistently, matching all training features. No checkpoint weights '
            'or optimizer updates changed during evaluation. '
            f'Online/cache max discrepancy{parity.max():.4f}mm; original test replay max discrepancy '
            f'{previous_difference.max():.4f}mm. Full metrics in summary.json. RAM cache released on exit.\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
