"""Export XYZ learning/localization/generalization diagnostics for e108."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root
    train = root / 'train'; test = root / 'test_xyz'
    learning = [json.loads(line) for line in (train / 'validation.jsonl').read_text().splitlines()]
    best = json.loads((train / 'best_validation.json').read_text())
    heldout = json.loads((test / 'predictions.json').read_text())
    summary = json.loads((test / 'summary.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
        'axes.labelsize': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout='constrained')
    colors = {'visible': '#0072B2', 'shuffle': '#E69F00', 'post_shuffle': '#CC79A7',
              'exposed': '#009E73', 'lift': '#D55E00'}
    for phase, color in colors.items():
        axes[0].plot([r['epoch'] for r in learning], [r['validation']['by_phase_mm'][phase] for r in learning],
                     color=color, linewidth=1.1, label=phase.replace('_', ' '))
    axes[0].plot([r['epoch'] for r in learning], [r['validation']['macro_phase_mm'] for r in learning],
                 color='#222222', linewidth=2, label='phase macro')
    axes[0].set(title='A  Validation learning curve', xlabel='Training epoch', ylabel='Mean XYZ error (mm)')
    axes[0].legend(frameon=False, ncol=2, fontsize=7)
    points = [r for r in best['records'] if r['phase'] == 'post_shuffle']
    truth = np.array([r['truth_base_m'] for r in points])[:, :2] * 1000
    predicted = np.array([r['prediction_base_m'] for r in points])[:, :2] * 1000
    for a, b in zip(truth, predicted): axes[1].plot([a[0], b[0]], [a[1], b[1]], color='#BBBBBB', linewidth=.6, zorder=1)
    axes[1].scatter(*truth.T, s=22, facecolors='none', edgecolors='#0072B2', label='Ground truth', zorder=3)
    axes[1].scatter(*predicted.T, s=18, marker='x', color='#D55E00', linewidths=.9, label='Prediction', zorder=4)
    axes[1].set(title='B  Post-shuffle target XY (validation)', xlabel='Base X (mm)', ylabel='Base Y (mm)')
    axes[1].set_aspect('equal', adjustable='datalim'); axes[1].legend(frameon=False, fontsize=8)
    for name, label, color in [('composition', 'Unseen order', '#0072B2'),
            ('spatial', 'Unseen layout', '#D55E00'), ('combined', 'Both unseen', '#009E73')]:
        errors = np.sort([r['error_mm'] for r in heldout if r['group'] == name])
        axes[2].step(errors, np.arange(1, len(errors) + 1) / len(errors) * 100,
                     where='post', color=color, linewidth=1.6, label=label)
    axes[2].axvline(10, color='#777777', linestyle=':', linewidth=1)
    axes[2].set(title='C  Held-out XYZ accuracy', xlabel='XYZ error threshold (mm)',
                ylabel='Endpoints within threshold (%)', ylim=(0, 102), xlim=(0, None))
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.18, linewidth=.6)
    for suffix in ('png', 'svg', 'pdf'): fig.savefig(root / f'xyz_diagnostics.{suffix}', dpi=180)
    plt.close(fig)
    (root / 'xyz_diagnostics.md').write_text(
        '# XYZ localization diagnostics\n\nFrozen original TCOW + frozen ImageNet ViT-B/16 using camera left as a wrist surrogate; '
        'trainable shared context decoder and XYZ readout. Part1 training only, selected by validation macro phase mean. '
        'PanelB shows all32post-shuffle validation query endpoints. '
        f'PanelC uses {summary["episodes"]}held-out episodes /{summary["endpoints"]}fixed phase endpoints, '
        f'checkpoint epoch{summary["checkpoint_epoch"]}/step{summary["checkpoint_step"]}. '
        'Recorded expert trajectories; these figures evaluate object-center predictions, not learned robot-control success. '
        'Sources: train/validation.jsonl,train/best_validation.json,test_xyz/predictions.json and summary.json.\n')
    print('Exported xyz_diagnostics.png/.svg/.pdf and caption.', flush=True)


if __name__ == '__main__': main()
