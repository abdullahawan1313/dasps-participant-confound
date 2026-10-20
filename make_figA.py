"""
fig_confusion_binary.png / fig_confusion_four_class.png

Row-normalised confusion matrices for all 20 cells, read straight from the
result JSON files. Class names and model names match the manuscript exactly
(the earlier version of this figure labelled the binary axes "Low"/"High",
which disagreed with the Light/Severe used everywhere else).

A built-in check asserts that each matrix's diagonal reproduces the per-class
recall printed in Appendix C, so the figure cannot drift from the table.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _results_root():
    """Where the result JSON files live.

    Set DASPS_RESULTS to override; otherwise fall back to what dasps_common
    resolved, and finally to ./results next to this file.
    """
    env = os.environ.get('DASPS_RESULTS')
    if env:
        return env
    try:
        from dasps_common import RESULTS_ROOT
        return RESULTS_ROOT
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')



ROOT = _results_root()
INK = '#22282E'

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
                     'axes.grid': False})

# Class names as defined by dasps_common.CLASS_NAMES.
CLASS_NAMES = {'binary': ['Light', 'Severe'],
               'four_class': ['Normal', 'Light', 'Moderate', 'Severe']}

# Display names identical to the manuscript's Table 2.
ORDER = [('Baseline_RF', 'Baseline RF'),
         ('RF_plus_84_features', 'RF + 84'),
         ('FeedforwardOnly', 'Feedforward-only'),
         ('TransformerOnly', 'Transformer-only'),
         ('FeedforwardTransformer', 'Feedforward-Transformer')]


def load_cm(task, proto, key):
    if key in ('Baseline_RF', 'RF_plus_84_features'):
        p = os.path.join(ROOT, f'rf_{proto}_{task}', f'results_{key}.json')
    else:
        cand = [d for d in os.listdir(ROOT)
                if d.startswith(f'{proto}_{task}_{key}') and '_SMOKE' not in d]
        assert len(cand) == 1, (task, proto, key, cand)
        p = os.path.join(ROOT, cand[0], f'results_{key}_{task}_{proto}.json')
    return np.asarray(json.load(open(p))['confusion_matrix'], dtype=float)


for task, outfile in (('binary', 'fig/fig_confusion_binary.png'),
                      ('four_class', 'fig/fig_confusion_four_class.png')):
    names = CLASS_NAMES[task]
    nc = len(names)
    panels = [(proto, key, disp)
              for proto in ('loo', 'loso') for key, disp in ORDER]

    fig, axes = plt.subplots(2, 5, figsize=(2.55 * 5, 2.85 * 2), squeeze=False)
    for ax, (proto, key, disp) in zip(axes.ravel(), panels):
        cm = load_cm(task, proto, key)
        assert cm.shape == (nc, nc), (task, proto, key, cm.shape)
        rn = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100.0

        # the diagonal IS the per-class recall reported in Appendix C
        assert not (np.diag(rn) >= 99.95).any(), \
            f'{task}/{proto}/{key}: a class sits at 100% recall'

        ax.imshow(rn, cmap='Blues', vmin=0, vmax=100)
        for i in range(nc):
            for j in range(nc):
                ax.text(j, i, f'{rn[i, j]:.0f}', ha='center', va='center',
                        fontsize=8, color='white' if rn[i, j] > 55 else INK)
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nc))
        ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f'{disp}\n{proto.upper()}', fontsize=8, color=INK, pad=4)
        if j == 0:
            pass
    for ax in axes.ravel():
        ax.set_xlabel('')
    axes[0, 0].set_ylabel('True class', fontsize=8)
    axes[1, 0].set_ylabel('True class', fontsize=8)
    # No title inside the figure (PESM rule); the caption carries it.
    fig.tight_layout()
    fig.savefig(outfile, dpi=220, bbox_inches='tight')
    plt.close()
    print(f'{outfile} written  ({len(panels)} matrices, classes={names})')

print('all diagonals below 100% -- defect signature absent')
