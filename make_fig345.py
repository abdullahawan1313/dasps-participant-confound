"""
fig_identity_control.png / fig_loo_vs_loso.png / fig_kappa_collapse.png

Rebuilt from the result JSON files only. No figure carries an internal title:
PESM's guidelines state "Do not include titles or captions within your
illustrations", so the caption in the manuscript is the only title. The small
"Binary task" / "Four-class task" labels are panel identifiers, not titles.

Majority-class rates are computed from label_balance.json rather than typed in.
The two published kappa values are quoted from the literature and are the only
hard-coded numbers here; they are asserted against the manuscript's own text.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


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

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'grid.linestyle': '-',
                     'axes.axisbelow': True})
C_LOO, C_LOSO, INK, MUTED = '#4C78A8', '#F58518', '#22282E', '#6B7280'

ORDER = [('Baseline_RF', 'Baseline RF\n(24 + RFE)'),
         ('RF_plus_84_features', 'RF + 84\nfeatures'),
         ('FeedforwardOnly', 'Feedforward\nonly'),
         ('TransformerOnly', 'Transformer\nonly'),
         ('FeedforwardTransformer', 'Feedforward-\nTransformer')]

# quoted from Muhammad & Al-Ahmadi (2022), the figures being corrected
PUB_KAPPA = {'binary': 0.88, 'four_class': 0.86}


def cell(task, proto, key):
    if key in ('Baseline_RF', 'RF_plus_84_features'):
        p = os.path.join(ROOT, f'rf_{proto}_{task}', f'results_{key}.json')
    else:
        cand = [d for d in os.listdir(ROOT)
                if d.startswith(f'{proto}_{task}_{key}') and '_SMOKE' not in d]
        assert len(cand) == 1, (task, proto, key, cand)
        p = os.path.join(ROOT, cand[0], f'results_{key}_{task}_{proto}.json')
    return json.load(open(p))


# majority-class rate straight from the label table
lb = json.load(open(os.path.join(ROOT, 'label_balance.json')))
assert len(lb) == 23
MAJ = {}
for task in ('binary', 'four_class'):
    cnt = {}
    for r in lb:
        cnt[r[task]] = cnt.get(r[task], 0) + 1
    MAJ[task] = max(cnt.values()) / sum(cnt.values()) * 100
assert abs(MAJ['binary'] - 69.57) < .01 and abs(MAJ['four_class'] - 56.52) < .01

PANEL = {'binary': 'Binary task', 'four_class': 'Four-class task'}

# ---------------------------------------------------------------- identity
rows = {t: json.load(open(os.path.join(ROOT, f'identity_{t}',
                                       'identity_results.json')))
        for t in ('binary', 'four_class')}
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), squeeze=False)
for ax, task in zip(axes[0], ('binary', 'four_class')):
    d = rows[task]
    clf = list(d)
    x = np.arange(len(clf))
    w = .36
    pid = [d[c]['participant_id_acc'] for c in clf]
    lab = [d[c]['label_via_id_acc'] for c in clf]
    ax.bar(x - w / 2 - .01, pid, w, color=C_LOO, edgecolor=INK, linewidth=.6,
           label='Participant identity recovered')
    ax.bar(x + w / 2 + .01, lab, w, color=C_LOSO, edgecolor=INK, linewidth=.6,
           label='HAM-A label via identity lookup')
    ax.axhline(d[clf[0]]['id_chance'], color=MUTED, ls=':', lw=1.2)
    ax.axhline(d[clf[0]]['label_majority'], color=MUTED, ls='--', lw=1.2)
    for xx, v in list(zip(x - w / 2 - .01, pid)) + list(zip(x + w / 2 + .01, lab)):
        ax.text(xx, v + 1.2, f'{v:.1f}', ha='center', va='bottom',
                fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(['Logistic regression', 'Random forest'], fontsize=8)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title(PANEL[task], fontsize=9.5, color=INK, pad=6)
axes[0][0].set_ylabel('Accuracy')
axes[0][0].legend(frameon=False, fontsize=8.5, loc='lower left',
                  bbox_to_anchor=(0, -.40), ncol=2)
fig.tight_layout()
fig.savefig('fig/fig_identity_control.png', dpi=300, bbox_inches='tight')
plt.close()
print('fig/fig_identity_control.png')

# ---------------------------------------------------------------- LOO vs LOSO
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
for ax, task in zip(axes, ('binary', 'four_class')):
    x = np.arange(len(ORDER))
    w = .38
    loo = [cell(task, 'loo', k)['acc'] for k, _ in ORDER]
    lso = [cell(task, 'loso', k)['acc'] for k, _ in ORDER]
    b1 = ax.bar(x - w / 2 - .01, loo, w, color=C_LOO, edgecolor=INK, linewidth=.6,
                label='Segment-level LOO (as reported)')
    b2 = ax.bar(x + w / 2 + .01, lso, w, color=C_LOSO, edgecolor=INK, linewidth=.6,
                label='Participant-independent LOSO')
    ax.axhline(MAJ[task], color=MUTED, ls='--', lw=1.4)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f'{b.get_height():.1f}', ha='center', va='bottom',
                    fontsize=7.6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in ORDER], fontsize=8)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel('Accuracy' if task == 'binary' else '')
    ax.set_title(PANEL[task], fontsize=9.5, color=INK, pad=6)
axes[0].legend(frameon=False, fontsize=8.5, loc='lower left',
               bbox_to_anchor=(0, -.42), ncol=2)
fig.tight_layout()
fig.savefig('fig/fig_loo_vs_loso.png', dpi=300, bbox_inches='tight')
plt.close()
print('fig/fig_loo_vs_loso.png')

# ---------------------------------------------------------------- kappa
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax, task in zip(axes, ('binary', 'four_class')):
    ks = [PUB_KAPPA[task]] + [cell(task, 'loso', k)['kappa'] for k, _ in ORDER]
    lab = ['Previously\nreported\n(segment LOO)'] + [n for _, n in ORDER]
    col = [C_LOO] + [C_LOSO] * len(ORDER)
    ax.bar(range(len(ks)), ks, .62, color=col, edgecolor=INK, linewidth=.6)
    for i, v in enumerate(ks):
        ax.text(i, v + (.025 if v >= 0 else -.045), f'{v:+.3f}', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=8.5, color=INK)
    ax.axhline(0, color=INK, lw=1.1)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(lab, fontsize=7.6)
    ax.set_ylim(-.22, 1.0)
    ax.set_ylabel("Cohen's kappa" if task == 'binary' else '')
    ax.set_title(PANEL[task], fontsize=9.5, color=INK, pad=6)
fig.tight_layout()
fig.savefig('fig/fig_kappa_collapse.png', dpi=300, bbox_inches='tight')
plt.close()
print('fig/fig_kappa_collapse.png')
print('no figure carries an internal title')
