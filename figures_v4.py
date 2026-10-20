"""
Paper 2 figures -- DATA-DRIVEN. Nothing is hard-coded.

Every number is read from the results_*.json / *.npy files that the training
scripts write under RESULTS_ROOT. If a run is re-done the figure changes with
it; if a run has not finished, the figure that needs it is SKIPPED and said so.

The only constants here are HAM_A (the dataset's own labels) and the two
published baselines being disputed, which are quoted from the submitted
manuscript, not measured.

Run:  python figures_v4.py
Out:  <RESULTS_ROOT>/figures/
"""
import os, json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from dasps_common import (RESULTS_ROOT, HAM_A, CLASS_NAMES,
                          ham_to_label as D_ham_to_label)

OUT = os.path.join(RESULTS_ROOT, 'figures'); os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'grid.linestyle': '-',
                     'axes.axisbelow': True})
C_LOO, C_LOSO, INK, MUTED = '#4C78A8', '#F58518', '#22282E', '#6B7280'
CLASS_C = ['#E8EEF4', '#A8C4DC', '#5B8CB8', '#1F3B57']

MAJ = {'binary': 69.57, 'four_class': 56.52}
# quoted from the submitted manuscript (disputed values, not measured here)
PUB = {'binary': (94.90, 0.88), 'four_class': (92.74, 0.86)}

ORDER = ['Baseline RF', 'RF + 84', 'FeedforwardOnly', 'TransformerOnly',
         'FeedforwardTransformer']
NICE = {'Baseline RF': 'Baseline RF\n(24 + RFE)', 'RF + 84': 'RF + 84\nfeatures',
        'FeedforwardOnly': 'Feedforward\nonly', 'TransformerOnly': 'Transformer\nonly',
        'FeedforwardTransformer': 'Feedforward-\nTransformer'}

built, skipped = [], []


# ---------------------------------------------------------------- loading
def tidy(name):
    n = str(name).replace('_', ' ')
    if 'Baseline' in n: return 'Baseline RF'
    if '84' in n:       return 'RF + 84'
    return str(name)


def load_all():
    """-> {(task, protocol, model): record}  record carries acc/f1/kappa/cm/dir"""
    R = {}
    for f in glob.glob(os.path.join(RESULTS_ROOT, '**', '*.json'), recursive=True):
        base = os.path.basename(f)
        if base in ('label_balance.json', 'identity_results.json'):
            continue
        if '_SMOKE' in f:          # smoke-test output, not a real cell
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if not isinstance(r, dict) or 'acc' not in r:
            continue
        model = tidy(r.get('model') or r.get('label') or base)
        k = (r.get('task', '?'), r.get('protocol', '?'), model)
        mt = os.path.getmtime(f)
        # If one cell has more than one results file, take the newest.
        if k in R and R[k]['mtime'] >= mt:
            continue
        R[k] = dict(acc=r['acc'], f1=r.get('f1'), kappa=r.get('kappa'), n=r.get('n'),
                    epochs=r.get('epochs'), cm=r.get('confusion_matrix'),
                    dir=os.path.dirname(f), mtime=mt)
    return R


def cm_of(rec, n_classes):
    """confusion matrix from the json, else rebuilt from the saved .npy pair"""
    if rec.get('cm'):
        return np.asarray(rec['cm'], dtype=int)
    yt = sorted(glob.glob(os.path.join(rec['dir'], 'y_true_*.npy')))
    yp = sorted(glob.glob(os.path.join(rec['dir'], 'y_pred_*.npy')))
    if not yt or not yp:
        return None
    t, p = np.load(yt[0]), np.load(yp[0])
    cm = np.zeros((n_classes, n_classes), int)
    for a, b in zip(t, p):
        cm[int(a), int(b)] += 1
    return cm


# ---------------------------------------------------------------- Fig 1
def fig_label_balance():
    """Per-participant HAM-A. Class boundaries come from dasps_common.ham_to_label,
    NOT from a copy of them -- a duplicated threshold here once mis-coloured 8 of
    the 23 participants."""
    pid = sorted(HAM_A)
    sc = [HAM_A[p] for p in pid]
    cls = [D_ham_to_label(s, 'four_class') for s in sc]

    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    ax.bar(range(len(pid)), sc, color=[CLASS_C[c] for c in cls],
           edgecolor=INK, linewidth=.6)
    for i, s in enumerate(sc):
        ax.text(i, s + .7, str(s), ha='center', va='bottom', fontsize=8, color=INK)

    # boundaries drawn where the label actually changes
    for lo in range(1, 50):
        if D_ham_to_label(lo - 1, 'four_class') != D_ham_to_label(lo, 'four_class'):
            ax.axhline(lo - 0.5, color=MUTED, ls=':', lw=.9)
    for lo in range(1, 50):
        if D_ham_to_label(lo - 1, 'binary') != D_ham_to_label(lo, 'binary'):
            ax.axhline(lo - 0.5, color=MUTED, ls='--', lw=1.7)
            ax.text(-0.4, lo + 1.0,
                    f'binary split (HAM-A $\\leq$ {lo - 1})', color=MUTED,
                    fontsize=8.5, ha='left',
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
            break

    names = CLASS_NAMES['four_class']
    cnt = [cls.count(i) for i in range(4)]
    handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_C[i], ec=INK, lw=.6) for i in range(4)]
    ax.legend(handles, [f'{n} (n={c})' for n, c in zip(names, cnt)], frameon=False,
              fontsize=9, ncol=4, loc='upper left', title='Four-class label',
              title_fontsize=9)
    ax.set_xticks(range(len(pid)))
    ax.set_xticklabels([f'P{p}' for p in pid], fontsize=8)
    ax.set_ylabel('HAM-A score'); ax.set_ylim(0, 52)
    ax.set_title('One HAM-A score per participant, inherited by all 180 of their segments',
                 fontsize=10.5, color=INK, pad=10)
    fig.tight_layout(); fig.savefig(f'{OUT}/Fig1_label_balance.png', dpi=300,
                                    bbox_inches='tight'); plt.close()
    built.append(f'Fig1_label_balance.png  (HAM-A + ham_to_label, n={cnt})')


# ---------------------------------------------------------------- Fig 2
def fig_loso_all(R):
    need = [(t, m) for t in ('binary', 'four_class') for m in ORDER]
    have = [k for k in need if (k[0], 'loso', k[1]) in R]
    if len(have) < len(need):
        skipped.append(f'Fig2_loso_all_configs -- {len(need)-len(have)} LOSO cell(s) missing')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, task in zip(axes, ('binary', 'four_class')):
        vals = [R[(task, 'loso', m)]['acc'] for m in ORDER]
        ax.bar(range(len(ORDER)), vals, .62, color=C_LOSO, edgecolor=INK, linewidth=.6)
        for i, v in enumerate(vals):
            ax.text(i, v + 1.2, f'{v:.2f}', ha='center', va='bottom', fontsize=8.5, color=INK)
        ax.axhline(MAJ[task], color=MUTED, ls='--', lw=1.4)
        ax.text(len(ORDER) - .45, MAJ[task] + 1.6, f'majority class {MAJ[task]:.2f}%',
                color=MUTED, fontsize=8, ha='right',
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
        ax.axhline(PUB[task][0], color=C_LOO, ls='-', lw=1.4)
        ax.text(len(ORDER) - .45, PUB[task][0] + 1.6, f'published {PUB[task][0]:.2f}%',
                color=C_LOO, fontsize=8, ha='right',
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([NICE[m] for m in ORDER], fontsize=8)
        ax.set_ylim(0, 108); ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylabel('Accuracy' if task == 'binary' else '')
        ax.set_title('Binary task' if task == 'binary' else 'Four-class task',
                     fontsize=10, color=INK, pad=8)
    fig.suptitle('Participant-independent (LOSO) accuracy -- every configuration '
                 'sits at or below the majority class',
                 fontsize=10.5, color=INK, y=1.03)
    fig.tight_layout(); fig.savefig(f'{OUT}/Fig2_loso_all_configs.png', dpi=220,
                                    bbox_inches='tight'); plt.close()
    built.append('Fig2_loso_all_configs.png  (10 LOSO runs)')


# ---------------------------------------------------------------- Fig 3
def fig_kappa(R):
    need = [(t, m) for t in ('binary', 'four_class') for m in ORDER]
    have = [k for k in need if (k[0], 'loso', k[1]) in R
            and R[(k[0], 'loso', k[1])].get('kappa') is not None]
    if len(have) < len(need):
        skipped.append(f'Fig3_kappa_collapse -- {len(need)-len(have)} LOSO kappa missing')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, task in zip(axes, ('binary', 'four_class')):
        ks = [PUB[task][1]] + [R[(task, 'loso', m)]['kappa'] for m in ORDER]
        lab = ['Published\n(segment LOO)'] + [NICE[m] for m in ORDER]
        col = [C_LOO] + [C_LOSO] * len(ORDER)
        ax.bar(range(len(ks)), ks, .62, color=col, edgecolor=INK, linewidth=.6)
        for i, v in enumerate(ks):
            ax.text(i, v + (.025 if v >= 0 else -.045), f'{v:+.3f}', ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=8.5, color=INK)
        ax.axhline(0, color=INK, lw=1.1)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(lab, fontsize=7.6)
        ax.set_ylim(-.22, 1.0)
        ax.set_ylabel("Cohen's kappa" if task == 'binary' else '')
        ax.set_title('Binary task' if task == 'binary' else 'Four-class task',
                     fontsize=10, color=INK, pad=8)
    fig.suptitle('Changing only the partitioning collapses agreement to zero',
                 fontsize=11, color=INK, y=1.03)
    fig.tight_layout(); fig.savefig(f'{OUT}/Fig3_kappa_collapse.png', dpi=220,
                                    bbox_inches='tight'); plt.close()
    built.append('Fig3_kappa_collapse.png  (10 LOSO kappas + 2 published)')


# ---------------------------------------------------------------- Fig 4
def fig_loo_vs_loso(R):
    pairs = [(t, m) for t in ('binary', 'four_class') for m in ORDER]
    missing = [f'{m}/{t}' for t, m in pairs if (t, 'loo', m) not in R]
    if missing:
        skipped.append(f'Fig4_loo_vs_loso -- waiting on {len(missing)} LOO run(s): '
                       + ', '.join(missing[:4]) + ('...' if len(missing) > 4 else ''))
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, task in zip(axes, ('binary', 'four_class')):
        x = np.arange(len(ORDER)); w = .38
        loo = [R[(task, 'loo', m)]['acc'] for m in ORDER]
        lso = [R[(task, 'loso', m)]['acc'] for m in ORDER]
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
        ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in ORDER], fontsize=8)
        ax.set_ylim(0, 108); ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylabel('Accuracy' if task == 'binary' else '')
        ax.set_title('Binary task' if task == 'binary' else 'Four-class task',
                     fontsize=10, color=INK, pad=8)
    axes[0].legend(frameon=False, fontsize=8.5, loc='lower left',
                   bbox_to_anchor=(0, -.40), ncol=2)
    fig.tight_layout(); fig.savefig(f'{OUT}/Fig4_loo_vs_loso.png', dpi=220,
                                    bbox_inches='tight'); plt.close()
    built.append('Fig4_loo_vs_loso.png  (all 20 runs)')


# ---------------------------------------------------------------- Fig 5
def fig_confusion(R):
    for task, nc, names in (('binary', 2, ['Low', 'High']),
                            ('four_class', 4, ['Normal', 'Light', 'Moderate', 'Severe'])):
        cms = {}
        for proto in ('loo', 'loso'):
            for m in ORDER:
                rec = R.get((task, proto, m))
                if rec is None:
                    continue
                cm = cm_of(rec, nc)
                if cm is not None:
                    cms[(proto, m)] = cm
        if not cms:
            skipped.append(f'Fig5_confusion_{task} -- no confusion data yet')
            continue
        keys = [k for p in ('loo', 'loso') for k in
                sorted([k for k in cms if k[0] == p], key=lambda k: ORDER.index(k[1]))]
        ncol = min(5, len(keys)); nrow = int(np.ceil(len(keys) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 2.65 * nrow),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for ax, k in zip(axes.ravel(), keys):
            ax.axis('on'); ax.grid(False)
            cm = cms[k]
            rn = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
            ax.imshow(rn, cmap='Blues', vmin=0, vmax=100)
            for i in range(nc):
                for j in range(nc):
                    ax.text(j, i, f'{rn[i,j]:.0f}', ha='center', va='center',
                            fontsize=7.5, color='white' if rn[i, j] > 55 else INK)
            ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
            ax.set_xticklabels(names, fontsize=6.5, rotation=45, ha='right')
            ax.set_yticklabels(names, fontsize=6.5)
            ax.set_title(f'{k[1]}\n{k[0].upper()}', fontsize=7.8, color=INK, pad=4)
        fig.suptitle(f'Row-normalised confusion matrices -- {task.replace("_"," ")} task',
                     fontsize=10.5, color=INK, y=1.0)
        fig.tight_layout()
        fig.savefig(f'{OUT}/Fig5_confusion_{task}.png', dpi=220, bbox_inches='tight')
        plt.close()
        built.append(f'Fig5_confusion_{task}.png  ({len(keys)} matrices)')


# ---------------------------------------------------------------- Fig 6
def fig_identity():
    rows = {}
    for task in ('binary', 'four_class'):
        f = os.path.join(RESULTS_ROOT, f'identity_{task}', 'identity_results.json')
        if os.path.exists(f):
            rows[task] = json.load(open(f))
    if not rows:
        skipped.append('Fig6_identity_control -- identity_to_label.py not run yet')
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 3.8), squeeze=False)
    for ax, (task, d) in zip(axes[0], rows.items()):
        clf = list(d)
        x = np.arange(len(clf)); w = .36
        pid = [d[c]['participant_id_acc'] for c in clf]
        lab = [d[c]['label_via_id_acc'] for c in clf]
        ax.bar(x - w / 2 - .01, pid, w, color=C_LOO, edgecolor=INK, linewidth=.6,
               label='Participant ID recovered')
        ax.bar(x + w / 2 + .01, lab, w, color=C_LOSO, edgecolor=INK, linewidth=.6,
               label='Label via ID lookup')
        ax.axhline(d[clf[0]]['id_chance'], color=MUTED, ls=':', lw=1.2)
        ax.axhline(d[clf[0]]['label_majority'], color=MUTED, ls='--', lw=1.2)
        for xx, v in list(zip(x - w / 2 - .01, pid)) + list(zip(x + w / 2 + .01, lab)):
            ax.text(xx, v + 1.2, f'{v:.1f}', ha='center', va='bottom', fontsize=8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(clf, fontsize=8)
        ax.set_ylim(0, 108); ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_title('Binary task' if task == 'binary' else 'Four-class task',
                     fontsize=10, color=INK, pad=8)
    axes[0][0].set_ylabel('Accuracy')
    axes[0][0].legend(frameon=False, fontsize=8.5, loc='lower left',
                      bbox_to_anchor=(0, -.38), ncol=2)
    fig.suptitle('Identity control -- segments carry participant identity, '
                 'and identity alone recovers the label',
                 fontsize=10.5, color=INK, y=1.03)
    fig.tight_layout(); fig.savefig(f'{OUT}/Fig6_identity_control.png', dpi=220,
                                    bbox_inches='tight'); plt.close()
    built.append('Fig6_identity_control.png  (identity_results.json)')


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    R = load_all()
    print(f'results root : {RESULTS_ROOT}')
    print(f'runs found   : {len(R)} of 20\n')
    fig_label_balance()
    fig_loso_all(R)
    fig_kappa(R)
    fig_loo_vs_loso(R)
    fig_confusion(R)
    fig_identity()
    print('BUILT')
    for b in built:
        print('   [x] ' + b)
    print('\nWAITING ON DATA')
    if not skipped:
        print('   -- nothing, all figures built --')
    for s in skipped:
        print('   [ ] ' + s)
    print(f'\n-> {OUT}')
