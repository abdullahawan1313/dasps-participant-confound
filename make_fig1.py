"""
fig_label_structure.png -- per-participant HAM-A scores and the labels they induce.

Labels are read from label_balance.json, which is written by label_balance.py
using dasps_common.ham_to_label. Nothing about the class boundaries is
re-derived here, so the figure cannot disagree with the code that produced the
experiments.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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



LB = os.path.join(_results_root(), 'label_balance.json')

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25,
                     'grid.linestyle': '-', 'axes.axisbelow': True})
INK, MUTED = '#22282E', '#6B7280'
ORDER = ['Normal', 'Light', 'Moderate', 'Severe']
COL = {'Normal': '#E8EEF4', 'Light': '#A8C4DC',
       'Moderate': '#5B8CB8', 'Severe': '#1F3B57'}

rows = sorted(json.load(open(LB)), key=lambda r: r['participant'])
pid = [r['participant'] for r in rows]
ham = [r['ham_a'] for r in rows]
cls = [r['four_class'] for r in rows]

# class boundaries inferred from the data itself, not hard-coded
bounds = []
for a, b in zip(ORDER, ORDER[1:]):
    lo = max([r['ham_a'] for r in rows if r['four_class'] == a], default=None)
    hi = min([r['ham_a'] for r in rows if r['four_class'] == b], default=None)
    if lo is not None and hi is not None:
        bounds.append((a, b, (lo + hi) / 2))

fig, ax = plt.subplots(figsize=(11.5, 3.9))
ax.bar(range(len(pid)), ham, color=[COL[c] for c in cls],
       edgecolor=INK, linewidth=.6)
for i, (s, c) in enumerate(zip(ham, cls)):
    ax.text(i, s + .7, str(s), ha='center', va='bottom', fontsize=8, color=INK)

for _a, _b, y in bounds:
    ax.axhline(y, color=MUTED, ls=':', lw=.9)

# binary split: dasps_common.ham_to_label uses HAM-A <= 20 -> Light.
# Verified against the data rather than assumed.
BIN_T = 20
assert max(r['ham_a'] for r in rows if r['binary'] == 'Light') <= BIN_T
assert min(r['ham_a'] for r in rows if r['binary'] == 'Severe') > BIN_T
ax.axhline(BIN_T + 0.5, color=MUTED, ls='--', lw=1.7)
ax.text(-0.4, BIN_T + 1.5,
        f'binary split: Light (HAM-A $\\leq$ {BIN_T}) vs Severe ($>$ {BIN_T})',
        color=MUTED, fontsize=8.5, ha='left',
        bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[c], ec=INK, lw=.6) for c in ORDER]
counts = {c: sum(1 for x in cls if x == c) for c in ORDER}
ax.legend(handles, [f'{c} ($n={counts[c]}$)' for c in ORDER], frameon=False,
          fontsize=8.5, ncol=4, loc='upper left',
          title='Four-class label', title_fontsize=9)

ax.set_xticks(range(len(pid)))
ax.set_xticklabels([f'P{p}' for p in pid], fontsize=8)
ax.set_ylabel('HAM-A score')
ax.set_ylim(0, 52)
# No title inside the figure: PESM requires the caption to be the only title.
fig.tight_layout()
fig.savefig('fig/fig_label_structure.png', dpi=300, bbox_inches='tight')
print('fig/fig_label_structure.png written')
print('class boundaries used:', [(a, b, y) for a, b, y in bounds])
print('counts:', counts)
assert sum(counts.values()) == 23
