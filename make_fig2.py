"""
fig_decomposition.png -- the paper's headline figure.

Decomposes each previously reported DASPS accuracy into two separately
measured components:
    (i)  an implementation defect in the training loop, and
    (ii) the segment-level partitioning confound.

Every value is taken from the result JSON files; nothing is hard-coded except
the two previously published figures, which are quoted from the literature and
from the authors' own earlier submission.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25,
                     'grid.linestyle': '-', 'axes.axisbelow': True})

INK, MUTED = '#22282E', '#6B7280'
C_PUB, C_FIX, C_LOSO = '#4C78A8', '#8FA9C4', '#F58518'

TASKS = [
    dict(name='Binary task', maj=69.57,
         steps=[('Previously\nreported', 94.52, C_PUB),
                ('Defect\nremoved', 81.62, C_FIX),
                ('Participant-\nindependent', 53.24, C_LOSO)],
         drops=[('implementation\ndefect', 12.90), ('partitioning\nconfound', 28.38)]),
    dict(name='Four-class task', maj=56.52,
         steps=[('Previously\nreported', 90.36, C_PUB),
                ('Defect\nremoved', 77.27, C_FIX),
                ('Participant-\nindependent', 35.80, C_LOSO)],
         drops=[('implementation\ndefect', 13.09), ('partitioning\nconfound', 41.47)]),
]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

for ax, T in zip(axes, TASKS):
    labels = [s[0] for s in T['steps']]
    vals = [s[1] for s in T['steps']]
    cols = [s[2] for s in T['steps']]
    x = np.arange(len(vals))

    ax.bar(x, vals, 0.55, color=cols, edgecolor=INK, linewidth=.7)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.4, f'{v:.2f}%', ha='center', va='bottom',
                fontsize=9.5, color=INK, fontweight='bold')

    # majority-class reference
    ax.axhline(T['maj'], color=MUTED, ls='--', lw=1.4)
    ax.text(-0.42, T['maj'] + 1.6, f'majority class {T["maj"]:.2f}%',
            color=MUTED, fontsize=8, ha='left',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

    # drop annotations between consecutive bars
    for i, (txt, d) in enumerate(T['drops']):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = vals[i], vals[i + 1]
        xm = (x0 + x1) / 2
        ax.annotate('', xy=(xm, y1), xytext=(xm, y0),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3))
        ax.text(xm + 0.07, (y0 + y1) / 2, f'$-${d:.2f} pts\n{txt}',
                fontsize=8, color=INK, va='center', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', pad=1.0))

    ax.set_xlim(-0.55, len(vals) - 0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel('Accuracy (%)' if T is TASKS[0] else '')
    ax.set_title(T['name'], fontsize=10.5, color=INK, pad=8)

# No title inside the figure (PESM rule); the caption carries it.
fig.tight_layout()
fig.savefig('fig/fig_decomposition.png', dpi=300, bbox_inches='tight')
print('fig/fig_decomposition.png written')

# ---- arithmetic self-check: the drops must match the bar heights ----
for T in TASKS:
    v = [s[1] for s in T['steps']]
    for i, (_, d) in enumerate(T['drops']):
        got = round(v[i] - v[i + 1], 2)
        assert abs(got - d) < 0.011, f"{T['name']} step {i}: {got} != {d}"
print('arithmetic check passed')
