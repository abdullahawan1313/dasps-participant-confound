"""
analysis_stats.py -- significance testing over results already on disk.
=======================================================================

Discovers every completed cell under RESULTS_ROOT and reports:

  1. Accuracy per cell with a naive Wilson 95% interval and a
     participant-level cluster bootstrap 95% interval
  2. LOO versus LOSO for each model, under both tests
  3. Configuration versus configuration within one protocol
  4. Everything written to stats_summary.json

    python -u analysis_stats.py

Two tests are reported, for a reason:

  * McNemar treats every segment as an independent unit. It is the standard
    paired test, but here 180 segments come from one participant, so it is
    anticonservative.
  * The participant-level paired bootstrap resamples participants rather
    than segments and respects that clustering. This is the test to report;
    McNemar is shown only for comparison.

No test is possible against the previously published figures, because their
per-segment predictions were never released. For that comparison a
confidence interval is the appropriate summary, not a significance test.
"""
import os, sys, glob, json, math
import numpy as np
from scipy import stats

from dasps_common import RESULTS_ROOT

ORDER = ['Baseline RF', 'RF + 84', 'FeedforwardOnly', 'TransformerOnly',
         'FeedforwardTransformer']
N_BOOT = 10000
SEED = 42


# ---------------------------------------------------------------- discovery
def tidy(n):
    n = str(n)
    if 'Baseline' in n.replace('_', ' '): return 'Baseline RF'
    if '84' in n: return 'RF + 84'
    return n


def discover():
    """-> {(task, protocol, model): dict(y, p, sids, acc)}

    Each cell has its own folder, but one folder may hold more than one
    configuration (the rf_* folders hold two), so cells are keyed by their
    tag rather than by the directory they live in."""
    cells = {}
    for jf in glob.glob(os.path.join(RESULTS_ROOT, '**', '*.json'), recursive=True):
        base = os.path.basename(jf)
        if base in ('label_balance.json', 'identity_results.json',
                    'stats_summary.json'):
            continue
        if '_SMOKE' in jf:
            continue
        try:
            r = json.load(open(jf))
        except Exception:
            continue
        if not isinstance(r, dict) or 'acc' not in r:
            continue

        d = os.path.dirname(jf)
        tag = base[len('results_'):-len('.json')] if base.startswith('results_') else None
        if tag is None:
            continue
        yp = os.path.join(d, f'y_pred_{tag}.npy')
        yt = os.path.join(d, f'y_true_{tag}.npy')
        if not (os.path.exists(yp) and os.path.exists(yt)):
            continue

        sf = os.path.join(d, 'subject_ids.npy')
        sids = np.load(sf) if os.path.exists(sf) else None
        key = (r.get('task', '?'), r.get('protocol', '?'),
               tidy(r.get('model') or r.get('label') or tag))
        mt = os.path.getmtime(jf)
        if key in cells and cells[key]['mtime'] >= mt:
            continue
        cells[key] = dict(y=np.load(yt), p=np.load(yp), sids=sids,
                          acc=r['acc'], kappa=r.get('kappa'), mtime=mt, file=jf)
    return cells


# ---------------------------------------------------------------- intervals
def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (0., 0.)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n * n)) / d
    return (max(0, c - h) * 100, min(1, c + h) * 100)


def _group_stats(correct, groups):
    """Return per-participant sums and counts of correct predictions.

    This is what lets the bootstrap be vectorised: instead of concatenating
    23 arrays on every replicate, the resampled accuracy is a ratio of summed
    sums to summed counts. The result is identical and about 100x faster."""
    uq, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=correct.astype(float), minlength=len(uq))
    c = np.bincount(inv, minlength=len(uq)).astype(float)
    return uq, s, c


def cluster_bootstrap_ci(correct, groups, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Resample participants, not segments.

    The 180 segments of one participant are not independent observations, so
    an interval that treats them as independent is far too narrow."""
    uq, s, c = _group_stats(correct, groups)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(uq), size=(n_boot, len(uq)))
    accs = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(accs, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return lo * 100, hi * 100


# ---------------------------------------------------------------- tests
def mcnemar(y, a, b):
    """Exact McNemar test. b01 = a correct and b wrong; b10 = the reverse.

    It treats the segments as independent, which is not true here; it is
    reported only for comparison with the cluster bootstrap."""
    ca, cb = (a == y), (b == y)
    b01 = int(np.sum(ca & ~cb)); b10 = int(np.sum(~ca & cb))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    p = min(1.0, 2 * stats.binom.cdf(min(b01, b10), n, 0.5))
    return b01, b10, p


def paired_cluster_bootstrap(y, a, b, groups, n_boot=N_BOOT, seed=SEED):
    """Participant-level paired bootstrap of the accuracy difference (a - b).

    The same resampled participants are used for both configurations, so the
    comparison stays paired. Returns the difference in percentage points, a
    95% interval and a two-sided p-value."""
    ca = (a == y).astype(float); cb = (b == y).astype(float)
    uq, sa, c = _group_stats(ca, groups)
    _,  sb, _ = _group_stats(cb, groups)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(uq), size=(n_boot, len(uq)))
    den = c[idx].sum(1)
    diffs = (sa[idx].sum(1) - sb[idx].sum(1)) / den
    obs = ca.mean() - cb.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided p: centre the bootstrap distribution on zero, then count
    centred = diffs - diffs.mean()
    p = (np.sum(np.abs(centred) >= abs(obs)) + 1) / (n_boot + 1)
    return obs * 100, lo * 100, hi * 100, float(min(1.0, p))


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'


def pfmt(p, n_boot=N_BOOT):
    """Format a bootstrap p-value.

    The smallest value the bootstrap can produce is 1/(n_boot+1). Printing
    that with '=' would be misleading, because the true p may be smaller."""
    floor = 1.0 / (n_boot + 1)
    return f'< {floor:.1g}' if p <= floor else f'= {p:.3g}'


# ---------------------------------------------------------------- report
def main():
    cells = discover()
    if not cells:
        print(f"No completed cells found under: {RESULTS_ROOT}")
        return

    out = {'cells': {}, 'loo_vs_loso': [], 'within_protocol': []}
    W = 86

    print('\n' + '=' * W)
    print('  1. ACCURACY WITH CONFIDENCE INTERVALS')
    print('=' * W)
    print('  Wilson = naive (segments independent). Cluster = participants'
          ' resampled.')
    print('  The cluster interval is the one to report.\n')
    print(f'  {"Cell":<42}{"Acc %":>8}{"Wilson 95%":>20}{"Cluster 95%":>20}')
    print('  ' + '-' * (W - 2))

    for task in ('binary', 'four_class'):
        for proto in ('loo', 'loso'):
            for m in ORDER:
                c = cells.get((task, proto, m))
                if not c:
                    continue
                corr = (c['y'] == c['p']).astype(float)
                k, n = int(corr.sum()), len(corr)
                wlo, whi = wilson_ci(k, n)
                name = f'{m} / {proto.upper()} / {task}'
                rec = {'acc': k / n * 100, 'n': n,
                       'wilson': [wlo, whi], 'cluster': None}
                if c['sids'] is not None and len(c['sids']) == n:
                    clo, chi = cluster_bootstrap_ci(corr, c['sids'])
                    rec['cluster'] = [clo, chi]
                    cl = f'[{clo:.2f}, {chi:.2f}]'
                else:
                    cl = 'no subject_ids'
                print(f'  {name:<42}{k/n*100:>8.2f}'
                      f'{f"[{wlo:.2f}, {whi:.2f}]":>20}{cl:>20}')
                out['cells'][name] = rec

    # ---- 2. LOO vs LOSO for one model: the central comparison ----
    print('\n' + '=' * W)
    print('  2. LOO vs LOSO -- same model, only the partitioning changes')
    print('=' * W)
    any_pair = False
    for task in ('binary', 'four_class'):
        for m in ORDER:
            a = cells.get((task, 'loo', m)); b = cells.get((task, 'loso', m))
            if not (a and b):
                continue
            if len(a['y']) != len(b['y']) or not np.array_equal(a['y'], b['y']):
                print(f'  [skip] {m} / {task}: y_true does not match')
                continue
            any_pair = True
            sids = a['sids'] if a['sids'] is not None else b['sids']
            b01, b10, pmc = mcnemar(a['y'], a['p'], b['p'])
            print(f'\n  {m} -- {task}')
            print(f'    LOO  {a["acc"]:.2f}%      LOSO {b["acc"]:.2f}%'
                  f'      difference {a["acc"]-b["acc"]:+.2f} points')
            print(f'    McNemar (assumes independent segments)  : '
                  f'p = {pmc:.3g}  {stars(pmc)}   [b01={b01}, b10={b10}]')
            rec = {'pair': f'{m}/{task}', 'loo': a['acc'], 'loso': b['acc'],
                   'mcnemar_p': pmc, 'b01': b01, 'b10': b10}
            if sids is not None and len(sids) == len(a['y']):
                d, lo, hi, pb = paired_cluster_bootstrap(a['y'], a['p'], b['p'], sids)
                print(f'    Participant-cluster bootstrap (report this)     : '
                      f'{d:+.2f} pts, 95% CI [{lo:+.2f}, {hi:+.2f}], '
                      f'p {pfmt(pb)}  {stars(pb)}')
                rec.update(diff=d, ci=[lo, hi], cluster_p=pb)
            out['loo_vs_loso'].append(rec)
    if not any_pair:
        print('  (no model yet has both protocols complete)')

    # ---- 3. config vs config, same protocol ----
    print('\n' + '=' * W)
    print('  3. CONFIGURATION vs CONFIGURATION -- within one protocol')
    print('  (No test against the previously published figures is possible:')
    print('   their per-segment predictions were never released, so those')
    print('   comparisons are reported as intervals, not as tests.)')
    print('=' * W)
    for task in ('binary', 'four_class'):
        for proto in ('loo', 'loso'):
            have = [m for m in ORDER if (task, proto, m) in cells]
            if len(have) < 2:
                continue
            print(f'\n  --- {task} / {proto.upper()} ---')
            for i in range(len(have)):
                for j in range(i + 1, len(have)):
                    A = cells[(task, proto, have[i])]
                    B = cells[(task, proto, have[j])]
                    if not np.array_equal(A['y'], B['y']):
                        continue
                    sids = A['sids'] if A['sids'] is not None else B['sids']
                    b01, b10, pmc = mcnemar(A['y'], A['p'], B['p'])
                    line = (f'    {have[i]:<24} vs {have[j]:<24} '
                            f'{A["acc"]-B["acc"]:+6.2f} pts   McNemar p={pmc:.3g} {stars(pmc)}')
                    rec = {'a': have[i], 'b': have[j], 'task': task,
                           'protocol': proto, 'mcnemar_p': pmc}
                    if sids is not None and len(sids) == len(A['y']):
                        d, lo, hi, pb = paired_cluster_bootstrap(
                            A['y'], A['p'], B['p'], sids)
                        line += f'   cluster p{pfmt(pb)} {stars(pb)}'
                        rec.update(diff=d, ci=[lo, hi], cluster_p=pb)
                    print(line)
                    out['within_protocol'].append(rec)

    path = os.path.join(RESULTS_ROOT, 'stats_summary.json')
    json.dump(out, open(path, 'w'), indent=2)
    print(f'\n  Saved: {path}\n')


if __name__ == '__main__':
    main()
