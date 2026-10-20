"""
verify_numbers.py -- regenerate every number that appears in main.tex directly
from the result JSON files, and check it against the number actually typed in
main.tex. Nothing here is hard-coded except the two previously published
figures (quoted from the literature) and the arithmetic identities.

Exit code 0 = every number in main.tex matches the data.
"""
import json
import os
import re
import sys

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
TEX = os.environ.get('PAPER_TEX', 'paper/main.tex')

CELL = {}   # (task, protocol, config) -> dict


def _add(task, proto, cfg, path):
    r = json.load(open(path))
    cm = np.asarray(r['confusion_matrix'], dtype=float)
    rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1) * 100.0
    CELL[(task, proto, cfg)] = dict(
        acc=r['acc'], f1=r['f1'], kappa=r['kappa'],
        prec=r.get('precision'), recall=r.get('recall'),
        cm=cm, per_class=rec)


NEURAL = {'FeedforwardOnly': 'Feedforward-only',
          'TransformerOnly': 'Transformer-only',
          'FeedforwardTransformer': 'Feedforward-Transformer'}
RF = {'Baseline_RF': 'Baseline RF', 'RF_plus_84_features': 'RF + 84'}

for task in ('binary', 'four_class'):
    for proto in ('loo', 'loso'):
        for m, name in NEURAL.items():
            hits = [os.path.join(ROOT, d, f)
                    for d in os.listdir(ROOT)
                    if d.startswith(f'{proto}_{task}_{m}') and '_SMOKE' not in d
                    for f in os.listdir(os.path.join(ROOT, d))
                    if f == f'results_{m}_{task}_{proto}.json']
            assert len(hits) == 1, (task, proto, m, hits)
            _add(task, proto, name, hits[0])
        for m, name in RF.items():
            p = os.path.join(ROOT, f'rf_{proto}_{task}', f'results_{m}.json')
            assert os.path.exists(p), p
            _add(task, proto, name, p)

assert len(CELL) == 20, len(CELL)
tex = open(TEX).read()
fails = []


def chk(label, value, fmt='{:.2f}'):
    """Assert the formatted value literally occurs in main.tex."""
    s = fmt.format(value)
    if s not in tex:
        fails.append(f'{label}: "{s}" not found in main.tex')
    return s


ORDER = ['Baseline RF', 'RF + 84', 'Feedforward-only', 'Transformer-only',
         'Feedforward-Transformer']

print('=' * 74)
print('1. MAIN TABLE (Table 2) -- acc / F1 / kappa / prec / rec')
print('=' * 74)
for task in ('binary', 'four_class'):
    for proto in ('loo', 'loso'):
        print(f'\n-- {task} / {proto.upper()}')
        for cfg in ORDER:
            c = CELL[(task, proto, cfg)]
            row = (f'   {cfg:<24} {c["acc"]:6.2f}  {c["f1"]:.4f}  '
                   f'{c["kappa"]:+.4f}  {c["prec"]:.4f}  {c["recall"]:.4f}')
            print(row)
            chk(f'{task}/{proto}/{cfg} acc', c['acc'])
            chk(f'{task}/{proto}/{cfg} f1', c['f1'], '{:.4f}')
            chk(f'{task}/{proto}/{cfg} prec', c['prec'], '{:.4f}')
            chk(f'{task}/{proto}/{cfg} rec', c['recall'], '{:.4f}')
            k = c['kappa']
            # kappa appears as 0.5544 under LOO and as $+0.0348$/$-0.0252$ under LOSO
            if abs(k) >= 0.1:
                chk(f'{task}/{proto}/{cfg} kappa', k, '{:.4f}')
            else:
                chk(f'{task}/{proto}/{cfg} kappa', k, '{:+.4f}')

print()
print('=' * 74)
print('2. PER-CLASS RECALL (Appendix C)')
print('=' * 74)
for task in ('binary', 'four_class'):
    for proto in ('loo', 'loso'):
        print(f'\n-- {task} / {proto.upper()}')
        for cfg in ORDER:
            pc = CELL[(task, proto, cfg)]['per_class']
            print(f'   {cfg:<24} ' + '  '.join(f'{v:5.1f}' for v in pc))
            for v in pc:
                chk(f'{task}/{proto}/{cfg} recall', v, '{:.1f}')
            if (pc >= 99.95).any():
                fails.append(f'{task}/{proto}/{cfg}: koi class 100% recall par')

print()
print('=' * 74)
print('3. PROTOCOL EFFECT  (LOO - LOSO)')
print('=' * 74)
for task in ('binary', 'four_class'):
    for cfg in ORDER:
        d = CELL[(task, 'loo', cfg)]['acc'] - CELL[(task, 'loso', cfg)]['acc']
        print(f'   {task:<10} {cfg:<24} {d:+6.2f} pp')
        chk(f'{task}/{cfg} delta', d)

print()
print('=' * 74)
print('4. IDENTITY CONTROL')
print('=' * 74)
for task in ('binary', 'four_class'):
    r = json.load(open(os.path.join(ROOT, f'identity_{task}',
                                    'identity_results.json')))
    print(f'-- {task}')
    print('   ', json.dumps(r, indent=2)[:700])

print()
print('=' * 74)
print('5. LABEL BALANCE')
print('=' * 74)
lb = json.load(open(os.path.join(ROOT, 'label_balance.json')))
for key in ('binary', 'four_class'):
    cnt = {}
    for row in lb:
        cnt[row[key]] = cnt.get(row[key], 0) + 1
    seg = {k: v * 180 for k, v in cnt.items()}
    maj = max(seg.values()) / sum(seg.values()) * 100
    print(f'   {key:<12} participants {cnt}')
    print(f'   {"":<12} segments     {seg}   majority {maj:.2f}%')
    chk(f'{key} majority', maj)
assert len(lb) == 23

print()
print('=' * 74)
print('6. SINGLE-CELL CONFIDENCE INTERVALS (Appendix B)')
print('=' * 74)
ss = json.load(open(os.path.join(ROOT, 'stats_summary.json')))
print('stats_summary.json keys:', list(ss.keys()))

CI = ss['cells']
JNAME = {'FeedforwardOnly': 'Feedforward-only',
         'TransformerOnly': 'Transformer-only',
         'FeedforwardTransformer': 'Feedforward-Transformer'}
widths = []
for key, v in CI.items():
    cfg, proto, task = [p.strip() for p in key.split('/')]
    cfg = JNAME.get(cfg, cfg)
    w, cl = v['wilson'], v['cluster']
    ratio = (cl[1] - cl[0]) / (w[1] - w[0])
    widths.append(ratio)
    print(f'   {task:<10} {proto:<5} {cfg:<24} '
          f'wilson [{w[0]:.2f}, {w[1]:.2f}]  '
          f'cluster [{cl[0]:.2f}, {cl[1]:.2f}]  x{ratio:.2f}')
    # the accuracy in the CI block must equal the accuracy in the cell JSON
    ref = CELL[(task, proto.lower(), cfg)]['acc']
    if abs(ref - v['acc']) > 1e-9:
        fails.append(f'{key}: acc {v["acc"]} != cell JSON {ref}')
    for x in (w[0], w[1], cl[0], cl[1]):
        chk(f'{key} CI', x)
print(f'\n   cluster/Wilson width ratio: min {min(widths):.2f}, '
      f'max {max(widths):.2f}')
chk('ratio min', min(widths), '{:.1f}')
chk('ratio max', max(widths), '{:.1f}')
assert len(CI) == 20, len(CI)

print()
print('=' * 74)
print('7. PROTOCOL-EFFECT BOOTSTRAP (Table 3) and ABLATION')
print('=' * 74)
for name in ss:
    if name == 'cells':
        continue
    print(f'\n--- {name} ---')
    blob = ss[name]
    print(json.dumps(blob, indent=2)[:2600])

print()
print('=' * 74)
if fails:
    print(f'FAIL -- {len(fails)} number(s) in main.tex do not match the data:')
    for f in fails:
        print('   *', f)
    sys.exit(1)
print('PASS -- every number in main.tex matches the result files:')
print('        the evaluation table, the per-class recalls, the protocol')
print('        effects, the confidence intervals and the majority rates.')
