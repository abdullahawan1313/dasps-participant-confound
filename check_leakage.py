"""
check_leakage.py -- per-class recall for every completed cell, with alarms.

    python check_leakage.py

The defect described in the paper left a specific signature: exactly 100.0%
recall on one class, with all errors confined to a single column, because a
`best_pred = 0` fallback assigned every never-correct segment to class 0.

This script reads the confusion matrix from each results_*.json and

  * prints per-class recall,
  * raises an ALARM on 100.0% recall,
  * raises an ALARM when one column holds 90% or more of the predictions.

It states plainly when nothing is found.
"""
import os, json, glob
import numpy as np
from dasps_common import RESULTS_ROOT, CLASS_NAMES

ORDER = ['Baseline RF', 'RF + 84', 'FeedforwardOnly', 'TransformerOnly',
         'FeedforwardTransformer']


def tidy(n):
    n = str(n)
    if 'Baseline' in n.replace('_', ' '): return 'Baseline RF'
    if '84' in n: return 'RF + 84'
    return n


def main():
    cells = {}
    for f in glob.glob(os.path.join(RESULTS_ROOT, '**', '*.json'), recursive=True):
        if os.path.basename(f) in ('label_balance.json', 'identity_results.json',
                                   'stats_summary.json'):
            continue
        if '_SMOKE' in f:
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if not isinstance(r, dict) or 'confusion_matrix' not in r:
            continue
        k = (r.get('task', '?'), r.get('protocol', '?'),
             tidy(r.get('model') or r.get('label') or os.path.basename(f)))
        mt = os.path.getmtime(f)
        if k in cells and cells[k][1] >= mt:
            continue
        cells[k] = (r, mt)

    if not cells:
        print(f'No completed cells found under: {RESULTS_ROOT}')
        return

    alarms = []
    for task in ('binary', 'four_class'):
        names = CLASS_NAMES[task]
        for proto in ('loo', 'loso'):
            got = [(m, cells[(task, proto, m)][0]) for m in ORDER
                   if (task, proto, m) in cells]
            if not got:
                continue
            print(f'\n{"="*78}\n  {task.upper().replace("_"," ")}  /  {proto.upper()}\n{"="*78}')
            head = ''.join(f'{n:>12}' for n in names)
            print(f'  {"Configuration":<26}{"Acc %":>8}{head}')
            print('  ' + '-' * (34 + 12 * len(names)))
            for m, r in got:
                cm = np.asarray(r['confusion_matrix'], dtype=float)
                rec = np.diag(cm) / np.maximum(cm.sum(1), 1) * 100
                col = cm.sum(0) / max(cm.sum(), 1) * 100       # predictions per class
                cells_txt = ''
                for c, v in enumerate(rec):
                    mark = '!' if v >= 99.95 else ' '
                    cells_txt += f'{v:>11.1f}{mark}'
                    if v >= 99.95:
                        alarms.append(f'{m} / {proto.upper()} / {task}: '
                                      f'class "{names[c]}" recall {v:.2f}%')
                print(f'  {m:<26}{r["acc"]:>8.2f}{cells_txt}')
                if col.max() >= 90:
                    j = int(col.argmax())
                    alarms.append(f'{m} / {proto.upper()} / {task}: '
                                  f'{col[j]:.1f}% of ALL predictions went to '
                                  f'"{names[j]}"')

    print(f'\n{"="*78}')
    if alarms:
        print(f'  ALARM -- {len(alarms)} finding(s) consistent with leakage:')
        print('=' * 78)
        for a in alarms:
            print(f'    !! {a}')
        print('\n  Inspect these cells before reporting any of these results.\n')
    else:
        print('  CLEAN -- no class at 100% recall, no single-column pile-up.')
        print('  The leakage signature described in the paper is absent.')
        print('=' * 78 + '\n')


if __name__ == '__main__':
    main()
