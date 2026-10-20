"""Reads every results_*.json under RESULTS_ROOT and prints one table.
   Run it at any time to see the state of the whole grid."""
import os, json, glob
from dasps_common import RESULTS_ROOT

MAJ = {'binary': 69.57, 'four_class': 56.52}
PUB = {'binary': (94.90, 0.88), 'four_class': (92.74, 0.86)}
ORDER = ['Baseline RF', 'RF + 84', 'FeedforwardOnly', 'TransformerOnly', 'FeedforwardTransformer']

def tidy(name):
    n = name.replace('_', ' ')
    if 'Baseline' in n: return 'Baseline RF'
    if '84' in n:       return 'RF + 84'
    return name

def main():
    rows = []
    for f in glob.glob(os.path.join(RESULTS_ROOT, '**', '*.json'), recursive=True):
        if 'label_balance' in f or 'identity_results' in f: continue
        try: r = json.load(open(f))
        except Exception: continue
        if not isinstance(r, dict) or 'acc' not in r: continue
        model = r.get('model') or tidy(r.get('label', os.path.basename(f)))
        rows.append(dict(task=r.get('task', '?'), proto=r.get('protocol', '?'),
                         model=tidy(model), acc=r['acc'], f1=r.get('f1'),
                         kappa=r.get('kappa'), n=r.get('n'), epochs=r.get('epochs')))
    if not rows:
        print(f"No results found under {RESULTS_ROOT}"); return

    for task in ('binary', 'four_class'):
        sub = [r for r in rows if r['task'] == task]
        if not sub: continue
        pa, pk = PUB[task]
        print(f"\n{'='*86}\n  {task.upper().replace('_',' ')} TASK      "
              f"majority class = {MAJ[task]:.2f}%      published baseline = {pa}% (kappa {pk})\n{'='*86}")
        print(f"  {'Configuration':<26}{'Protocol':<10}{'Folds':>7}{'Ep':>5}{'Acc %':>9}{'F1':>9}{'Kappa':>9}")
        print('  ' + '-' * 82)
        for proto in ('loo', 'loso'):
            got = [r for r in sub if r['proto'] == proto]
            if not got: continue
            got.sort(key=lambda r: ORDER.index(r['model']) if r['model'] in ORDER else 99)
            for r in got:
                ep = r['epochs'] if r['epochs'] else '-'
                f1 = f"{r['f1']:.4f}" if r['f1'] is not None else '-'
                kp = f"{r['kappa']:+.4f}" if r['kappa'] is not None else '-'
                print(f"  {r['model']:<26}{proto.upper():<10}{r['n']:>7}{str(ep):>5}"
                      f"{r['acc']:>9.2f}{f1:>9}{kp:>9}")
            print()

    # identity control
    for task in ('binary', 'four_class'):
        f = os.path.join(RESULTS_ROOT, f'identity_{task}', 'identity_results.json')
        if not os.path.exists(f): continue
        d = json.load(open(f))
        print(f"{'='*86}\n  IDENTITY CONTROL -- {task}\n{'='*86}")
        print(f"  {'Classifier':<24}{'Participant-ID %':>18}{'Label via lookup %':>22}")
        print('  ' + '-' * 82)
        any_k = next(iter(d))
        print(f"  {'chance / majority':<24}{d[any_k]['id_chance']:>18.2f}{d[any_k]['label_majority']:>22.2f}")
        for k, v in d.items():
            print(f"  {k:<24}{v['participant_id_acc']:>18.2f}{v['label_via_id_acc']:>22.2f}")
        print()

    missing = []
    for task in ('binary', 'four_class'):
        for proto in ('loo', 'loso'):
            for m in ORDER:
                if not any(r['task'] == task and r['proto'] == proto and r['model'] == m for r in rows):
                    missing.append(f"{m} / {proto.upper()} / {task}")
    print(f"{'='*86}\n  STILL MISSING ({len(missing)} of 20)\n{'='*86}")
    for m in missing: print(f"    - {m}")
    print()

if __name__ == '__main__':
    main()
