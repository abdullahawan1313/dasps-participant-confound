"""Step (i) of the diagnostic protocol: per-participant label balance.
   Shows that each participant carries exactly one label, which is what
   makes participant identity sufficient to recover the class. Instant."""
import os, json
import numpy as np
from dasps_common import HAM_A, ham_to_label, load_data, RESULTS_ROOT, CLASS_NAMES

if __name__ == '__main__':
    rows = []
    for sid in range(1, 24):
        rows.append(dict(participant=sid, ham_a=HAM_A[sid],
                         binary=CLASS_NAMES['binary'][ham_to_label(HAM_A[sid], 'binary')],
                         four_class=CLASS_NAMES['four_class'][ham_to_label(HAM_A[sid], 'four_class')]))
    print(f"\n{'P':>3} {'HAM-A':>6}  {'Binary':<8} {'Four-class':<10}  Segments")
    print('-' * 48)
    for r in rows:
        print(f"{r['participant']:>3} {r['ham_a']:>6}  {r['binary']:<8} {r['four_class']:<10}  180")
    print('-' * 48)

    for task in ['binary', 'four_class']:
        lab = [ham_to_label(HAM_A[s], task) for s in range(1, 24)]
        u, c = np.unique(lab, return_counts=True)
        print(f"\n{task}: participants per class")
        for cls, cnt in zip(u, c):
            print(f"   {CLASS_NAMES[task][cls]:<10}: {cnt:2d} participants = {cnt*180:5d} segments "
                  f"({cnt*180/4140*100:5.2f}%)")
    print("\nEvery participant has exactly ONE label for all 180 of their segments.")
    print("Recovering participant identity is therefore sufficient to recover the label.\n")

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    json.dump(rows, open(os.path.join(RESULTS_ROOT, 'label_balance.json'), 'w'), indent=2)
    print(f"Saved: {os.path.join(RESULTS_ROOT, 'label_balance.json')}")
