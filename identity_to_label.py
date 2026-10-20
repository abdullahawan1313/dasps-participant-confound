"""Identity control: segment -> participant identity -> HAM-A label lookup.
   Step (ii) of the diagnostic protocol. CPU only, no torch."""
import os, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from dasps_common import (set_seed, load_data, extract_all, HAM_A,
                          ham_to_label, RESULTS_ROOT)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='binary', choices=['binary', 'four_class'])
    ap.add_argument('--per-subject', type=int, default=22, help='folds sampled per participant')
    a = ap.parse_args()

    set_seed(42)
    X, y, sids = load_data(a.task)
    Xf = extract_all(X)
    n = len(sids)
    LOOKUP = {s: ham_to_label(HAM_A[s], a.task) for s in range(1, 24)}
    maj_id = np.bincount(sids).max() / n * 100
    maj_lab = np.bincount(y).max() / n * 100

    rng = np.random.RandomState(0); pick = []
    for s in np.unique(sids):
        pick.extend(rng.choice(np.where(sids == s)[0], a.per_subject, replace=False))
    pick = np.array(sorted(pick))
    print(f"\n  Task: {a.task} | folds: {len(pick)} | "
          f"ID chance = {maj_id:.2f}% | label majority = {maj_lab:.2f}%\n")

    sd = os.path.join(RESULTS_ROOT, f'identity_{a.task}'); os.makedirs(sd, exist_ok=True)
    out = {}
    for name, mk in [('LogisticRegression', lambda: LogisticRegression(max_iter=2000, n_jobs=-1)),
                     ('RandomForest',       lambda: RandomForestClassifier(
                                                n_estimators=300, random_state=42, n_jobs=-1))]:
        id_pred = np.zeros(len(pick), int)
        for k, i in enumerate(pick):
            tr = np.ones(n, bool); tr[i] = False
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xf[tr]); Xte = sc.transform(Xf[i:i+1])
            m = mk(); m.fit(Xtr, sids[tr]); id_pred[k] = m.predict(Xte)[0]
            if (k + 1) % 100 == 0:
                print(f"    {name}: {k+1}/{len(pick)}", flush=True)
        id_acc = accuracy_score(sids[pick], id_pred) * 100
        lab_via = np.array([LOOKUP[p] for p in id_pred])
        lab_acc = accuracy_score(y[pick], lab_via) * 100
        out[name] = {'participant_id_acc': id_acc, 'label_via_id_acc': lab_acc,
                     'id_chance': maj_id, 'label_majority': maj_lab}
        np.save(f'{sd}/id_pred_{name}.npy', id_pred)
        np.save(f'{sd}/label_via_id_{name}.npy', lab_via)
        print(f"\n  {'='*60}\n  {name}  ({a.task})\n  {'='*60}")
        print(f"  Participant-ID accuracy    : {id_acc:6.2f}%   (chance {maj_id:.2f}%)")
        print(f"  Label via ID lookup        : {lab_acc:6.2f}%   (majority {maj_lab:.2f}%)")
        print(f"  {'='*60}\n")
    np.save(f'{sd}/y_true.npy', y[pick]); np.save(f'{sd}/subject_ids.npy', sids[pick])
    json.dump(out, open(f'{sd}/identity_results.json', 'w'), indent=2)
    print(f"Saved: {sd}")
