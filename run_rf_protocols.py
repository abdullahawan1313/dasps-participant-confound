"""Random-forest configurations under both protocols (CPU only, no torch).

   Baseline RF (24 descriptors + recursive feature elimination) and
   RF + 84 features.

     * Checkpoint and resume: a leave-one-out run is 4,140 folds, so a crash
       or reboot does not lose the work; re-run the same command.
     * `--n-jobs` defaults to 2 rather than -1, so a run does not take every
       core and starve anything else on the machine.
     * A configuration that is already complete is skipped on a re-run.
"""
import os, json, argparse, time, warnings
warnings.filterwarnings('ignore')          # sklearn/joblib parallel notices flood the log
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix, precision_score, recall_score)
from dasps_common import set_seed, load_data, extract_all, RESULTS_ROOT, CLASS_NAMES

BASE = slice(60, 84)          # the 24 baseline-family descriptors


def _save_ckpt(path, preds, done, total):
    # The temporary name must end in .npz: np.savez appends '.npz' otherwise,
    # and the subsequent os.replace would then look for a file that does not
    # exist. Windows also refuses to replace a file that is still open, so the
    # rename is retried.
    tmp = path + '.tmp.npz'
    np.savez(tmp, preds=preds, done=np.int64(done), total=np.int64(total))
    for _ in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5)
    raise PermissionError(f"checkpoint replace nakaam: {path}")


def run(Xf, y, sids, protocol, rfe_n=None, seed=42, max_folds=None, tag='',
        ckpt=None, ckpt_every=25, n_jobs=2):
    n = len(y); preds = np.full(n, -1, int)
    if protocol == 'loo':
        splits = list(LeaveOneOut().split(Xf))
    else:
        splits = [(np.where(sids != s)[0], np.where(sids == s)[0]) for s in np.unique(sids)]
    if max_folds and max_folds < len(splits):
        rng = np.random.RandomState(0); pick = set()
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            k = max(1, max_folds // len(np.unique(y)))
            pick |= set(int(i) for i in rng.choice(idx, min(k, len(idx)), replace=False))
        splits = [(tr, te) for tr, te in splits if int(te[0]) in pick]
        print(f"    [subsample] {len(splits)} stratified folds")

    total = len(splits)
    every = 1 if protocol == 'loso' else max(1, total // 100)

    start = 0
    if ckpt and os.path.exists(ckpt):
        try:
            with np.load(ckpt) as z:          # handle band karna zaroori (WinError 5)
                ok = int(z['total']) == total
                if ok:
                    preds = z['preds'].copy(); start = int(z['done'])
            if ok:
                print(f"    >>> RESUMING {tag}: {start}/{total} folds done <<<", flush=True)
        except Exception as e:
            print(f"    (checkpoint unreadable: {e})", flush=True)

    t0 = time.time()
    for i in range(start, total):
        tr, te = splits[i]
        sc = StandardScaler(); Xtr = sc.fit_transform(Xf[tr]); Xte = sc.transform(Xf[te])
        if rfe_n:
            sel = RFE(RandomForestClassifier(n_estimators=100, random_state=seed,
                                             n_jobs=n_jobs),
                      n_features_to_select=rfe_n)
            Xtr = sel.fit_transform(Xtr, y[tr]); Xte = sel.transform(Xte)
        clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=n_jobs)
        clf.fit(Xtr, y[tr]); preds[te] = clf.predict(Xte)

        if ckpt and ((i + 1) % ckpt_every == 0 or (i + 1) == total):
            try:
                _save_ckpt(ckpt, preds, i + 1, total)
            except Exception as e:
                print(f"    [warn] checkpoint save failed ({e}) -- continuing", flush=True)
        if (i + 1) % every == 0:
            el = time.time() - t0
            pf = el / max(i + 1 - start, 1)
            print(f"    {tag} fold {i+1}/{total} | {pf:.2f} s/fold "
                  f"| ETA {pf*(total-i-1)/3600:.1f} h", flush=True)
    d = preds >= 0
    return y[d], preds[d], sids[d]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='binary', choices=['binary', 'four_class'])
    ap.add_argument('--protocol', default='loso', choices=['loo', 'loso'])
    ap.add_argument('--n-features', type=int, default=10)
    ap.add_argument('--max-folds', type=int, default=None)
    ap.add_argument('--n-jobs', type=int, default=2,
                    help='cores per random forest; 2 leaves headroom for other work.')
    ap.add_argument('--ckpt-every', type=int, default=25)
    a = ap.parse_args()

    set_seed(42)
    X, y, sids = load_data(a.task)
    Xf = extract_all(X)
    names = CLASS_NAMES[a.task]
    maj = np.bincount(y).max() / len(y) * 100
    _sfx = '' if a.max_folds is None else f'_SMOKE{a.max_folds}'
    sd = os.path.join(RESULTS_ROOT, f'rf_{a.protocol}_{a.task}{_sfx}'); os.makedirs(sd, exist_ok=True)
    print(f"\n  Majority-class baseline = {maj:.2f}%   (n_jobs={a.n_jobs})\n", flush=True)

    for label, Xin, rfe in [(f"Baseline RF (24 feats + RFE-{a.n_features})", Xf[:, BASE], a.n_features),
                            ("RF + 84 features",                            Xf,          None)]:
        tag = label.split('(')[0].strip().replace(' ', '_').replace('+', 'plus')
        jpath = f'{sd}/results_{tag}.json'
        if os.path.exists(jpath):
            print(f"  SKIP (already complete): {label}\n", flush=True)
            continue
        print(f"  Running: {label} | {a.protocol}", flush=True)
        ckpt = f'{sd}/checkpoint_{tag}.npz'
        yt, yp, ss = run(Xin, y, sids, a.protocol, rfe, max_folds=a.max_folds,
                         tag=label[:12], ckpt=ckpt, ckpt_every=a.ckpt_every,
                         n_jobs=a.n_jobs)
        cm = confusion_matrix(yt, yp)
        np.save(f'{sd}/y_true_{tag}.npy', yt); np.save(f'{sd}/y_pred_{tag}.npy', yp)
        np.save(f'{sd}/subject_ids.npy', ss)
        res = {'label': label, 'task': a.task, 'protocol': a.protocol, 'n': int(len(yt)),
               'acc': accuracy_score(yt, yp) * 100,
               'f1': f1_score(yt, yp, average='weighted'),
               'kappa': cohen_kappa_score(yt, yp),
               'precision': precision_score(yt, yp, average='weighted', zero_division=0),
               'recall': recall_score(yt, yp, average='weighted', zero_division=0),
               'confusion_matrix': cm.tolist(), 'majority': maj}
        json.dump(res, open(jpath, 'w'), indent=2)
        if os.path.exists(ckpt):
            try: os.remove(ckpt)
            except OSError: pass
        print(f"\n  {'='*56}\n  {label}  [{a.protocol.upper()}]\n  {'='*56}")
        print(f"  Accuracy : {res['acc']:.2f}%   (majority {maj:.2f}%)")
        print(f"  F1       : {res['f1']:.4f}")
        print(f"  Kappa    : {res['kappa']:.4f}")
        for c, nm in enumerate(names):
            t = cm[c].sum()
            print(f"    {nm:<9}: {cm[c,c]}/{t} = {cm[c,c]/t*100:.1f}%")
        print(f"  {'='*56}\n", flush=True)
    print(f"Saved: {sd}")
