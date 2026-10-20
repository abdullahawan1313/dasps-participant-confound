"""
train_model.py -- leakage-free training for the DASPS anxiety classifier.
========================================================================

This script replaces an earlier implementation that contained two coupled
faults in its training loop. Both are documented here because the paper
quantifies their effect.

  1. Oracle epoch selection (removed).
     The earlier code predicted the held-out test segment after every epoch
     and kept the prediction whenever it matched the held-out label
     (`local_acc = float(p == yte[0])`). The reported epoch was therefore
     chosen using the test label. Here the held-out segment is predicted
     exactly once, after training has finished.

  2. The `best_pred = 0` fallback (removed).
     In the earlier code, a segment the model never classified correctly at
     any epoch was recorded as class 0. This produced the characteristic
     signature of exactly 100% recall on the first class with all errors
     confined to a single column.

Other properties of this implementation:

  * Checkpoint and resume. A leave-one-out run is 4,140 folds and takes
    tens of hours. A checkpoint is written atomically every `--ckpt-every`
    folds, and re-running the same command continues from where it stopped.

  * Deterministic save folder (no timestamp). One cell writes into one
    folder, so a re-run replaces the previous result instead of leaving two
    result files for the same cell.

  * Per-fold seeding (`seed + fold_index`), which makes a resumed run
    bit-identical to an uninterrupted one.

  * No early stopping and no validation split, so no quantity derived from
    the test labels can influence the model.

Usage:
  python -u train_model.py --task binary     --model FeedforwardTransformer --epochs 100
  python -u train_model.py --task four_class --model TransformerOnly        --epochs 200
  python -u train_model.py --task binary     --epochs 100 --protocol loso
"""
import os, time, argparse, json
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix, precision_score, recall_score)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dasps_common import (RESULTS_ROOT, set_seed, load_data, extract_all, DEVICE, PAPER,
                          CLASS_NAMES, MODELS)


# ----------------------------------------------------------------------
def train_one_fold(Xtr, ytr, Xte, n_classes, model_name, epochs,
                   lr=1e-3, weight_decay=1e-4, batch_size=256,
                   label_smoothing=0.05, collect_loss=False):
    """Train one fold and return a single prediction for the held-out sample(s).

    The test labels are never passed into this function, so selecting an
    epoch or a threshold using them is structurally impossible."""
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    w = torch.FloatTensor(1.0 / (counts + 1e-6)).to(DEVICE)
    w = w / w.sum() * n_classes

    Xtr_t = torch.FloatTensor(Xtr).to(DEVICE)
    ytr_t = torch.LongTensor(ytr).to(DEVICE)
    Xte_t = torch.FloatTensor(Xte).to(DEVICE)

    ntr = Xtr_t.shape[0]
    model = MODELS[model_name](n_features=Xtr.shape[1], n_classes=n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)

    ep_losses = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(ntr, device=DEVICE)
        tot, nb = 0.0, 0
        for st in range(0, ntr, batch_size):
            idx = perm[st:st + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if collect_loss:            # .item() syncs; only first 10 folds
                tot += loss.item(); nb += 1
        sched.step()
        if collect_loss:
            ep_losses.append(tot / max(nb, 1))

    # The held-out segment is predicted once, here, after training has ended.
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    return pred, ep_losses


# ----------------------------------------------------------------------
def _save_ckpt(path, preds, done, total, losses):
    """Write atomically: save to .tmp.npz first, then rename into place.

    If the process dies mid-write, the old checkpoint survives intact.

    The temporary name must END in .npz: np.savez appends '.npz' otherwise,
    and the subsequent os.replace would look for a file that does not exist.

    Windows also refuses to replace a file another handle still holds, which
    happens if another process is reading the checkpoint at that moment, so
    the rename is retried a few times."""
    tmp = path + '.tmp.npz'
    L = np.asarray(losses, dtype=np.float64) if losses else np.zeros((0, 0))
    np.savez(tmp, preds=preds, done=np.int64(done), total=np.int64(total), losses=L)
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5)
    raise PermissionError(f"checkpoint replace 6 koshishon ke baad bhi nakaam: {path}")


def run(Xf, y, sids, task, model_name, epochs, protocol, sd,
        max_folds=None, seed=42, ckpt_every=25):
    n = len(y)
    n_classes = len(np.unique(y))
    preds = np.full(n, -1, dtype=int)
    fold_losses = []
    iters, accs = [], []

    if protocol == 'loo':
        splits = list(LeaveOneOut().split(Xf))
    else:                                    # loso
        splits = [(np.where(sids != s)[0], np.where(sids == s)[0])
                  for s in np.unique(sids)]

    if max_folds and max_folds < len(splits):
        # The data is in subject order, so the first N folds would otherwise
        # contain a single class only (the first Light segment is at index
        # 1260). Take a stratified subset instead.
        rng = np.random.RandomState(0)
        per = max(1, max_folds // len(np.unique(y)))
        pick = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            pick.extend(rng.choice(idx, size=min(per, len(idx)), replace=False))
        pick = set(int(i) for i in pick)
        splits = [(tr, te) for (tr, te) in splits if int(te[0]) in pick]
        print(f"  [smoke test] {len(splits)} folds, stratified across all classes")

    total = len(splits)
    report_every = 1 if protocol == 'loso' else max(1, total // 100)

    # ---------------- FIX 7: resume ----------------
    ckpt = os.path.join(sd, 'checkpoint.npz')
    start = 0
    if os.path.exists(ckpt):
        try:
            # The context manager matters: np.load keeps the file handle
            # open, and Windows will not replace a file that is still open.
            with np.load(ckpt) as z:
                ck_total = int(z['total'])
                if ck_total == total:
                    preds = z['preds'].copy()
                    start = int(z['done'])
                    L = z['losses']
                    fold_losses = [list(r) for r in L] if L.size else []
            if ck_total == total:
                print(f"  >>> RESUMING: {start}/{total} folds already done. <<<",
                      flush=True)
            else:
                print(f"  (checkpoint holds {ck_total} folds, this run needs {total} "
                      f"ka -- checkpoint ignore kiya ja raha hai)", flush=True)
        except Exception as e:
            print(f"  (checkpoint unreadable: {e} -- starting from scratch)",
                  flush=True)

    if start >= total:
        print("  All folds already complete; writing the results file only.",
              flush=True)

    print(f"  {protocol.upper()} | {model_name} | {task} | {total} folds | {epochs} epochs",
          flush=True)

    t0 = time.time()
    for i in range(start, total):
        tr, te = splits[i]
        set_seed(seed + i)                       # FIX 9: resume bit-identical
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xf[tr])
        Xte = sc.transform(Xf[te])
        p, losses = train_one_fold(Xtr, y[tr], Xte, n_classes, model_name,
                                   epochs, collect_loss=(i < 10))
        preds[te] = p
        if i < 10 and losses:
            fold_losses.append(losses)

        if (i + 1) % ckpt_every == 0 or (i + 1) == total:
            # A failed checkpoint write is not a reason to kill a long run.
            try:
                _save_ckpt(ckpt, preds, i + 1, total, fold_losses)
            except Exception as e:
                print(f"    [warn] checkpoint save failed ({e}) -- continuing",
                      flush=True)

        if (i + 1) % report_every == 0:
            done = preds >= 0
            a = accuracy_score(y[done], preds[done]) * 100
            el = time.time() - t0
            per_fold = el / max(i + 1 - start, 1)
            iters.append(int(done.sum())); accs.append(a)
            print(f"    fold {i+1}/{total} | running acc={a:.2f}% "
                  f"| {per_fold:.1f} s/fold | ETA {per_fold*(total-i-1)/3600:.1f} h",
                  flush=True)

    done = preds >= 0
    yt, yp = y[done], preds[done]
    res = {
        'task': task, 'model': model_name, 'protocol': protocol, 'epochs': epochs,
        'n': int(done.sum()),
        'acc':       accuracy_score(yt, yp) * 100,
        'f1':        f1_score(yt, yp, average='weighted'),
        'kappa':     cohen_kappa_score(yt, yp),
        'precision': precision_score(yt, yp, average='weighted', zero_division=0),
        'recall':    recall_score(yt, yp, average='weighted', zero_division=0),
        'time_min':  (time.time() - t0) / 60,       # is session ka waqt
        'resumed_from': start,
    }
    iters.append(int(done.sum())); accs.append(res['acc'])
    return res, yt, yp, confusion_matrix(yt, yp), fold_losses, iters, accs, sids[done], ckpt


# ----------------------------------------------------------------------
def save_all(res, yt, yp, cm, fold_losses, iters, accs, sd, tag, task, sids_eval=None):
    names = CLASS_NAMES[task]
    np.save(os.path.join(sd, f'y_true_{tag}.npy'), yt)
    np.save(os.path.join(sd, f'y_pred_{tag}.npy'), yp)   # for McNemar and the bootstrap intervals
    if sids_eval is not None:
        np.save(os.path.join(sd, 'subject_ids.npy'), sids_eval)

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), max(5, len(names) * 2)))
    im = ax.imshow(cm, cmap='Blues'); plt.colorbar(im)
    ax.set(xticks=range(len(names)), yticks=range(len(names)),
           xticklabels=names, yticklabels=names,
           ylabel='True label', xlabel='Predicted label',
           title=f'Confusion matrix -- {tag}\nAcc = {res["acc"]:.2f}%')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    th = cm.max() / 2.
    for a in range(cm.shape[0]):
        for b in range(cm.shape[1]):
            ax.text(b, a, str(cm[a, b]), ha='center', va='center',
                    color='white' if cm[a, b] > th else 'black', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(sd, f'confusion_matrix_{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    if len(iters) > 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(iters, accs, 'b-o', markersize=4, label=tag)
        # Reference line only: the previously reported figure, plotted for
        # comparison. It is not a result of this run.
        ax.axhline(PAPER[task]['acc'], color='r', ls='--',
                   label=f"as submitted, disputed ({PAPER[task]['acc']}%)")
        ax.set(title=f'Running accuracy -- {tag}', xlabel='Folds completed',
               ylabel='Accuracy (%)')
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(sd, f'progress_{tag}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    if fold_losses:
        fig, ax = plt.subplots(figsize=(12, 5))
        for idx, l in enumerate(fold_losses):
            ax.plot(l, alpha=.6, label=f'Fold {idx+1}')
        ax.set(title=f'Training loss, first 10 folds -- {tag}',
               xlabel='Epoch', ylabel='Loss')
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(sd, f'training_curves_{tag}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    with open(os.path.join(sd, f'results_{tag}.json'), 'w') as f:
        json.dump({**res, 'confusion_matrix': cm.tolist()}, f, indent=2)

    print(f"\n  {'='*54}")
    print(f"  {tag}")
    print(f"  {'='*54}")
    print(f"  Accuracy  : {res['acc']:.2f}%")
    print(f"  F1        : {res['f1']:.4f}")
    print(f"  Kappa     : {res['kappa']:.4f}")
    print(f"  Precision : {res['precision']:.4f}")
    print(f"  Recall    : {res['recall']:.4f}")
    print(f"\n  Per-class recall:")
    for c, nm in enumerate(names):
        tot = cm[c].sum()
        print(f"    {nm:<10}: {cm[c,c]}/{tot} = {cm[c,c]/tot*100:.1f}%")
    print("\n  >>> A class at exactly 100.0% recall indicates label leakage. <<<")
    print(f"  {'='*54}\n")


# ----------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='binary', choices=['binary', 'four_class'])
    ap.add_argument('--model', default='FeedforwardTransformer', choices=list(MODELS))
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--protocol', default='loo', choices=['loo', 'loso'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-folds', type=int, default=None,
                    help='short smoke test, e.g. 300 folds')
    ap.add_argument('--ckpt-every', type=int, default=25,
                    help='itne folds ke baad checkpoint likho')
    ap.add_argument('--force', action='store_true',
                    help='recompute a cell even if it is already complete')
    args = ap.parse_args()

    set_seed(args.seed)
    # Deterministic output folder: one cell, one folder; a re-run replaces it.
    # A --max-folds run writes into a _SMOKE folder so that a short test can
    # never overwrite a full 4,140-fold cell. Reporting scripts skip _SMOKE.
    suffix = '' if args.max_folds is None else f'_SMOKE{args.max_folds}'
    sd = os.path.join(RESULTS_ROOT, f'{args.protocol}_{args.task}_{args.model}{suffix}')
    os.makedirs(sd, exist_ok=True)
    tag = f'{args.model}_{args.task}_{args.protocol}'

    # If the cell is already complete, do not recompute it. This makes the
    # whole grid safe to re-launch after an interruption.
    _done_json = os.path.join(sd, f'results_{tag}.json')
    if os.path.exists(_done_json) and not args.force:
        print(f"SKIP: {tag} is already complete.\n  {_done_json}\n"
              f"  Pass --force to recompute it.", flush=True)
        raise SystemExit(0)

    print(f"Device: {DEVICE}\nSave dir: {sd}\nSeed: {args.seed}\n", flush=True)

    X, y, sids = load_data(args.task)
    Xf = extract_all(X)

    res, yt, yp, cm, fl, it, ac, se, ckpt = run(
        Xf, y, sids, args.task, args.model, args.epochs, args.protocol, sd,
        max_folds=args.max_folds, seed=args.seed, ckpt_every=args.ckpt_every)

    save_all(res, yt, yp, cm, fl, it, ac, sd, tag, args.task, se)

    # Results are safely written; the checkpoint is no longer needed.
    if os.path.exists(ckpt):
        try: os.remove(ckpt)
        except OSError: pass
    print(f"Saved to: {sd}")
