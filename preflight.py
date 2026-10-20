"""
preflight.py -- check everything before launching a long run.

A folder that looks right is not enough. This script actually loads the
data, extracts features, runs one forward pass of the model and verifies
that results can be written to disk. Each check is independent, so a
failure in one does not stop the rest from running.

    python -u preflight.py
"""
import os, sys, glob, time, shutil, traceback

OK, BAD = [], []


def check(name, fn):
    t0 = time.time()
    try:
        msg = fn()
        OK.append((name, msg, time.time() - t0))
        print(f"  [x] {name:<42} {msg}", flush=True)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        BAD.append((name, detail, traceback.format_exc()))
        print(f"  [!] {name:<42} {detail}", flush=True)


print("\n" + "=" * 78)
print("  PREFLIGHT -- Paper 2, laptop")
print("=" * 78 + "\n")

# ---------------------------------------------------------------- 1. python
check("python version", lambda: f"{sys.version.split()[0]}  ({sys.executable})")


# ---------------------------------------------------------------- 2. packages
def _pkgs():
    import numpy, sklearn, scipy, matplotlib, pywt, torch
    return (f"numpy {numpy.__version__} | sklearn {sklearn.__version__} | "
            f"scipy {scipy.__version__} | pywt {pywt.__version__} | "
            f"torch {torch.__version__}")


check("packages installed", _pkgs)


# ---------------------------------------------------------------- 3. common
def _common():
    import dasps_common as D
    missing = [n for n in ('RAW_DIR', 'PRE_DIR', 'RESULTS_ROOT', 'FS', 'SEG_LEN',
                           'BANDS', 'SEL_CH', 'CH_NAMES', 'N_CH', 'ASYM_PAIRS',
                           'WAVELET', 'WAV_LEVEL', 'HAM_A', 'MODELS')
               if not hasattr(D, n)]
    if missing:
        raise RuntimeError("dasps_common is missing these constants: " + ", ".join(missing))
    if D.RAW_DIR is None and D.PRE_DIR is None:
        raise RuntimeError("no data path resolved: none of the candidate locations exist")
    return f"{len(D.HAM_A)} participants | {D.N_CH} channels | models: {', '.join(D.MODELS)}"


check("dasps_common imports + constants", _common)

import dasps_common as D  # noqa: E402


# ---------------------------------------------------------------- 4. dataset
def _data_files():
    out = []
    for nm, p in (('raw', D.RAW_DIR), ('preprocessed', D.PRE_DIR)):
        if p and os.path.isdir(p):
            n = len(glob.glob(os.path.join(p, '*.mat')))
            out.append(f"{nm}: {n} .mat")
        else:
            out.append(f"{nm}: FOLDER NOT FOUND")
    if all('NOT FOUND' in o for o in out):
        raise RuntimeError(" / ".join(out))
    return " | ".join(out)


check("dataset folders", _data_files)


# ---------------------------------------------------------------- 5. load
_cache = {}


def _load(task):
    def inner():
        import numpy as np
        X, y, sids = D.load_data(task, verbose=False)
        _cache[task] = (X, y, sids)
        cnt = np.bincount(y)
        maj = cnt.max() / len(y) * 100
        return (f"X{X.shape} | {len(np.unique(sids))} subjects | "
                f"class counts {cnt.tolist()} | majority {maj:.2f}%")
    return inner


check("load_data('binary')", _load('binary'))
check("load_data('four_class')", _load('four_class'))


# ---------------------------------------------------------------- 6. expected shape
def _shape():
    # dasps_common.load_data returns (n_segments, SEG_LEN, N_CH): time first,
    # channel second, which is why extract_segment indexes with seg[:, c].
    X, y, sids = _cache['binary']
    if X.shape[0] != 4140:
        raise RuntimeError(f"{X.shape[0]} segments mile, 4140 expected (23 x 12 x 15)")
    if X.shape[1] != D.SEG_LEN:
        raise RuntimeError(f"segment length {X.shape[1]}, {D.SEG_LEN} expected "
                           f"(layout n x SEG_LEN x N_CH hona chahiye)")
    if X.shape[2] != D.N_CH:
        raise RuntimeError(f"got {X.shape[2]} channels, expected {D.N_CH}")
    return f"{X.shape[0]} x {X.shape[1]} x {X.shape[2]} -- as expected"


check("expected shape 4140 x 128 x 6", _shape)


# ---------------------------------------------------------------- 7. features
def _feat():
    import numpy as np
    X = _cache['binary'][0]
    v = D.extract_segment(X[0])
    if v.shape != (84,):
        raise RuntimeError(f"feature vector {v.shape}, (84,) expected")
    if not np.isfinite(v).all():
        raise RuntimeError("feature vector contains NaN or Inf")
    t0 = time.time()
    D.extract_all(X[:200], verbose=False)
    per = (time.time() - t0) / 200
    return f"84 dims, finite | ~{per*4140:.0f} s for the full dataset"


check("feature extraction (84 dims)", _feat)


# ---------------------------------------------------------------- 8. models
def _models():
    import torch
    out = []
    for nm, cls in D.MODELS.items():
        m = cls(n_features=84, n_classes=2)
        with torch.no_grad():
            o = m(torch.zeros(4, 84))
        if tuple(o.shape) != (4, 2):
            raise RuntimeError(f"{nm} ka output {tuple(o.shape)}, (4,2) expected")
        out.append(f"{nm} {sum(p.numel() for p in m.parameters()):,}p")
    return " | ".join(out)


check("all 3 models forward pass", _models)


# ---------------------------------------------------------------- 9. threads
def _threads():
    import torch
    n = torch.get_num_threads()
    env = os.environ.get('TORCH_THREADS', '(unset)')
    if n != 1:
        raise RuntimeError(f"torch is using {n} threads; set TORCH_THREADS=1 "
                           f"before launching the run.")
    return f"1 thread (TORCH_THREADS={env})"


check("torch thread pinning", _threads)


# ---------------------------------------------------------------- 10. results dir
def _results():
    os.makedirs(D.RESULTS_ROOT, exist_ok=True)
    p = os.path.join(D.RESULTS_ROOT, '.write_test')
    open(p, 'w').write('ok'); os.remove(p)
    free = shutil.disk_usage(D.RESULTS_ROOT).free / 1e9
    if free < 5:
        raise RuntimeError(f"sirf {free:.1f} GB khali hai -- 5 GB se kam")
    return f"likha ja sakta hai | {free:.0f} GB khali"


check("results folder writable + disk space", _results)


# ---------------------------------------------------------------- 11. done so far
def _done():
    import json
    n = 0
    for f in glob.glob(os.path.join(D.RESULTS_ROOT, '**', '*.json'), recursive=True):
        if os.path.basename(f) in ('label_balance.json', 'identity_results.json'):
            continue
        try:
            r = json.load(open(f))
            if isinstance(r, dict) and 'acc' in r:
                n += 1
        except Exception:
            pass
    return f"{n} result file(s) already present (LOSO column 10 honi chahiye)"


check("results already present", _done)


# ---------------------------------------------------------------- summary
print("\n" + "=" * 78)
if BAD:
    print(f"  {len(BAD)} CHECK(S) FAILED -- fix these before launching a run:")
    print("=" * 78)
    for name, detail, tb in BAD:
        print(f"\n  >>> {name}\n      {detail}")
        print("      " + tb.strip().splitlines()[-2].strip())
    print("\n  Poora output mujhe bhej do.\n")
    sys.exit(1)
print("  ALL CHECKS PASSED -- the pipeline is ready to run.")
print("=" * 78 + "\n")
