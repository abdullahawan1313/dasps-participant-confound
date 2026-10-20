"""
dasps_common.py -- shared data loading, feature extraction and models.
======================================================================
Single source of truth for the data, the features and the model
definitions, so that every script in this repository sees exactly the
same inputs. The order of the feature families is preserved from the
original implementation so that results remain directly comparable.

84-D feature vector composition (verified against this code):
    Differential entropy    6 ch x 2 bands              = 12
    Hjorth (act,mob,comp)   6 ch x 3                    = 18
    Wavelet sub-band energy 6 ch x 5 (db4, level 4)     = 30
    Mean power              6 ch x 2 bands              = 12
    Rational asymmetry      3 pairs x 2 bands           =  6
    Asymmetry index         3 pairs x 2 bands           =  6
                                                   TOTAL = 84
Bands: theta [5, 8) Hz, beta [13, 30) Hz.
Asymmetry pairs: (AF3,AF4), (FC5,FC6), (P7,P8).
"""
import os, random, warnings
import numpy as np, h5py, pywt
# torch is needed only for the neural configurations. The analysis and
# figure scripts work on the released predictions without it, so the import
# is optional and failures are deferred to the point of use.
try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:                                   # pragma: no cover
    torch = None
    HAVE_TORCH = False

    class _TorchMissing:
        """Stand-in for torch.nn so the model classes can still be defined.

        Any real use raises a clear error instead of a confusing one.
        """

        class Module:
            def __init__(self, *a, **k):
                raise ImportError(
                    'PyTorch is required for the neural configurations. '
                    'Install it with:  pip install torch')

        def __getattr__(self, name):
            raise ImportError(
                'PyTorch is required for the neural configurations. '
                'Install it with:  pip install torch')

    nn = _TorchMissing()

# ---- threading ----------------------------------------------------------
# This model is small enough that torch's intra-op threading HURTS.
# Measured on the project laptop (4 logical / 2 physical cores):
#     1 thread 0.553 s/epoch | 2 -> 0.574 | 3 -> 0.654 | 4 -> 0.788
# So we pin to ONE thread, which also lets several runs go in parallel,
# one per core, for roughly 2x aggregate throughput.
# Override with:  $env:TORCH_THREADS="2"
# On a GPU the work happens on the device, so the single-thread default
# buys nothing there; override it with TORCH_THREADS.
if HAVE_TORCH:
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '1')))

from numpy.fft import rfft, rfftfreq
warnings.filterwarnings('ignore')

# ---------------- paths ----------------
# The data and output locations are resolved in this order:
#
#   1. the DASPS_RAW, DASPS_PRE and DASPS_RESULTS environment variables
#   2. a Kaggle input folder, discovered automatically
#   3. ./data/raw, ./data/preprocessed and ./results, relative to this file
#
# DASPS itself is distributed as .mat files; point DASPS_RAW at the folder
# holding the raw recordings and DASPS_PRE at the preprocessed ones.
#
#   export DASPS_RAW=/path/to/DASPS/raw
#   export DASPS_PRE=/path/to/DASPS/preprocessed
#   export DASPS_RESULTS=/path/to/results
def _find_kaggle():
    """Locate the dataset inside /kaggle/input without relying on its slug.

    Any folder below /kaggle/input that contains .mat files is taken as a
    data directory; files whose name contains 'preprocessed' mark it as the
    preprocessed set, otherwise it is treated as raw.
    """
    base = '/kaggle/input'
    if not os.path.isdir(base):
        return None
    raw = pre = None
    for root, _dirs, files in os.walk(base):
        mats = [f for f in files if f.lower().endswith('.mat')]
        if not mats:
            continue
        if any('preprocessed' in f.lower() for f in mats):
            pre = pre or root
        else:
            raw = raw or root
    if raw or pre:
        return dict(raw=raw or '', pre=pre or '', out='/kaggle/working/results')
    return None


_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = []

# 1) explicit configuration
if os.environ.get('DASPS_RAW') or os.environ.get('DASPS_PRE'):
    _CANDIDATES.append(dict(
        raw=os.environ.get('DASPS_RAW', ''),
        pre=os.environ.get('DASPS_PRE', ''),
        out=os.environ.get('DASPS_RESULTS', os.path.join(_HERE, 'results'))))

# 2) Kaggle
_k = _find_kaggle()
if _k:
    _CANDIDATES.append(_k)

# 3) alongside this file
_CANDIDATES.append(dict(
    raw=os.path.join(_HERE, 'data', 'raw'),
    pre=os.path.join(_HERE, 'data', 'preprocessed'),
    out=os.environ.get('DASPS_RESULTS', os.path.join(_HERE, 'results'))))

RAW_DIR = PRE_DIR = RESULTS_ROOT = None
for _c in _CANDIDATES:
    if os.path.isdir(_c['raw']) or os.path.isdir(_c['pre']):
        RAW_DIR, PRE_DIR, RESULTS_ROOT = _c['raw'], _c['pre'], _c['out']
        break
if RAW_DIR is None:
    # Nothing found on disk yet. Fall back to the last candidate so that the
    # paths are defined and the error message names where we looked.
    _c = _CANDIDATES[-1]
    RAW_DIR, PRE_DIR, RESULTS_ROOT = _c['raw'], _c['pre'], _c['out']
    print("WARNING: DASPS data folder not found. Set DASPS_RAW and DASPS_PRE, or place the data in ./data/raw and ./data/preprocessed.")

FS = 128
SEG_LEN = 128
BANDS = [(5, 8), (13, 30)]              # theta, beta  (paper text must match this)
SEL_CH = [0, 13, 3, 10, 5, 8]           # AF3, AF4, FC5, FC6, P7, P8
CH_NAMES = ['AF3', 'AF4', 'FC5', 'FC6', 'P7', 'P8']
N_CH = 6
ASYM_PAIRS = [(0, 1), (2, 3), (4, 5)]   # (AF3,AF4), (FC5,FC6), (P7,P8)
WAVELET, WAV_LEVEL = 'db4', 4

DEVICE = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          if HAVE_TORCH else 'cpu')

HAM_A = {1:37, 2:30, 3:36, 4:33, 5:36, 6:38, 7:27, 8:19, 9:17, 10:12, 11:24,
         12:35, 13:25, 14:7, 15:18, 16:26, 17:29, 18:17, 19:44, 20:38,
         21:23, 22:28, 23:14}

PAPER = {'binary':     {'acc': 94.90, 'f1': 0.93, 'kappa': 0.88},
         'four_class': {'acc': 92.74, 'f1': 0.90, 'kappa': 0.86}}


# ---------------- reproducibility ----------------
def set_seed(seed=42):
    """Seed every RNG the pipeline touches, so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    if HAVE_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------- labels ----------------
def ham_to_label(ham, task):
    if task == 'binary':
        return 0 if ham <= 20 else 1
    if ham <= 12:  return 0     # Normal
    if ham <= 20:  return 1     # Light
    if ham <= 25:  return 2     # Moderate
    return 3                    # Severe


CLASS_NAMES = {'binary':     ['Light', 'Severe'],
               'four_class': ['Normal', 'Light', 'Moderate', 'Severe']}


# ---------------- data ----------------
def load_data(task='binary', verbose=True):
    """Returns X (n,128,6), y (n,), subject_ids (n,)."""
    X_all, y_all, sid_all = [], [], []
    for sid in range(1, 24):
        y_subj = ham_to_label(HAM_A[sid], task)
        pre = os.path.join(PRE_DIR, f'S{sid:02d}preprocessed.mat')
        raw = os.path.join(RAW_DIR, f'S{sid:02d}.mat')
        eeg = None
        if os.path.exists(pre):
            try:
                with h5py.File(pre, 'r') as f:
                    for k in f.keys():
                        if k.startswith('#'):
                            continue
                        d = np.array(f[k])
                        if d.ndim == 3:
                            eeg = d
                            break
            except Exception:
                pass
        if eeg is None:
            with h5py.File(raw, 'r') as f:
                eeg = np.array(f['data'])

        a, b, c = eeg.shape
        if   a == 12 and c == 14: pass
        elif a == 12 and b == 14: eeg = eeg.transpose(0, 2, 1)
        elif c == 12:             eeg = eeg.transpose(2, 1, 0)

        sel = eeg[:, :, SEL_CH]
        for ep in range(sel.shape[0]):
            epoch = sel[ep]
            if epoch.shape[0] != 1920:
                if epoch.shape[1] == 1920:
                    epoch = epoch.T
                else:
                    continue
            for s in range(epoch.shape[0] // SEG_LEN):
                X_all.append(epoch[s * SEG_LEN:(s + 1) * SEG_LEN, :])
                y_all.append(y_subj)
                sid_all.append(sid)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)
    sids = np.array(sid_all)
    if verbose:
        u, c = np.unique(y, return_counts=True)
        print(f"  Segments: {len(X)} | Classes: {dict(zip(u.tolist(), c.tolist()))}")
    return X, y, sids


# ---------------- features (order preserved from original code) ----------------
def _band_mean_power(seg, lo, hi):
    freqs = rfftfreq(SEG_LEN, 1.0 / FS)
    mask = (freqs >= lo) & (freqs < hi)
    out = []
    for c in range(N_CH):
        ps = np.abs(rfft(seg[:, c])) ** 2
        out.append(float(np.mean(ps[mask])) if mask.any() else 1e-10)
    return np.array(out)


def extract_segment(seg):
    feats = []
    # 1) Differential entropy: 0.5*log(2*pi*e*sigma^2), sigma^2 ~ mean band power  -> 12
    for lo, hi in BANDS:
        for v in _band_mean_power(seg, lo, hi):
            feats.append(0.5 * np.log(2 * np.pi * np.e * max(v, 1e-10)))
    # 2) Hjorth -> 18
    for c in range(N_CH):
        s = seg[:, c].astype(float)
        d1, d2 = np.diff(s), np.diff(np.diff(s))
        act = float(np.var(s))
        mob = float(np.sqrt(np.var(d1) / (act + 1e-10)))
        comp = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-10)) / (mob + 1e-10))
        feats.extend([act, mob, comp])
    # 3) Wavelet SUB-BAND ENERGY (not raw coefficients) -> 30
    for c in range(N_CH):
        for co in pywt.wavedec(seg[:, c], WAVELET, level=WAV_LEVEL):
            feats.append(float(np.sum(co ** 2)))
    # 4) Mean power -> 12
    for lo, hi in BANDS:
        feats.extend(_band_mean_power(seg, lo, hi).tolist())
    # 5) Rational asymmetry + asymmetry index -> 6 + 6
    for lo, hi in BANDS:
        mp = _band_mean_power(seg, lo, hi)
        for l, r in ASYM_PAIRS:
            lp, rp = mp[l] + 1e-10, mp[r] + 1e-10
            feats.append(rp / lp)                       # rational asymmetry
            feats.append(np.log(rp) - np.log(lp))       # asymmetry index
    f = np.array(feats)
    assert f.shape[0] == 84, f"Expected 84 features, got {f.shape[0]}"
    return f


def extract_all(X, verbose=True):
    if verbose:
        print("  Extracting 84 features...", flush=True)
    out = []
    for i, seg in enumerate(X):
        out.append(extract_segment(seg))
        if verbose and (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(X)}", flush=True)
    F = np.array(out, dtype=np.float32)
    if verbose:
        print(f"  Feature matrix: {F.shape}")
    return F


# ---------------- models ----------------
class FeedforwardTransformer(nn.Module):
    """Paper's proposed model. NOTE: the projection stage is fully connected,
    not convolutional -- 'feedforward' is the accurate name."""
    def __init__(self, n_features=84, n_classes=2, d_model=128,
                 n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_features, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                         dim_feedforward=d_model * 4, dropout=dropout,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, n_classes))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.ff(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.classifier(x)


class FeedforwardOnly(nn.Module):
    def __init__(self, n_features=84, n_classes=2, d_model=128, dropout=0.3, **kw):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class TransformerOnly(nn.Module):
    def __init__(self, n_features=84, n_classes=2, d_model=128,
                 n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_features, d_model),
                                  nn.LayerNorm(d_model), nn.GELU())
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                         dim_feedforward=d_model * 4, dropout=dropout,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, n_classes))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.classifier(x)


MODELS = {'FeedforwardTransformer': FeedforwardTransformer,
          'FeedforwardOnly':        FeedforwardOnly,
          'TransformerOnly':        TransformerOnly}
