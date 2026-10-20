# Participant identity, not anxiety

Code and per-segment predictions for the study

> **Participant identity, not anxiety: segment-level partitioning inflates EEG
> affect classification on the DASPS dataset**
> Abdullah Javed, Bin Guo, Milembolo Miantezila Junior
> Changchun University of Science and Technology

---

## What this is

In the DASPS EEG dataset a single Hamilton Anxiety Rating Scale (HAM-A) score
is recorded **per participant** and inherited by all 180 one-second segments
from that participant. The segment-level leave-one-out protocol used
throughout the DASPS literature therefore leaves 179 same-participant,
same-label segments in the training set of every fold.

This repository contains everything needed to reproduce the result that
follows from that fact:

| | Binary | Four-class |
|---|---|---|
| Participant identity recovered from one second of EEG | 71.94% | (chance 4.35%) |
| HAM-A label recovered from identity alone, no anxiety supervision | 86.17% | 79.84% |
| Majority-class rate | 69.57% | 56.52% |
| Best segment-level (LOO) accuracy | 84.54% | 77.46% |
| Best participant-independent (LOSO) accuracy | **63.94%** | **46.86%** |

Every participant-independent result falls at or below the majority-class
rate, and Cohen's kappa falls to between -0.0658 and +0.0348.

Five classifier configurations are evaluated under two partitioning schemes on
two tasks: 20 cells, 41,400 independently trained models.

---

## Requirements

```
python >= 3.9
numpy, scipy, scikit-learn, h5py, PyWavelets, matplotlib
torch            # only for the neural configurations
```

```bash
pip install -r requirements.txt
```

## Getting the data

DASPS is publicly available from IEEE DataPort:
<https://ieee-dataport.org/open-access/dasps-database>

Point the code at it with environment variables:

```bash
export DASPS_RAW=/path/to/DASPS/raw            # the raw .mat recordings
export DASPS_PRE=/path/to/DASPS/preprocessed   # the preprocessed .mat files
export DASPS_RESULTS=/path/to/results          # where output is written
```

Without these, the code looks for `./data/raw` and `./data/preprocessed`, and
writes to `./results`. On Kaggle the dataset folder is discovered
automatically.

## Reproducing the paper

Run `preflight.py` first; it loads the data, extracts features, runs one
forward pass and checks that results can be written.

```bash
python -u preflight.py
```

**Step 1 -- what the label is bound to**

```bash
python -u label_balance.py
```

**Step 2 -- the identity control**

Trains a classifier to predict *which participant* a segment came from, then
composes that prediction with the dataset's participant-to-label lookup. No
anxiety label is used in training.

```bash
python -u identity_to_label.py
```

**Step 3 -- the evaluation grid**

Each command is one cell and is safe to re-run: it checkpoints every
`--ckpt-every` folds and skips a cell that is already complete.

```bash
# random forests, both protocols, both tasks
python -u run_rf_protocols.py --task binary     --protocol loo
python -u run_rf_protocols.py --task binary     --protocol loso
python -u run_rf_protocols.py --task four_class --protocol loo
python -u run_rf_protocols.py --task four_class --protocol loso

# neural configurations
for M in FeedforwardOnly TransformerOnly FeedforwardTransformer; do
  python -u train_model.py --task binary     --model $M --epochs 100 --protocol loo
  python -u train_model.py --task binary     --model $M --epochs 100 --protocol loso
  python -u train_model.py --task four_class --model $M --epochs 200 --protocol loo
  python -u train_model.py --task four_class --model $M --epochs 200 --protocol loso
done
```

A leave-one-out cell is 4,140 folds. On a laptop CPU the full grid takes
several days; on a GPU it is considerably faster.

**Step 4 -- audit, statistics and figures**

```bash
python -u check_leakage.py     # per-class recall; alarms on the leakage signature
python -u analysis_stats.py    # cluster bootstrap + McNemar -> stats_summary.json
python -u figures_v4.py        # diagnostic figures
```

## What each script does

| Script | Purpose |
|---|---|
| `dasps_common.py` | data loading, the 84-dimensional feature vector, the three model definitions, seeding |
| `preflight.py` | eleven independent checks before a long run |
| `label_balance.py` | per-participant HAM-A scores and the classes they induce |
| `identity_to_label.py` | the identity control |
| `train_model.py` | neural configurations, both protocols, with checkpoint/resume |
| `run_rf_protocols.py` | random-forest configurations, both protocols |
| `check_leakage.py` | per-class recall, with alarms for the leakage signature |
| `analysis_stats.py` | participant-level cluster bootstrap and McNemar |
| `collect_results.py` | tabulates every completed cell |
| `figures_v4.py` | diagnostic figures from the result files |
| `make_fig*.py` | the figures as they appear in the paper |
| `verify_numbers.py` | checks every number printed in the manuscript against the result files |

`verify_numbers.py` is included deliberately: it re-derives each figure in the
paper's tables from the JSON files and fails if any of them disagree.

## The implementation defect

An earlier implementation of this work contained two coupled faults, both
removed here and both documented in `train_model.py`:

1. the held-out segment was predicted after every epoch and the prediction
   kept whenever it matched the held-out label, so the reported epoch was
   selected using the test label;
2. the stored prediction was initialised to class 0, so any segment never
   classified correctly was recorded as class 0.

Together they produced exactly 100% recall on the first class with all errors
in a single column. `check_leakage.py` tests for that signature; it is absent
from all 20 cells reported here.

## Results in this repository

`results/` holds, for every cell:

- `results_*.json` -- accuracy, weighted F1, Cohen's kappa, precision, recall
  and the confusion matrix
- `y_true_*.npy` and `y_pred_*.npy` -- the **per-segment predictions**

The per-segment predictions are released because the paper's own analysis is
limited by their absence elsewhere: no paired test against previously
published figures is possible, since those predictions were never made
available. Releasing ours lets anyone run that test against this work.

## Citing

See `CITATION.cff`. The dataset itself should be cited as
Baghdadi et al., *DASPS: A Database for Anxious States based on a
Psychological Stimulation*, arXiv:1901.02942.

## License

MIT, see `LICENSE`. The DASPS dataset is distributed under its own terms by
IEEE DataPort and is not included here.
