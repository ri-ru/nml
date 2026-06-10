# NBA Multi-Agent Trajectory Prediction (EE-452, NML 2026)

Given **8 context frames** of an NBA possession, predict the next **12 positions** of all
**11 entities** (10 players + the ball). Scored by mean-squared error in court coordinates (feet),
lower is better, averaged uniformly over entities, timesteps, and axes (the ADE metric).

**Method.** A single **graph-attention network** (spatio-temporal GAT + GRU + mixture-of-Gaussians
head), trained six ways under one architecture and then combined by an **error-analysis-driven, 2-group
ensemble** that automatically specialises models to the ball vs. the players.

**Result.** Kaggle ADE **2.954** (a 6-model, single-architecture ensemble: 3 ball specialists + 3
player generalists, members retrained on 100 % of the labelled data). The honest 80/20 cross-validation
gives local ADE **2.758**.

---

## Repository structure

```
run.py       One launcher for everything (train / ensemble / both submissions). Start here.
utils/       Shared infrastructure (split / train / score).
demarche/    The construction path + the ONE architecture (demarche/nba_v46.py);
             the 80/20 validated ensemble pipeline lives in demarche/ensemble/.
final/       The two final Kaggle submissions (6-model ensemble = ADE 2.954; single model).
outputs/     weights/ (every trained checkpoint) + solutions/ (the submitted CSVs).
```

The architecture is defined **once**, in `demarche/nba_v46.py`. The 6 ensemble members are not copies
of it: they are entries in a small registry (`demarche/ensemble/_config.py` → `MEMBERS`, just seed +
ball-weight) that all load that one file; `final/` loads it too. There is exactly one copy of the
network in the whole repo.

### `utils/`: shared infrastructure (3 files)

- **`make_split.py`**: makes the 80/20 train/val split once (fixed seed 42); writes
  `train_files.txt` + `val_files.txt` so every model uses the *same* split.
- **`run_one.py`**: the training entry point: `python utils/run_one.py <model.py> <epochs> <batch>`.
  Dynamically imports the model's `Trainer`, trains, saves an epoch-tagged checkpoint, evaluates.
- **`evaluate.py`**: local scorer: de-normalises to court feet and computes the MSE (same metric as
  Kaggle), using a fixed 3-windows-per-sequence sampler.

### `demarche/`: how we got here (one file per milestone)

`baseline`, `v6`, `v38` are self-contained files (genuinely *different* earlier networks). `v46` is
the architecture: every later component (ensemble members, final) imports it. `v60` is `v46`
with a ball-weighted loss. Each is runnable via `utils/run_one.py` with its trained weight. Local ADE
is on the 80/20 val split.

| Step | File | What's new | Local ADE |
|---|---|---|---|
| 0 Baseline | `nba_baseline.py` | provided non-graph GRU, each entity independent (MSE) | 54.01 |
| 1 Graph | `nba_v6.py` | **graph attention** + ball-relative features (the graph is born) | 4.36 |
| 2 Gaussian | `nba_v38.py` | Gaussian NLL (predict a distribution) + kNN social features | 3.44 |
| 3 Mixture | `nba_v46.py` | **mixture-of-2-Gaussians head** + timestep-weighted NLL, 300 ep (**the canonical model**) | 2.86 |
| 4 Ball-weighting | `nba_v60_base.py` | the ball-weighted loss + trainer factory; ball specialists are trained via `demarche/ensemble/train_member.py` | n/a |

The five components a grader probes, all in `demarche/nba_v46.py`:
**I/O** `NBADataset` (X=8 ctx, y=12 horizon) · **architecture** `NBAv46Model` (2× GRU+attention →
MoG head) · **graph** `TypeAwareDistanceBiasedAttention` (distance + type-pair bias over 11 nodes) +
kNN `_social_features` · **temporal** per-node `GRUCell` + autoregressive rollout · **objective**
`MoGNLLLoss`.

### `demarche/ensemble/`: the 80/20 validated pipeline

Members trained on 80 % of the data; the 2-group **ball | players** weights fit on the held-out
20 %. The finding from our error analysis: the **ball alone is ~39 % of the total error**
(~6× any player). So we fit *two* weight vectors (one for the ball, one shared across players), and
the optimiser routes ball-weighted models to the ball and generalists to the players.

Three scripts, one job each (plus the shared config):

- `_config.py`: paths + the **`MEMBERS` registry** (the 6 members = label → seed, ball-weight).
- `train_member.py`: **(1) train.** Train one member by name. Each member has two checkpoints,
  differing only in the data seen: the 80% split (to fit weights honestly) and the 100% `--full`
  model (what the submission loads):
  ```bash
  python demarche/ensemble/train_member.py v46_s3 300          # 80% split -> model_v46_s3_300ep.pth
  python demarche/ensemble/train_member.py v46_s3 300 --full   # 100% data -> model_v46_s3_full.pth
  ```
- `save_val_tta_preds.py`: **(2) cache.** Run the 6 (80%) members on the val split with 4-flip TTA,
  dump predictions to `val_preds/` (gitignored) so the fit is CPU-only and repeatable.
- `ball_player_ensemble.py`: **(3) fit.** Least-squares fit of the 2-group weights on those cached
  val predictions, and print them to paste into `final/ensemble_weights.json`.
- checkpoints for all members land in `outputs/weights/` (shared with every other model).

### `final/`: the two final Kaggle submissions

We submitted two entries, one script each (both load 100 %-data `_full` checkpoints and run 4-flip
TTA; all the same architecture, loaded from `demarche/nba_v46.py`):

- **`build_ensemble_6model.py`**: the 6-model ensemble (3 ball specialists `v60_*` + 3 player
  generalists `v46*`), combined by the 2-group weights. Results in **Kaggle ADE 2.954.**
- **`build_single_model.py [label]`**: a single model, no ensembling (default `v46`). Modular: works
  for any member, e.g. `python final/build_single_model.py v60_w8`.

| Model | Role | Ball w | Player w |
|---|---|---:|---:|
| `v60_w8`    | ball specialist (×8)        | 0.606 | 0.000 |
| `v60_w5_s1` | ball specialist (×5, seed 1)| 0.246 | 0.000 |
| `v60_w8_s2` | ball specialist (×8, seed 2)| 0.099 | 0.000 |
| `v46_s3`    | player generalist (seed 3)  | 0.048 | 0.285 |
| `v46`       | player generalist (baseline)| 0.000 | 0.328 |
| `v46_s7`    | player generalist (seed 7)  | 0.000 | 0.387 |

`build_ensemble_6model.py` loads the 6 checkpoints → 4-flip TTA on the test set → applies the weights
from `ensemble_weights.json` → writes `outputs/solutions/solution_ensemble_6model.csv`.

### `outputs/`: weights & submissions

`outputs/weights/` holds every trained checkpoint; the two CSVs submitted to Kaggle live in
`outputs/solutions/`:
- `solution_ensemble_6model.csv`: the 6-model ensemble (ADE **2.954**).
- `solution_v46_full.csv`: a single v46 (100 % data, 4-flip TTA, no ensemble).

---

## Data

The dataset is not committed (because of its size). Tell the code where it is in either way: the
target is the folder that contains `train/train/*.pt` and `test/test/*.pt` (kagglehub layout):

```bash
export NML_DATA=/path/to/nml-2026          # macOS / Linux  (option 1: env var)
$env:NML_DATA = "C:\path\to\nml-2026"      # Windows PowerShell
```

…or just drop/symlink the dataset at **`./data/nml-2026`** in the repo (the gitignored default location
the code falls back to when `NML_DATA` is unset). No env var needed.

Then generate the 80/20 split manifests for your machine (they hold absolute paths, so they are
**not** committed: `make_split.py` recreates the exact same split via its fixed seed):

```bash
python utils/make_split.py                 # writes train_files.txt + val_files.txt at the repo root
```

## Setup

```bash
pip install -r requirements.txt            # torch, numpy, pandas, scipy, tqdm
```

A CUDA GPU is needed for training and prediction; the ensemble weight-fitting is CPU-only.

## Reproduce

Everything runs through one launcher, **`run.py`**, which prints each underlying command before it runs
it (so nothing is hidden). `python run.py` with no args lists every subcommand.

**A. Fastest: rebuild both submissions from the committed `_full` weights** (no training; the 6
`_full` checkpoints are already in `outputs/weights/`):

```bash
python run.py submit               # 6-model ensemble  -> outputs/solutions/solution_ensemble_6model.csv
python run.py submit-single v46    # single model      -> outputs/solutions/solution_v46_full.csv
```

**B. Everything from scratch**: retrain all 6 members twice (an 80% model to fit the weights honestly
+ a 100% `--full` model for the submission), re-fit the weights, rebuild the CSVs. One command:

```bash
python run.py reproduce            # GPU, hours: split -> train-all (80% + full) -> fit-weights -> submit
```

…which is exactly this, if you'd rather run the steps yourself:

```bash
python run.py split                          # fixed-seed 80/20 manifests for THIS machine
python run.py train-all                      # GPU: all 6 members, 80% split   -> model_<m>_300ep.pth
python run.py train-all --full               # GPU: all 6 members, 100% data   -> model_<m>_full.pth
python run.py fit-weights                    # GPU+CPU: cache val/test preds, fit 2-group weights
python run.py submit                         # CPU: ensemble CSV
python run.py submit-single v46              # single-model CSV
```

**C. Train one model** (any ensemble member, or a milestone file):

```bash
python run.py train v46 300                  # one member, 80% split  -> model_v46_300ep.pth
python run.py train v46 300 --full           # same member, 100% data -> model_v46_full.pth
python utils/run_one.py demarche/nba_v38.py 300 32   # a milestone (baseline/v6/v38), 80% split
```

> The 80% (`_300ep`) and 100% (`_full`) checkpoints are the same architecture and command: the
> only difference is `--full`, which trains on every labelled sequence instead of the 80% split. The
> honest 2.758 local ADE comes from the 80% models; the 2.954 Kaggle submission uses the `_full`
> models.
