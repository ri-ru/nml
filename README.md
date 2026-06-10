# NBA Multi-Agent Trajectory Prediction — EE-452 (NML 2026)

Given **8 context frames** of an NBA possession, predict the next **12 positions** of all
**11 entities** (10 players + the ball). Scored by mean-squared error in court coordinates (feet),
lower is better, averaged uniformly over entities, timesteps, and axes (the ADE metric).

**Method.** A single **graph-attention network** (spatio-temporal GAT + GRU + mixture-of-Gaussians
head), trained six ways under one architecture and combined by an **error-analysis-driven, 2-group
ensemble** that automatically specialises models to the ball vs. the players.

**Result.** Kaggle ADE **2.954** — a 6-model, single-architecture ensemble (3 ball specialists +
3 player generalists), members retrained on 100 % of the labelled data. Honest 80/20
cross-validation gives local ADE **2.758**.

---

## Repository structure

```
core/        Shared infrastructure used by every model.
demarche/    The construction path + the ONE architecture: demarche/nba_v46.py.
ensemble/    The 80/20 validated pipeline (honest local CV -> local ADE 2.758).
final/       The final submission (Kaggle ADE 2.954).
```

The architecture is defined **once**, in `demarche/nba_v46.py`. The 6 ensemble members are not copies
of it — they are 3-line configs (seed + ball-weight) that import it, and `final/` loads it from
`demarche/` too. There is exactly one copy of the network in the whole repo.

### `core/` — shared infrastructure (3 files)

- **`make_split.py`** — makes the 80/20 train/val split once (fixed seed 42); writes
  `train_files.txt` + `val_files.txt` so every model uses the *same* split.
- **`run_one.py`** — the training entry point: `python core/run_one.py <model.py> <epochs> <batch>`.
  Dynamically imports the model's `Trainer`, trains, saves an epoch-tagged checkpoint, evaluates.
- **`evaluate.py`** — local scorer: de-normalises to court feet and computes the MSE (same metric as
  Kaggle), using a fixed 3-windows-per-sequence sampler.

### `demarche/` — how we got here (one file per milestone)

`baseline`, `v6`, `v38` are self-contained files (genuinely *different* earlier networks). `v46` is
**the** architecture — every later component (ensemble members, final) imports it. `v60` is `v46`
with a ball-weighted loss. Each is runnable via `core/run_one.py` with its trained weight. Local ADE
is on the 80/20 val split.

| Step | File | What's new | Local ADE |
|---|---|---|---|
| 0 Baseline | `nba_baseline.py` | provided non-graph GRU, each entity independent (MSE) | 54.0 |
| 1 Graph | `nba_v6.py` | **graph attention** + ball-relative features (the graph is born) | 4.36 |
| 2 Gaussian | `nba_v38.py` | Gaussian NLL (predict a distribution) + kNN social features | 3.44 |
| 3 Mixture | `nba_v46.py` | **mixture-of-2-Gaussians head** + timestep-weighted NLL, 300 ep — **the canonical model** | 2.86 |
| 4 Ball-weighting | `nba_v60_base.py`, `nba_v60_w8.py` | up-weight the ball ×8 in the loss (train ball specialists) | — |

The five components a grader probes, all in `demarche/nba_v46.py`:
**I/O** `NBADataset` (X=8 ctx, y=12 horizon) · **architecture** `NBAv46Model` (2× GRU+attention →
MoG head) · **graph** `TypeAwareDistanceBiasedAttention` (distance + type-pair bias over 11 nodes) +
kNN `_social_features` · **temporal** per-node `GRUCell` + autoregressive rollout · **objective**
`MoGNLLLoss`.

### `ensemble/` — the 80/20 validated pipeline

Members trained on 80 % of the data; the 2-group **ball | players** weights fit on the held-out
20 %. The key finding (`analyze_error_sources.py`): the **ball alone is ~39 % of the total error**
(~6× any player). So we fit *two* weight vectors — one for the ball, one shared across players — and
the optimiser routes ball-weighted models to the ball and generalists to the players.

- `save_val_tta_preds.py` / `save_test_tta_preds.py` — cache the 6 members' 4-flip TTA predictions.
- `ball_player_ensemble.py` — fit the 2-group weights and write the submission CSV.
- `show_weights.py` — print the fitted ball/player weight vectors.
- `models/` — the 6 members as 3-line configs (seed + ball-weight) that import the architecture from
  `demarche/`; `weights/` — their 80 %-split checkpoints. (The `v46` member is the `demarche/nba_v46.py`
  milestone itself.)

### `final/` — the final submission

The 6 members (3 ball specialists `v60_*` + 3 player generalists `v46*`), retrained on 100 % of the
data, combined by the 2-group weights. All the same architecture, loaded from `demarche/nba_v46.py`.

| Model | Role | Ball w | Player w |
|---|---|---:|---:|
| `v60_w8`    | ball specialist (×8)        | 0.606 | 0.000 |
| `v60_w5_s1` | ball specialist (×5, seed 1)| 0.246 | 0.000 |
| `v60_w8_s2` | ball specialist (×8, seed 2)| 0.099 | 0.000 |
| `v46_s3`    | player generalist (seed 3)  | 0.048 | 0.285 |
| `v46`       | player generalist (baseline)| 0.000 | 0.328 |
| `v46_s7`    | player generalist (seed 7)  | 0.000 | 0.387 |

`build_submission.py` loads the 6 checkpoints → 4-flip TTA on the test set → applies the weights from
`ensemble_weights.json` → writes `solution.csv`.

---

## Data

The dataset is **not** committed. Point the code at it via an environment variable — the folder that
contains `train/train/*.pt` and `test/test/*.pt` (kagglehub competition layout):

```bash
export NML_DATA=/path/to/nml-2026          # macOS / Linux
$env:NML_DATA = "C:\path\to\nml-2026"      # Windows PowerShell
```

## Setup

```bash
pip install -r requirements.txt            # torch, numpy, pandas, scipy, tqdm
```

A CUDA GPU is needed for training and prediction; the ensemble weight-fitting is CPU-only.

## Reproduce

```bash
# 1. The final 2.954 submission (only final/ + the dataset are needed):
python final/build_submission.py                       # -> final/solution.csv

# 2. The validated 2.758 local result (from scratch):
python core/make_split.py                              # regenerate train/val manifests
python ensemble/save_val_tta_preds.py                 # GPU: cache val predictions
python ensemble/save_test_tta_preds.py                # GPU: cache test predictions
python ensemble/ball_player_ensemble.py               # CPU: fit 2-group weights -> CSV

# 3. Train any milestone model:
python core/run_one.py demarche/nba_v46.py 300 32     # <model.py> <epochs> <batch>
```
