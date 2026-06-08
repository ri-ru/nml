# NBA Multi-Agent Trajectory Prediction — EE-452 (NML 2026)

Given **8 context frames** of an NBA possession, predict the next **12 positions** of all
**11 entities** (10 players + the ball). Scored by mean-squared error in court coordinates (feet),
lower is better, averaged uniformly over entities, timesteps, and axes (the ADE metric).

**Method.** A single **graph-attention network** (spatio-temporal GAT + GRU + mixture-of-Gaussians
head), trained under four principled regimes and combined by an **error-analysis-driven, per-entity
ensemble** that automatically specialises models to the ball vs. the players.

**Result.** Kaggle **2.98086** (validated 80/20 ensemble) → **2.95986** (final, members retrained on
100 % of the labelled data). The full method write-up is in [`docs/REPORT.md`](docs/REPORT.md).

---

## Repository layout

```
demarche/     The méthode / journey — one model file per milestone step (baseline → v6 → v38
              → v46 → v58 → v60 ball-weighting), each with its trained weight. Read this to
              understand WHY the architecture looks the way it does. See demarche/README.md.

ensemble/     The 80/20 VALIDATED pipeline (honest local CV → Kaggle 2.98086).
              27 member weights + the graph model definitions + the scripts that cache
              predictions and fit the 2-group (ball | players) ensemble. See ensemble/README.md.

final_run/    The FINAL submission (Kaggle 2.95986). The same members RETRAINED on 100 % of the
              labelled data (train + val), reusing the 2-group weights fit in ensemble/.
              See final_run/README.md.

core/         Shared infra: make_split.py (80/20 split), run_one.py (train one model),
              evaluate.py (local val MSE).

docs/         REPORT.md (method), ERROR_ANALYSIS.md (the "ball dominates" finding),
              CHANGELOG.md + results.md + notes.md (full experimental log).

ORIGINAL.ipynb            The provided starter notebook (visualisation + naive baseline).
NML_Project_Proposal.pdf  The assignment.
train_files.txt / val_files.txt / train_files_full.txt   Split manifests (see "Data" below).
```

> The full research history (≈90 model variants, all checkpoints, every submission CSV, the
> autonomous training orchestrators) lives on the **`main-archive`** branch. This branch keeps
> only what's needed to understand and reproduce the result.

---

## Data

The dataset is **not** committed. Point the code at it once via an environment variable — the folder
that contains `train/train/*.pt` and `test/test/*.pt` (kagglehub competition layout):

```bash
export NML_DATA=/path/to/nml-2026          # macOS / Linux
$env:NML_DATA = "C:\path\to\nml-2026"      # Windows PowerShell
```

The `*_files.txt` manifests hold absolute paths from the machine they were generated on.
Regenerate them for your machine with:

```bash
python core/make_split.py        # writes train_files.txt + val_files.txt at the repo root
```

## Setup

```bash
pip install -r requirements.txt   # torch, numpy, pandas, scipy, tqdm
```

A CUDA GPU is needed for training and for generating predictions. The ensemble weight-fitting step
is CPU-only.

## Reproduce

- **The validated 2.98086 result** → see [`ensemble/README.md`](ensemble/README.md).
- **The final 2.95986 submission** → see [`final_run/README.md`](final_run/README.md).
