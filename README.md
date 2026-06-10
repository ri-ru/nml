# NBA Multi-Agent Trajectory Prediction — EE-452 (NML 2026)

Given **8 context frames** of an NBA possession, predict the next **12 positions** of all
**11 entities** (10 players + the ball). Scored by mean-squared error in court coordinates (feet),
lower is better, averaged uniformly over entities, timesteps, and axes (the ADE metric).

**Method.** A single **graph-attention network** (spatio-temporal GAT + GRU + mixture-of-Gaussians
head), trained six ways under one architecture and combined by an **error-analysis-driven, 2-group
ensemble** that automatically specialises models to the ball vs. the players.

**Result.** Kaggle ADE **2.954** — a 6-model, single-architecture ensemble (3 ball specialists +
3 player generalists) with the members retrained on 100 % of the labelled data. Honest 80/20
cross-validation gives local ADE **2.758**.

---

## Repository layout

```
demarche/   The journey — one model file per milestone step that moved the score:
            baseline (independent GRU) -> v6 (graph attention) -> v38 (Gaussian NLL)
            -> v46 (mixture-of-Gaussians head, THE architecture) -> v60 (ball-weighting).
            Read this to understand WHY the architecture looks the way it does.

ensemble/   The 80/20 VALIDATED pipeline (honest local CV, local ADE 2.758). The 6 member
            weights + the graph model definitions + the scripts that cache predictions and
            fit the 2-group (ball | players) ensemble. See ensemble/README.md.

final/      The FINAL submission (Kaggle ADE 2.954), self-contained. The 6 members retrained
            on 100 % of the labelled data + the architecture + ensemble_weights.json +
            build_submission.py + solution.csv. See final/README.md.

core/       Shared infra: make_split.py (80/20 split), run_one.py (train one model),
            evaluate.py (local val MSE).

docs/       figures/ (the report figures) + SCREENCAST.md (the code-walkthrough script).

ORIGINAL.ipynb            The provided starter notebook (visualisation + naive baseline).
NML_Project_Proposal.pdf  The assignment.
train_files.txt / val_files.txt        80/20 split manifests (regenerate with core/make_split.py).
train_files_full.txt                   100%-data manifest (for the final retrain).
```

## Data

The dataset is **not** committed. Point the code at it once via an environment variable — the folder
that contains `train/train/*.pt` and `test/test/*.pt` (kagglehub competition layout):

```bash
export NML_DATA=/path/to/nml-2026          # macOS / Linux
$env:NML_DATA = "C:\path\to\nml-2026"      # Windows PowerShell
```

The `*_files.txt` manifests hold absolute paths from the machine that generated them. Regenerate
them for your machine with:

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

- **The final 2.954 submission** → `python final/build_submission.py` (see [`final/README.md`](final/README.md)).
- **The validated 2.758 local result** → see [`ensemble/README.md`](ensemble/README.md).
- **Train any milestone model** → `python core/run_one.py demarche/nba_v46.py 300 32` (see [`demarche/README.md`](demarche/README.md)).
