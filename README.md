# NBA Multi-Agent Trajectory Prediction — EE-452 (NML 2026)

Given **8 context frames** of an NBA possession, predict the next **12 positions** of all
**11 entities** (10 players + the ball). Scored by mean-squared error in court coordinates (feet),
lower is better, averaged uniformly over entities, timesteps, and axes (the ADE metric).

**Method.** A single **graph-attention network** (spatio-temporal GAT + GRU + mixture-of-Gaussians
head), trained under four principled regimes and combined by an **error-analysis-driven, per-entity
ensemble** that automatically specialises models to the ball vs. the players.

**Result.** Kaggle **2.98086** (validated 80/20 ensemble) → **2.95986** (members retrained on
100 % of the labelled data) → **2.95815** (final, distilled to the **7 weight-carrying models**). The
full method write-up is in [`docs/REPORT.md`](docs/REPORT.md).

## Final ensemble (Kaggle 2.95815) — the 7 models and their weights

The full pipeline trained 27 variants, but the fitted ensemble showed only 7 carry any weight; refit
on just those 7 it **matches and slightly beats** the 27-model submission with ~4× fewer models. The
**2-group** ensemble fits one weight vector for the ball (entity 0) and one shared vector for all
players (entities 1–10); the optimiser cleanly splits the pool into ball experts and player experts.

| Model | Architecture | Ball weight | Player weight | Role |
|---|---|---:|---:|---|
| `v60_w8`    | nba_v46 / NBAv46Model | **0.606** | 0.000 | ball expert (ball-weighted loss ×8) |
| `v60_w5_s1` | nba_v46 / NBAv46Model | **0.246** | 0.000 | ball expert (ball-weighted loss ×5, seed 1) |
| `v60_w8_s2` | nba_v46 / NBAv46Model | **0.099** | 0.000 | ball expert (ball-weighted loss ×8, seed 2) |
| `v46_s3`    | nba_v46 / NBAv46Model | 0.048 | 0.204 | both (the only cross-over) |
| `v46`       | nba_v46 / NBAv46Model | 0.000 | 0.190 | player generalist (baseline) |
| `v58`       | nba_v58 / NBAv58Model | 0.000 | 0.263 | player generalist (pseudo-labeled) |
| `v46_s7`    | nba_v46 / NBAv46Model | 0.000 | **0.344** | player generalist (seed 7) |

6 of the 7 share the **same** network (`NBAv46Model`), differing only in training (ball-weighting,
seeds); `v58` uses `NBAv58Model` (same design, pseudo-labelled). Each member predicts under 4-flip TTA
on its 100 %-data (`_full`) checkpoint, then the per-group weights combine them. The self-contained
build (weights + the 2 architecture files + `ensemble_weights.json` + one script) lives in
[`final_7model/`](final_7model/README.md).

**6-model variant (drop `v58`, single architecture).** Removing `v58` leaves six pure-`NBAv46Model`
members — "one architecture, trained six ways". The ball side is **unchanged** (`v58` carried no ball
weight); on the player side its 0.263 is reabsorbed by the v46 generalists (`v46_s7` 0.387, `v46`
0.328, `v46_s3` 0.285). Local 2-group MSE is **2.7583 vs 2.7568** for the 7-model pool — a hair worse,
because `v58`'s pseudo-labelled, different-architecture diversity is lost. Refit purely over cached
predictions (no retraining): `python final_run/build_trimmed6_candidate.py` →
`final_run/solutions/solution_trimmed6_2grp.csv`. We keep the **7-model** ensemble as the headline
(better local number and the stronger story); the 6-model CSV is a candidate to compare on Kaggle.

---

## Repository layout

```
demarche/     The méthode / journey — one model file per milestone step (baseline → v6 → v38
              → v46 → v58 → v60 ball-weighting), each with its trained weight. Read this to
              understand WHY the architecture looks the way it does. See demarche/README.md.

ensemble/     The 80/20 VALIDATED pipeline (honest local CV → Kaggle 2.98086).
              27 member weights + the graph model definitions + the scripts that cache
              predictions and fit the 2-group (ball | players) ensemble. See ensemble/README.md.

final_run/    The 100 %-data submission (Kaggle 2.95986). The same members RETRAINED on 100 % of
              the labelled data (train + val), reusing the 2-group weights fit in ensemble/.
              See final_run/README.md.

final_7model/ The FINAL submission (Kaggle 2.95815), self-contained. The 27-model ensemble
              distilled to the 7 weight-carrying models (3 ball experts + 4 player generalists):
              7 checkpoints + 2 architecture files + ensemble_weights.json + build_submission.py,
              with no dependency on ensemble/ or final_run/. See final_7model/README.md.

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
- **The 100 %-data 2.95986 submission** → see [`final_run/README.md`](final_run/README.md).
- **The final 2.95815 (7-model) submission** → see [`final_7model/README.md`](final_7model/README.md).
