# final_7model/ — the final submission, distilled to 7 models (Kaggle **2.95815**)

This folder is the **complete, self-contained** final Kaggle submission. It contains everything
needed to regenerate the submitted CSV from scratch — the 7 model checkpoints, their 2 architecture
files, the ensemble weights, and one build script — with no dependency on `ensemble/` or `final_run/`.

## What this means

Our leaderboard pipeline trained **27** variants and combined them with a per-entity ensemble. But
the ensemble's fitted weights showed that only a handful of models actually carry any weight — the
rest get ≈ 0. We tested that directly: refit the ensemble on **just the 7 weight-carrying models**
and submit.

| Submission | Models | Kaggle MSE |
|---|---|---|
| Full per-entity / 2-group ensemble | 27 | 2.95986 |
| **This folder (`solution_final_7model.csv`)** | **7** | **2.95815** |

**The 7-model ensemble matches — in fact very slightly beats — the 27-model one** (lower MSE is
better). The other 20 models were dead weight: they cost training time and inference time and added
nothing to the score. Same accuracy, ~4× fewer models, a much cleaner story.

## The 7 models used (exactly)

The 2-group ensemble fits **one weight vector for the ball** (entity 0) and **one shared vector for
all players** (entities 1–10). The optimizer cleanly partitions the pool into *ball experts* and
*player experts* — almost no model contributes to both:

| Model | Architecture | Ball weight | Player weight | Role |
|---|---|---:|---:|---|
| `v60_w8`    | nba_v46 / NBAv46Model | **0.606** | 0.000 | ball expert (ball-weighted loss ×8) |
| `v60_w5_s1` | nba_v46 / NBAv46Model | **0.246** | 0.000 | ball expert (ball-weighted loss ×5, seed 1) |
| `v60_w8_s2` | nba_v46 / NBAv46Model | **0.099** | 0.000 | ball expert (ball-weighted loss ×8, seed 2) |
| `v46_s3`    | nba_v46 / NBAv46Model | 0.048 | 0.204 | both (the only cross-over) |
| `v46`       | nba_v46 / NBAv46Model | 0.000 | 0.190 | player generalist (baseline) |
| `v58`       | nba_v58 / NBAv58Model | 0.000 | 0.263 | player generalist (pseudo-labeled) |
| `v46_s7`    | nba_v46 / NBAv46Model | 0.000 | **0.344** | player generalist (seed 7) |

- The **3 ball-weighted models own the ball** (0.95 of the ball weight); they get **0** player weight.
- The **base / pseudo / seed generalists own the players**; they get **0** ball weight.
- This is the [§3 error analysis](../docs/REPORT.md) made literal: the ball is ~39 % of the error and
  needs dedicated specialists, the players are easy and just want accurate generalists.

Note the architecture column: **6 of the 7 share the *same* network** (`NBAv46Model`); they differ
only in how they were *trained* (ball-weighting, seeds). `v58` uses `NBAv58Model` (same design,
trained with pseudo-labels). So this is really **one architecture, trained 7 ways** — not 7 designs.

## How it works

```
test sequence (last 8 frames, normalized)
        │
        ├── v60_w8 ─┐
        ├── v60_w5_s1 ─┤   each model: 4-flip TTA (orig, x-, y-, xy-flip), averaged in feet
        ├── v60_w8_s2 ─┤   → per-model prediction [1243, 12, 11, 2]
        ├── v46 ──────┤
        ├── v58 ──────┤
        ├── v46_s3 ───┤
        └── v46_s7 ───┘
        │
        ▼  2-group weighted average (weights in ensemble_weights.json)
   entity 0  (ball)    = Σ ball_weights[m]   · pred[m]
   entity 1..10 (player) = Σ player_weights[m] · pred[m]
        │
        ▼
   solution_final_7model.csv   (Kaggle 2.95815)
```

1. **100 %-data checkpoints.** Each `model_<label>_full.pth` was trained on *all* 4 968 labelled
   sequences (train + val), so no labelled data is wasted in the final run. (Honest 80/20 numbers are
   in [`docs/REPORT.md`](../docs/REPORT.md); the local val score is meaningless here because val is
   inside these models' training set — they are judged only on Kaggle.)
2. **TTA.** Court geometry is symmetric, so each model predicts under 4 reflections and we average
   them back in real coordinates (+1.6 % over no TTA).
3. **2-group ensemble.** Predictions are combined with the per-group weights. The weights were fit by
   constrained least-squares (wᵢ ≥ 0, Σw = 1) on the **honest 80 % validation predictions** (so the
   weighting never saw the data it is scored on), then applied to these 100 %-data test predictions.
   `ensemble_weights.json` stores the exact fitted vectors.

## Files

```
weights/                 7 × model_<label>_full.pth   (100%-data checkpoints, 1.25 MB each)
models/                  nba_v46.py, nba_v58.py        (the 2 architecture definitions)
ensemble_weights.json    the fitted 2-group (ball | player) weights + provenance
build_submission.py      self-contained: weights+arch -> TTA test preds -> apply weights -> CSV
solution_final_7model.csv  the submission (Kaggle 2.95815), produced by build_submission.py
_fit_weights.py          how ensemble_weights.json was produced (refit on ../ensemble/val_preds)
```

## Reproduce

```bash
export NML_DATA=/path/to/nml-2026     # folder containing train/train/*.pt and test/test/*.pt
python final_7model/build_submission.py
# -> final_7model/solution_final_7model.csv  (byte-identical to the submitted file)
```

`build_submission.py` needs only this folder + the dataset (for the test sequences and normalization
statistics). It does **no training** — it is GPU inference of 7 small models (~1.25 MB each) plus a
weighted average. To re-derive `ensemble_weights.json` from scratch (the honest 80 % refit), run
`_fit_weights.py`, which reads the cached 80 % validation predictions in `../ensemble/val_preds`.
