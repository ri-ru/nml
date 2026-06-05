# NBA Trajectory Prediction — NML 2026

Multi-agent trajectory forecasting for 11 entities (10 players + 1 ball).
Given 8 context frames, predict x,y positions over the next 12 timesteps.
Scored by MSE in court coordinates (feet). Lower is better.

---

## Best Results (current)

| Submission | Kaggle MSE | Notes |
|---|---|---|
| **`solution_weighted_perentity_21m.csv`** | **~2.9995 projected** | Per-entity constrained weights over 21 models. Local val 2.7799. **Awaiting Kaggle submission.** |
| `solution_weighted_cons_21m.csv` | **3.00676** | Single-vector constrained weights over 21 models. Local 2.7865. Current submitted best. |
| `solution_tta_ensemble_top5_v46_v58_v46s1_v46s3_v46s3swa.csv` | 3.01420 | Equal-weight 5-model TTA ensemble. Local 2.7905. Previous best before this round. |
| `solution_tta_v46_300ep.csv` | 3.03702 | v46 solo + 4-flip TTA. Best single model. Local 2.8180. |

Local → Kaggle gap is consistently ~7.8% (multiply local by ~1.079).

---

## Architecture

All competitive models share the same backbone: **graph attention over the 11-entity scene** with learned edge features.

- **`TypeAwareDistanceBiasedAttention`** — fully-connected attention over the 11 nodes, with two extra edge features:
  - continuous pairwise distance (`||pos_i − pos_j||`, linearly projected per head)
  - categorical node-type-pair bias (player↔player, player↔ball, ball↔player) via an embedding
- **Encoder:** stack of attention blocks over context frames.
- **Decoder:** autoregressive rollout, 12 future timesteps.
- **Loss:** Gaussian NLL (output `μ_x, μ_y, log σ`); inference uses the means. This was the big jump that moved scores from the 3.7–4 range into the low 3s.
- **Augmentation:** random x/y court flips during training.
- **Inference TTA:** 4-flip test-time augmentation (orig, x-flip, y-flip, xy-flip), averaged in real coordinates.

`v46` and `v58` are minor variants of this architecture; `v58` adds pseudo-label fine-tuning. The `_sN` suffix (e.g. `v46_s3`) marks an alternative random seed; `_swa` averages the state dicts from epochs 280/285/290/295/300 of a 300-epoch run.

This is a graph attention network — it just happens to be hand-rolled rather than using `torch_geometric.nn.GATConv`.

---

## How submissions are built (today's pipeline)

The current best path is **ensemble of many seeded GAT models**, with weights fit on a held-out validation split.

```
run_one.py model.py 300 32          # train each model, saves model_<v>_300ep.pth + 5 SWA snapshots
misc/make_all_swa.py                # average ep 280/285/290/295/300 -> model_<v>_swa.pth
misc/save_val_tta_preds.py          # 4-flip TTA on val_files.txt, dump per-model val tensors
misc/save_test_tta_preds.py         # 4-flip TTA on test set, dump per-model test tensors + solo CSVs
misc/weighted_ensemble.py           # fit weights (free / constrained / greedy) on val, apply to test
misc/weighted_ensemble_per_entity.py # same idea, but one weight vector per entity (player vs ball)
```

Everything before `weighted_ensemble.py` is GPU work. The ensemble fits themselves are CPU + scipy.

### Where the error comes from — the ball dominates

The score averages MSE uniformly over all 11 entities, but the error is wildly concentrated. The
ball (always entity 0) has **~6.3× the MSE of any player** (12.04 vs 1.71–2.08 feet²) because it
moves faster and less predictably (passes, shots). So the ball is **9.1% of the entities but ~39%
of the total error**. If it were predicted as well as an average player, local MSE would drop from
2.82 → 1.90 (−33%). At short horizons (t=1–2) the ball is **58–74% of the entire error**, since
players move smoothly and the model nails their near future.

Full breakdown (ball-vs-player, per-timestep, per-axis) and the implied next experiments live in
[`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md). Reproduce with
`python misc/analyze_error_sources.py [--tta]`.

### Why per-entity weights help

This concentration is also why per-entity ensemble weights help. Fitting a single weight vector
across all entities forces a compromise between ball-friendly and player-friendly weighting.
Per-entity weights let each entity choose its own subset of models. Gain is small (Δ local ≈ −0.007)
but enough to cross the 3.0 line at submission time.

---

## Orchestration

`misc/orchestrate.ps1` polls every 60s for the absence of `python.exe` processes running any `nba_v*_s*.py`. When training is fully idle, it chains:

1. `make_all_swa.py` — builds SWA weights for any seed whose 5 epoch snapshots exist.
2. `save_val_tta_preds.py` — caches per-model val TTA tensors.
3. `save_test_tta_preds.py` — caches per-model test TTA tensors + writes one solo CSV per model.
4. `weighted_ensemble.py` — fits 4 strategies (free / constrained / greedyEq / greedyW), writes one submission CSV per strategy.

All output goes to `models/logs/orchestrate.log`. The orchestrator runs detached, so it survives terminal close.

---

## Repository layout

```
models/
  nba_v<NN>[_sN].py         # one file per model variant (self-contained: Dataset, Model, Trainer)
  weights/                  # *.pth checkpoints (final + per-epoch snapshots + _swa)
  solutions/                # Kaggle submission CSVs
  logs/                     # per-run train/err logs + orchestrate.log
  val_preds/  test_preds/   # cached per-model TTA tensors (gitignored — regenerate via save_*_tta_preds.py)
misc/
  make_tta_*.py             # legacy per-model TTA submission scripts (still useful for one-offs)
  make_all_swa.py           # generic SWA averager across v46/v58 seed families
  save_val_tta_preds.py     # GPU: dump per-model val predictions for ensemble fitting
  save_test_tta_preds.py    # GPU: dump per-model test predictions for ensemble application
  weighted_ensemble.py      # CPU: fit + apply ensemble weights (free/cons/greedy variants)
  weighted_ensemble_per_entity.py  # CPU: fit per-entity weights
  orchestrate.ps1           # waits for training to finish, then runs the pipeline
  make_avg_csvs.py          # quick equal-weight averaging of existing CSVs (CPU, no GPU)
run_one.py                  # wrapper that loads a model file, trains, saves, evaluates
evaluate.py                 # local val MSE + auto-append to results.md
make_split.py               # 80/20 train/val split — writes train_files.txt, val_files.txt
results.md                  # append-only log of local val scores (manually annotated)
```

---

## Reproducing the current best

```powershell
# 1. (one-time) make the train/val split
python make_split.py

# 2. train all model variants — repeat for each v46_sN / v58_sN you want.
#    NOTE: 4 concurrent training jobs is the safe ceiling on a single GPU.
#    8 concurrent caused a TDR-driven driver hang once.
python run_one.py models\nba_v46.py 300 32
python run_one.py models\nba_v58.py 300 32
python run_one.py models\nba_v46_s1.py 300 32   # ... s2, s3, s4, ...

# 3. build the per-model TTA caches and the weighted ensemble:
python misc\make_all_swa.py
python misc\save_val_tta_preds.py
python misc\save_test_tta_preds.py
python misc\weighted_ensemble.py
python misc\weighted_ensemble_per_entity.py

# Or just run the orchestrator instead of steps 3 onward:
powershell -ExecutionPolicy Bypass -File misc\orchestrate.ps1
```

---

## Conventions

- **Model interface:** `model(X)` in eval mode returns `[T, B*N, 2]` normalized positions. `evaluate.py` denormalizes and computes MSE.
- **Loss schedule:** `CosineAnnealingLR(T_max=epochs)`. The schedule must match the exact epoch count, otherwise the run never anneals.
- **Normalization:** computed from `train_files.txt` only — never from the full train dir, to avoid leakage.
- **Local → Kaggle gap:** ~7.8%. A local val of 2.78 lands around 3.00 on the leaderboard.
- **Bold in `results.md`:** only rows that improve on the previous best.
