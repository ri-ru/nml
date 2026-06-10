# Démarche — how we got here

This folder holds **one model file per milestone**: the steps that actually moved the score, with
the trained weight for each. Everything in between (≈80 variants and dead-ends) lives on
`main-archive`. Read top-to-bottom to see the reasoning; local MSE is on the 80/20 val split
(lower is better).

| # | Step / idea | File | Weight | Local MSE | Why it's a stepping stone |
|---|---|---|---|---|---|
| 0 | Independent GRU, no interactions | `nba_baseline.py` | `model_baseline.pth` | **54.0** | The non-graph baseline (the mandatory comparison point). |
| 1 | **Graph core**: ball-relative features + typed attention bias | `nba_v6.py` | `model_v6.pth` | **4.36** | Biggest single jump — the graph is born. First Kaggle (4.64). |
| 2→3 | Social kNN (2 nearest mates/opponents) + epoch scaling + court-symmetry augmentation | *(folded into v46)* | — | **3.73** | Longer cosine schedule beat all further feature engineering. |
| 4 | **Gaussian NLL loss** (+ honest train/val split) | `nba_v38.py` | `model_v38.pth` | **3.44** | Breakthrough #1 — predict a distribution, not a point. |
| 5 | **Mixture-of-Gaussians head (K=2)** + timestep-weighted NLL + 300 ep | `nba_v46.py` | `model_v46_300ep.pth` | **2.86** | Breakthrough #2 — models bimodal futures. **The canonical model.** |
| 6 | **Ball-weighting** (up-weight the ball 8× in the loss) | `nba_v60_base.py`, `nba_v60_w8.py` | `model_v60_w8_300ep.pth` | — | Our contribution: error analysis showed the ball is ~6.3× any player's error, so we train ball specialists for the ensemble. |

`nba_v46.py` is the architecture all later models share — steps 1–5 are cumulative inside it. The
ball-weighted models (step 6) reuse the v46 architecture with a re-weighted loss defined in
`nba_v60_base.py`.

## The 5 components a grader will probe (all in `nba_v46.py`)

| Component | Where |
|---|---|
| **I/O format** | `NBADataset.__getitem__` (X=8 ctx, y=12 horizon); normalization `_compute_normalization`; output in `forward` (eval → `[12, N, 2]`). |
| **Architecture** | `NBAv46Model` — 2 layers of GRU + graph-attention; MoG head `self.proj`. |
| **Graph creation** | `TypeAwareDistanceBiasedAttention` — fully-connected attention over the 11 nodes, biased by pairwise distance + a learned type-pair term; plus an explicit kNN social graph in `_social_features`. |
| **Temporal component** | per-node `GRUCell` carried across all 20 steps; autoregressive rollout (feeds its own prediction for t ≥ 8). |
| **Training objective** | `MoGNLLLoss` — mixture-of-2-Gaussians NLL, timestep-weighted (later steps up to 2×). |

## Train one of these

```bash
export NML_DATA=/path/to/nml-2026
python core/run_one.py demarche/nba_v46.py 300 32   # <model.py> <epochs> <batch>
```
