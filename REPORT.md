# NBA Multi-Agent Trajectory Prediction — Method Report
### EE-452 Network Machine Learning · nml-2026 Kaggle

**Task.** Given 8 context frames of an NBA possession, predict the next 12 positions of all
**11 entities** (10 players + the ball). Scored by mean-squared error in court coordinates (feet),
**lower is better**, averaged uniformly over all entities, timesteps, and axes.

**Result.** Kaggle **2.98086** (5th place). The method is **a single graph neural network**, trained
under four principled regimes and combined by an **error-analysis-driven per-entity ensemble**.

---

## 1. Data

| Split | Sequences | Use |
|---|---|---|
| Train | 3 974 (80%) | model training |
| Val | 994 (20%) | local scoring + ensemble-weight fitting |
| Test | 1 243 | Kaggle (labels hidden) |

Positions are z-normalized using statistics computed **only from train** (no leakage). Court
symmetry is exploited via x/y flips (training augmentation and test-time augmentation).

---

## 2. Model — a spatio-temporal graph attention network

The scene at each timestep is a **graph**: nodes = 11 entities, edges = all pairwise interactions.
One architecture underlies the entire system:

```
        per-entity input each step
   [ motion: pos, vel, pos-ball, vel-ball ]
   [ social: rel. pos/vel of 2 nearest teammates + 2 nearest opponents ]
   [ team embedding ]
                 │
                 ▼
        ┌──────────────────┐   ×2 layers
        │  GRU cell (128)  │   (temporal state per node)
        └──────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────────┐
   │ Type-/Distance-Biased Graph Attention│  ← the GNN interaction:
   │  attention over all 11 nodes, biased │    nodes attend to each other,
   │  by pairwise distance + type-pair    │    biased by geometry & role
   └─────────────────────────────────────┘
                 │   (autoregressive rollout, 12 steps)
                 ▼
        ┌──────────────────┐
        │ Mixture-of-Gauss │  K=2 components / entity / step
        │   output head    │  → captures bimodal futures
        └──────────────────┘
```

- **Graph attention (`TypeAwareDistanceBiasedAttention`)** is the network-ML core: every entity
  attends to every other, with the attention logits biased by **pairwise distance** and a learned
  **type-pair** term (player↔player, player↔ball). This is a graph attention network, hand-rolled
  rather than via `torch_geometric`.
- **GRU** carries each node's temporal state; rollout is autoregressive over the 12 future steps.
- **Mixture-of-Gaussians head (K=2):** outputs a 2-component predictive distribution per entity/step,
  trained with a **timestep-weighted negative log-likelihood** (later steps weighted up to 2×). At
  inference we return the mixture mean (MSE-optimal point estimate).

**Why MoG mattered (key ablation):** replacing MSE with a single Gaussian NLL improved the score
3.73 → 3.44; going to a 2-component mixture improved it further to 2.90. The mixture lets the model
represent bimodal futures (e.g., a defender switching vs. staying) instead of averaging them away.

---

## 3. The central finding: the ball dominates the error

We decomposed the validation error **per entity**. The result drove the rest of the design:

| Group | MSE | Share of total error |
|---|---|---|
| **Ball** (1 entity) | **12.0** | **~39%** |
| Players (10 entities) | 1.9 | ~61% |

- The ball has **~6.3× the error of any player**, despite being 1 of 11 entities.
- It is worst at **short horizons** (t=1–2 the ball is **58–74%** of all error): players move smoothly
  and are predicted near-perfectly, but the ball is **passed and shot** — abrupt events not present in
  the 8 observed frames.
- The model **under-moves the ball by ~25%** (predicted net displacement = 0.75× actual): facing an
  unobservable pass target, hedging toward the short expected move is in fact MSE-optimal. This error
  is therefore **largely irreducible aleatoric uncertainty**.

This single analysis motivated both **ball-weighted training** and the **per-entity ensemble** below.

---

## 4. Training regimes (one model, four principled variations)

We do not train 27 different models — we train the **one architecture** along four standard axes:

| Regime | Technique | Justification |
|---|---|---|
| Seeds (×8) | different init + data order | ensemble variance reduction (decorrelated members) |
| SWA | stochastic weight averaging (last epochs) | flatter minima → better generalization |
| Pseudo-labels | retrain on train + test labeled by a prior ensemble | semi-supervised use of the unlabeled test distribution |
| **Ball-weighted loss** | up-weight the ball entity 5–8× in the NLL | **targets the dominant error source (§3)** |

All use 300-epoch training, cosine LR (1e-3), weight decay 5e-4, batch 32.

---

## 5. Inference: TTA + per-entity ensemble

1. **Test-time augmentation:** each model predicts under 4 court reflections (orig, x-, y-, xy-flip),
   averaged in real coordinates (+1.6% over no TTA).
2. **Per-entity weighted ensemble.** Instead of one global weight per model, we fit **separate
   constrained weights per entity** (weights ≥ 0, sum to 1) by least-squares on the validation set.

**The ensemble rediscovers the §3 diagnosis.** With no prompting, the optimizer **partitions the pool
into disjoint "ball experts" and "player experts":**

| | Models selected (weight) |
|---|---|
| **Ball** (entity 0) | `v60_w8` 0.61, `v60_w5_s1` 0.25, `v60_w8_s2` 0.10 — **all ball-weighted models** |
| **Players** | `v46_s7` 0.29, `v58` 0.21, `v46_s3` 0.16, `v46` 0.15, … — **accurate generalists** |

Ball-weighted models get **0** player weight and generalists get **0** ball weight. Of 27 trained
variants the optimizer keeps **~13** (the rest are redundant). A 2-group variant (one ball vector +
one shared player vector) regularizes the player weights and generalizes best on the leaderboard.

---

## 6. Results

| Submission | Local val MSE | Kaggle MSE |
|---|---|---|
| Single GRU baseline (early) | — | ~4.0 |
| v46 graph-attention + MoG (300ep) + TTA | 2.818 | 3.037 |
| Equal-weight 5-model TTA ensemble | 2.791 | 3.014 |
| Global-weighted 21-model ensemble | 2.787 | 3.007 |
| **Per-entity ensemble + ball specialists** | **2.752** | **2.980** |
| **2-group (ball \| players) ensemble** | 2.756 | **2.98086** ← best, 5th place |

Local→Kaggle gap is stable (~7.9% for global/robust weights; ~8.3% for the higher-variance
per-entity fit). Progression of ideas: **Gaussian NLL → Mixture head → 300-epoch schedule →
TTA ensemble → ball-weighting + per-entity ensemble.**

---

## 7. Negative results (what we ruled out)

Rigorously testing alternatives bounded the problem:

- **Architectures that did *not* beat the graph-attention model:** agent-centric / vectorized
  transformer (4.1–4.7), message-passing GNN / R-GCN (2.96), two-stage refinement (learned the
  identity — no gain), bigger/deeper nets, Transformer + MoG, occupancy heatmap, diffusion, CVAE,
  goal-conditioning.
- **Ball-specific levers that lost to plain ball-weighting:** extra mixture modes for the ball,
  possession/event features, hard-example (tail-weighted) loss. None reduced the 0.75 hedging ratio.
- **No data leakage:** nearest-neighbour retrieval of test contexts against all training windows gives
  7.6 ft RMSE (no overlap) — the result is not exploitable; it is a genuine modeling task.

**Interpretation.** The convergence of these negatives indicates an **information ceiling**: at 8
frames of context the graph-attention model already extracts essentially all available signal, and
the residual (dominated by the ball) is irreducible. Further gains require *more information*
(more data / semi-supervision), not a cleverer network — consistent with the modest gain observed
when retraining on 100% of the labeled data.

---

## 8. Summary

A **single graph attention network** with a mixture-of-Gaussians head, trained under four standard
regimes (seeds, SWA, pseudo-labels, ball-weighting) and combined by a **per-entity ensemble** that
**automatically specializes models to the ball vs. the players** — a partition predicted by our
per-entity error analysis. Final score **2.98086 (5th)**, with a thorough negative-results study
establishing an information ceiling for the 8-frame setting.
