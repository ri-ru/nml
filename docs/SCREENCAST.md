# Screencast script — code walkthrough (≤ 5 min, MP4 + voice)

What the rubric wants: briefly explain the **structure/organisation** of the codebase, highlight
**clarity & documentation**, and include a **short demo of the code running**. This script also
hits the five things the **oral code review** emphasises — **I/O format, architecture design, graph
creation, temporal component, training objective** — so recording it once prepares both.

**Record on the machine that has PyTorch + the dataset + the GPU (the Windows box).** This Mac
can't run it. One-time setup, off-camera:

```powershell
$env:NML_DATA = "C:\path\to\nml-2026"
python core\make_split.py                  # regenerate train_files.txt / val_files.txt for this machine
python ensemble\save_val_tta_preds.py      # pre-cache val preds so the live demo is instant
```

Have two things open: a terminal, and the editor on `demarche/nba_v46.py`.

---

## Storyboard  (SHOW = on screen · SAY = narration)

### 1 · The project — 0:00–0:40
- **SHOW:** `README.md`, then the top-level folder tree.
- **SAY:** "The task: given 8 frames of an NBA possession — 10 players and the ball — predict the
  next 12 positions of all 11 entities, scored by mean-squared error, the ADE metric. The repo is
  organised as a story. `demarche/` is our journey: one model file per milestone. `ensemble/` is the
  validated pipeline. `final/` is the final submission — Kaggle 2.954. `core/` is shared training
  infrastructure. Every dataset path is configured once through the `NML_DATA` environment variable,
  and each folder has its own README."

### 2 · Where things are — 0:40–1:10
- **SHOW:** expand `demarche/` and `final/` in the file tree.
- **SAY:** "There's really one model. `demarche/nba_v46.py` is the architecture; everything else is
  that same network trained differently. `nba_baseline` is the provided non-graph GRU; `v6` adds the
  interaction graph; `v38` adds a Gaussian loss; `v46` adds the mixture head — that's the canonical
  model; `v60` adds ball-weighting. The final ensemble is six checkpoints of this one architecture,
  combined in `final/build_submission.py`. So I'll open `nba_v46.py` and walk the five key
  components, then show the ensemble."

### 3 · `nba_v46.py` — the five components — 1:10–3:40
- **SHOW:** `demarche/nba_v46.py`, scrolling to each class as you name it.
- **SAY (I/O format):** "First, I/O. `NBADataset` — line 27 — slices a 20-frame window from each
  play: `X` is the 8 context frames, `y` the 12 future frames. Each entity carries x, y, an
  is-player flag, and a team label. Positions are z-normalised using statistics computed from the
  **training set only** — no leakage — and we augment with random court flips. The model's `forward`
  returns the predicted positions for the 12 horizon steps, two coordinates per entity."
- **SAY (architecture design):** "`NBAv46Model` — line 165 — is the network. For each entity at each
  step we build motion features, social features, and a team embedding, then pass them through **two
  layers, each a GRU followed by graph attention**, and finally a mixture-of-Gaussians head."
- **SAY (graph creation):** "Here's the graph — `TypeAwareDistanceBiasedAttention`, line 132. The
  scene is a fully-connected graph of 11 nodes; every entity attends to every other. The attention
  scores are biased by two edge features: the pairwise **distance** — line 149 — and a learned
  **type-pair** term — line 152 — so player-to-player and player-to-ball edges are modelled
  differently. On top of that, `_social_features` at line 207 builds an explicit k-nearest-neighbour
  graph: the two closest teammates and two closest opponents."
- **SAY (temporal component):** "Temporally — the rollout loop at line 259 — a per-node `GRUCell`
  carries hidden state across all 20 steps. For the 8 context frames we feed observed positions; for
  the 12 future frames we feed the model's **own** previous prediction back in — an autoregressive
  rollout, identical at train and test time."
- **SAY (training objective):** "And the loss — `MoGNLLLoss`, line 85 — is the negative
  log-likelihood of a **mixture of two Gaussians**, weighted so later timesteps count more. Moving
  from an MSE point prediction to a Gaussian, and then to a *mixture*, were our two biggest jumps —
  from 3.7 down to 2.9 — because a mixture can represent the small number of distinct futures in a
  play instead of averaging them into a blur."

### 4 · The ensemble — and running it — 3:40–4:50
- **SHOW:** `ensemble/ball_player_ensemble.py` — the `POOL` list (6 models) and the two `fit()` calls.
- **SAY:** "The final result comes from an ensemble. The key insight is from error analysis: the
  **ball alone is about 39 % of the total error** — roughly six times any player — because it's
  passed and shot. So we train three ball-specialist models with a ball-weighted loss, and fit a
  **two-group ensemble**: one weight vector for the ball, one shared across the players."
- **SHOW + RUN:** the fitted weights, to make the specialisation visible:
  ```powershell
  python ensemble\show_weights.py
  ```
  - **SAY:** "These are the fitted weights. The ball column is dominated by the v60 ball-weighted
    models; the player column picks the v46 generalists — the split the error analysis predicted,
    recovered automatically by the optimiser."
- **SHOW:** `final/build_submission.py` + `final/solution.csv`.
  - **SAY:** "And `final/build_submission.py` loads the six 100 %-data checkpoints, runs four-flip
    test-time augmentation, applies those two weight vectors — ball weights to entity zero, player
    weights to the rest — and writes `solution.csv`: our Kaggle 2.954."

### 5 · Wrap — 4:50–5:00
- **SAY:** "So: one graph-attention network, trained six ways, combined by an error-driven two-group
  ensemble. Every folder has a README, paths are configured once, and the whole pipeline reproduces
  from these scripts. Thanks for watching."

---

## Timing cheat-sheet

| Section | Budget |
|---|---|
| The project | 0:40 |
| Where things are | 0:30 |
| nba_v46 — 5 components | 2:30 |
| Ensemble + live demo | 1:10 |
| Wrap | 0:10 |
| **Total** | **5:00** |

Tips: rehearse once to land under 5:00. Keep the editor font large. Speak the five component names
explicitly — "I/O format", "architecture", "graph creation", "temporal", "training objective" — as
those are the exact terms the oral review uses.
