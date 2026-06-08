# Screencast script — code walkthrough (≤ 5 min, MP4 + voice)

What the rubric wants: briefly explain the **structure/organisation** of the codebase, highlight
**clarity & documentation**, and include a **short demo of the code running**. This script also
covers the five things the oral code review emphasises — **I/O format, architecture design, graph
creation, temporal component, training objective** — so one recording prepares both.

**Record on the machine that has PyTorch + the dataset + the weights (the Windows box).** This Mac
can't run it.

---

## Before you hit record (one-time setup, off-camera)

```powershell
$env:NML_DATA = "C:\Users\jan\.cache\kagglehub\competitions\nml-2026"
python core\make_split.py                  # regenerate train_files.txt / val_files.txt for this machine
# Pre-cache predictions so the live demo is fast (these are the slow GPU steps):
python ensemble\save_val_tta_preds.py
python ensemble\save_test_tta_preds.py
```

Have two things open: a terminal, and the editor showing `demarche/nba_v46.py`.

---

## Storyboard  (SHOW = on screen · SAY = narration)

### 1 · Repo structure — 0:00–0:30
- **SHOW:** `README.md`, then the top-level folder tree.
- **SAY:** "The task: given 8 frames of an NBA possession, predict the next 12 positions of all 11
  entities — 10 players and the ball — scored by mean-squared error. The repo is organised around
  the story. `demarche/` is our journey: one model file per milestone. `ensemble/` is the validated
  pipeline that scored 2.98086. `final_run/` is the final submission, 2.95986, retrained on all the
  data. The full research history — about 90 variants — lives on the `main-archive` branch; this
  branch keeps only what's needed to understand and reproduce the result."

### 2 · The model — the five components — 0:30–2:15
- **SHOW:** `demarche/nba_v46.py`, scrolling to each class as you mention it.
- **SAY (I/O):** "Start with I/O. `NBADataset` slices a 20-frame window from each play: `X` is the
  8 context frames, `y` the 12 future frames. Each entity carries x, y, an is-player flag, and a
  team label. Positions are z-normalised using statistics computed from the **training set only** —
  no leakage — and we augment with random court flips."
- **SAY (architecture):** "`NBAv46Model` is the network. For each entity at each step we build motion
  features — position, velocity, and the same relative to the ball — plus social features, and a team
  embedding. That feeds two layers, each a GRU followed by graph attention, then a mixture head."
- **SAY (graph creation):** "Here's the graph — `TypeAwareDistanceBiasedAttention`. The scene is a
  fully-connected graph of 11 nodes; every entity attends to every other. The attention scores are
  biased by two edge features: the pairwise **distance**, and a learned **type-pair** term — so
  player-to-player and player-to-ball interactions are modelled differently. We also build an explicit
  k-nearest-neighbour social graph: the two closest teammates and two closest opponents."
- **SAY (temporal):** "Temporally, a per-node GRU carries state across all 20 steps. During the
  8 context frames we feed observed positions; for the 12 future frames we feed the model's **own**
  previous prediction — an autoregressive rollout."
- **SAY (training objective):** "And the loss — `MoGNLLLoss` — is a **mixture of two Gaussians**,
  negative log-likelihood, weighted so later timesteps count more. Predicting a distribution instead
  of a point, and then a *mixture*, were our two biggest jumps: from 3.7 to 2.9."

### 3 · The méthode & the ensemble — 2:15–3:15
- **SHOW:** `demarche/README.md` (the journey table), then `docs/ERROR_ANALYSIS.md` briefly.
- **SAY:** "The journey, in one table: a plain GRU scored 54; adding the interaction graph dropped it
  to about 4; the Gaussian and then mixture losses took us to 2.86. The key insight came from error
  analysis — the **ball alone is about 39 % of the total error**, roughly six times any player,
  because it's passed and shot. So we trained ball-specialist models and built a **two-group
  ensemble**: one weight vector for the ball, one shared across players. The optimiser then routes
  the ball-weighted models to the ball and the generalists to the players, automatically."
- **SHOW:** `ensemble/ball_player_ensemble.py` (the POOL list + the two `fit()` calls).

### 4 · Live demo — 3:15–4:40
- **SHOW + RUN:** the fitted weights, to make the specialisation visible:
  ```powershell
  python ensemble\show_weights.py
  ```
  - **SAY:** "These are the fitted weights. Notice the ball column is dominated by the `v60` ball-
    weighted models, while the player column picks the accurate generalists — the split the error
    analysis predicted."
- **SHOW + RUN:** rebuild the actual submission, live:
  ```powershell
  python ensemble\ball_player_ensemble.py
  ```
  - **SAY:** "Running the ensemble fits those weights on the held-out validation set and writes the
    submission CSV — local validation MSE about 2.756, which is the 2.98086 we scored on Kaggle."
- **(Optional, if time)** prove training runs end-to-end with a 2-epoch run:
  ```powershell
  python core\run_one.py demarche\nba_v46.py 2 32
  ```
  - **SAY:** "And training is one command — here's two epochs, with the loss going down."

### 5 · Wrap — 4:40–5:00
- **SHOW:** `final_run/README.md`.
- **SAY:** "For the final submission we retrain every member on 100 % of the labelled data and reuse
  the same two-group weights — that's `final_run`, and it took us to 2.95986, fifth place. Every
  dataset path is configured once through an environment variable, each folder has its own README,
  and the whole pipeline reproduces from these scripts. Thanks for watching."

---

## Timing cheat-sheet

| Section | Budget |
|---|---|
| Repo structure | 0:30 |
| Model / 5 components | 1:45 |
| Méthode + ensemble | 1:00 |
| Live demo | 1:25 |
| Wrap | 0:20 |
| **Total** | **5:00** |

Tips: rehearse once to land under 5:00; if tight, drop the optional training run. Keep the editor
font large. Speak to the five components explicitly — those are the exact terms the oral review uses.
