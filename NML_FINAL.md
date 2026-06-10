# NBA Trajectory Prediction — Final Walkthrough

Hi — this is our code for the NBA trajectory project. **Final Kaggle score: 2.954.**

For an overview: our method is a single **graph-attention network** — more precisely a
spatio-temporal GAT with a GRU and a mixture-of-Gaussians head — that is trained several ways and
combined by an **error-analysis-driven, per-entity ensemble**.

---

## How to run

Everything reproduces from **one command** and **one environment variable** (or a folder). First point
the code at the dataset — the folder that contains `train/train/*.pt` and `test/test/*.pt`:

```bash
export NML_DATA=/path/to/nml-2026      # …or just drop/symlink the dataset at ./data/nml-2026
pip install -r requirements.txt        # torch, numpy, pandas, scipy, tqdm
```

Then a single launcher, `run.py`, drives everything. It prints each underlying command before it runs
it, so nothing is hidden:

```bash
# Rebuild both submitted CSVs from the committed weights (fast — no training):
python run.py submit             # 6-model ensemble  -> outputs/solutions/solution_ensemble_6model.csv  (2.954)
python run.py submit-single v46  # single model      -> outputs/solutions/solution_v46_full.csv

# Train (GPU):
python run.py train v46 300          # one member, 80% split   -> model_v46_300ep.pth
python run.py train v46 300 --full   # one member, 100% data   -> model_v46_full.pth
python run.py train-all --full       # all 6 members on 100% of the data

# Re-fit the ensemble weights from scratch, or run the whole pipeline end to end:
python run.py fit-weights        # cache val predictions, least-squares fit the 2-group weights
python run.py reproduce          # split -> train-all (80% + full) -> fit-weights -> both submissions
```

`python run.py` with no arguments lists every subcommand. The committed
`outputs/solutions/solution_ensemble_6model.csv` is the submitted artifact (Kaggle 2.954).

---

## File structure

```
run.py        One launcher for everything (train / ensemble / both submissions). Start here.
utils/        Shared infrastructure: make_split.py (80/20 split), run_one.py (training runner),
              evaluate.py (local scorer, same MSE metric as Kaggle).
demarche/     One model file per milestone — the construction path — plus the ONE architecture.
final/        The two final Kaggle submissions, self-contained.
outputs/      weights/ (every trained checkpoint) + solutions/ (the submitted CSVs).
```

**`demarche/` — the milestones.** One file per breakthrough; the weights live in `outputs/weights/`.
We tested many variants and kept only the ones that correspond to the big milestones:

- **`nba_v6.py`** — our first model after the baseline: ball-relative features + typed attention.
- **`nba_v38.py`** — the next breakthrough: predict a **Gaussian distribution** instead of a point.
- **`nba_v46.py`** — **our core architecture** (walked through below). Every later component imports it.
- **`nba_v60_base.py`** — the **ball specialists**: the same network as v46, but trained with a
  **ball-weighted loss** (after we found most of the error came from the ball).

**`demarche/ensemble/`** — the 80/20-validated ensemble pipeline: `train_member.py` (train a member),
`save_val_tta_preds.py` (cache predictions on the held-out 20%), `ball_player_ensemble.py` (fit the
2-group weights). `_config.py` holds the 6-member registry.

**`final/`** — self-contained: the two build scripts (`build_ensemble_6model.py` and
`build_single_model.py`), the frozen ensemble weights (`ensemble_weights.json`), and one architecture
file loaded from `demarche/nba_v46.py`. The weights themselves are in `outputs/weights/`.

---

## Walkthrough of v46 (`demarche/nba_v46.py`)

### I/O — `NBADataset`

Each play is a tensor of **frames × 11 entities × features** — the features being `x`, `y`, an
**is-ball flag**, and a **team label**. We slice a 20-frame window: `x` is the **8 context frames**,
`y` is the **12 future frames**. Positions are **z-normalised** with statistics computed on the
training set, and we **augment by randomly flipping the court** in the x and/or y directions.

At inference the model returns the 12 predicted `(x, y)` positions per entity, which the build script
**denormalises back to feet** and writes in the Kaggle column format.

### Architecture — `NBAv46Model`

Per entity, per timestep, we build **motion features** — position, velocity, and both of those
**relative to the ball** — concatenate **social features** and a **learned team embedding**, and feed
that into **two stacked blocks**, each a **GRU cell followed by graph attention**. The head outputs
**5 numbers per mixture component** (mean x/y, log-σ x/y, and a mixture logit), with **2 components**.

### Graph attention — `TypeAwareDistanceBiasedAttention`

The graph is **fully connected over the 11 nodes**, and **two edge biases** are added to the attention
logits: a **projection of the pairwise distance** and a **learned type-pair embedding**. So
**player–player, player–opponent, and player–ball** edges are scored differently.

On top of that, **`_social_features`** builds an explicit **k-nearest-neighbour graph**: each player's
**two closest teammates and two closest opponents**, as relative positions and velocities.

### Rollout — `NBAv46Model.forward`

Temporally, a **per-node GRU** carries hidden state across all 20 steps. For the **8 context frames**
it sees the **observed positions**; **from frame 9 on, it is fed the model's own previous predicted
mean**. This teaches the model to **recover from its own drift**.

### Objective — `MoGNLLLoss`

We found the **negative log-likelihood of a mixture of two Gaussians** works best, with **timesteps
weighted linearly from 1 to 2** so far-horizon errors count more. The v60 ball specialists reuse
**exactly this loss** but **up-weight the ball entity**.

---

## The ensemble (`final/build_ensemble_6model.py`)

We ran an error analysis and found that the **ball alone is about 39 % of the total error** — roughly
**six times the average player**. So instead of one weight vector we fit **two**: one for the **ball**,
one **shared by all ten players**.

The fit (least-squares, weights ≥ 0 summing to 1, on the honest 20% validation predictions) lands the
**ball weights almost entirely on the three ball-weighted v60 specialists**, and the **player weights
on the three v46 generalists**. At inference, **entity 0 (the ball) gets the ball-weight vector**, and
**entities 1–10 get the player vector**.

The build script loads the six 100%-data checkpoints, runs 4-flip TTA on the test set, applies these
two weight vectors, and writes the final CSV — **Kaggle 2.954**.

Thanks!
