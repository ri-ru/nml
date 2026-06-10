# NBA Multi-Agent Trajectory Prediction (EE-452, NML 2026)

In this repo, we use an adapted version of the NBA highlight dataset introduced in SocialVAE (Xu et al., ECCV 2022, https://arxiv.org/abs/2203.08207). Given 8 context frames of an NBA possession, the goal is to predict the next 12 positions of all 11 entities (10 players + the ball) on the court. Scored by mean-squared error in court coordinates (feet), lower is better, averaged uniformly over entities, timesteps, and axes. 

We used a single graph-attention network (spatio-temporal GAT + GRU + mixture-of-Gaussians head), trained six ways under one architecture and then combined with a 2-group ensemble that specialises models to the ball vs. the players.

Our result on kaggle is 2.954 (for now....) for the 6-model, single-architecture ensemble: 3 ball specialists + 3 player generalists, members retrained on 100 % of the labelled data).  




## Setup

```bash
pip install -r requirements.txt            # torch, numpy, pandas, scipy, tqdm
```
A CUDA GPU is needed for training and prediction; the ensemble weight-fitting is CPU-only.

## Reproduce

Everything runs through one launcher, `run.py`, which prints each underlying command before it
runs it. `python run.py` with no args lists every subcommand.

**A. Fastest: rebuild both submissions from the committed `_full` weights** 

```bash
python run.py submit               # 6-model ensemble  -> outputs/solutions/solution_ensemble_6model.csv
python run.py submit-single v46    # single model      -> outputs/solutions/solution_v46_full.csv
```

**B. Everything from scratch(careful for gpu)**: retrain all 6 members twice (an 80% model to fit the weights honestly
+ a 100% `--full` model for the submission), re-fit the weights, rebuild the CSVs. One command:

```bash
python run.py reproduce            # GPU, hours: split -> train-all (80% + full) -> fit-weights -> submit
```

…which is exactly this, if you'd rather run the steps yourself:

```bash
python run.py split                          # fixed-seed 80/20 manifests for THIS machine
python run.py train-all                      # GPU: all 6 members, 80% split   -> model_<m>_300ep.pth
python run.py train-all --full               # GPU: all 6 members, 100% data   -> model_<m>_full.pth
python run.py fit-weights                    # GPU+CPU: cache val/test preds, fit 2-group weights
python run.py submit                         # CPU: ensemble CSV
python run.py submit-single v46              # single-model CSV
```

**C. Train one model** (any ensemble member, or a milestone file):

```bash
python run.py train v46 300                  # one member, 80% split  -> model_v46_300ep.pth
python run.py train v46 300 --full           # same member, 100% data -> model_v46_full.pth
python utils/run_one.py demarche/nba_v38.py 300 32   # a milestone (baseline/v6/v38), 80% split
```
 



---

## Repo structure (for the curious)

```
run.py       One launcher for everything (train / ensemble / both submissions). 
utils/       Shared infrastructure (split / train / score).
demarche/    The construction paths taken and our final best model (demarche/nba_v46.py);
             the 80/20 validated ensemble pipeline lives in demarche/ensemble/. Must be rerun. 
final/       The two final Kaggle submissions  
outputs/     weights/ (trained checkpoints , no 80/20 split) + solutions/ (the submitted CSVs).
```

### `utils/`: shared infrastructure

- `make_split.py`: makes the 80/20 train/val split once and writes
  `train_files.txt` + `val_files.txt` so every model uses the same split. 
- `run_one.py`: the training entry point: `python utils/run_one.py <model.py> <epochs> <batch>`.
  Imports the model's `Trainer`, trains, saves an checkpoint with epoch tagged, and evaluates.
- `evaluate.py`: local scorer:  

### `demarche/`: how we got here... 

| Step | File | What's new |  
|---|---|---|
| 0 Baseline | `nba_baseline.py` | provided non-graph GRU, each entity independent (MSE) | 
| 1 Graph | `nba_v6.py` | graph attention + ball-relative features | 
| 2 Gaussian | `nba_v38.py` | Gaussian NLL (predict a distribution) + kNN social features | 
| 3 Mixture | `nba_v46.py` | mixture-of-2-Gaussians head + timestep-weighted NLL, 300 ep | 
| 4 Ball-weighting | `nba_v60_base.py` | the ball-weighted loss + trainer factory; ball specialists are trained via `demarche/ensemble/train_member.py` | 


### `demarche/ensemble/`: 80/20 split 

We found from error analysis: ball alone is ~39 % of the total error. So we fit two weight vectors and optimiser routes the ball-weighted models to the ball and generalists to the players. 
(~6× any player). So we fit *two* weight vectors (one for the ball, one shared across players), and
the optimiser routes ball-weighted models to the ball and generalists to the players.

- `_config.py`: paths and the `MEMBERS` registry (the 6 members, each defined by a label, seed,
  and ball-weight).
- `train_member.py`: trains one member by name.  
  ```bash
  python demarche/ensemble/train_member.py v46_s3 300          # 80% split -> model_v46_s3_300ep.pth
  python demarche/ensemble/train_member.py v46_s3 300 --full   # 100% data -> model_v46_s3_full.pth
  ```
- `save_val_tta_preds.py`: runs the 6 (80%) members on the val split with 4-flip TTA and dumps
  predictions to `val_preds/` for cpu reasons... 
- `ball_player_ensemble.py`: least-squares fit of the 2-group weights on those cached val
  predictions, printed so they can be pasted into `final/ensemble_weights.json`.

### `final/`: kaggle submissions

- `build_ensemble_6model.py`: the 6-model ensemble (3 ball specialists `v60_*` + 3 player
  generalists `v46*`), combined by the 2-group weights. Results in Kaggle ADE 2.954.
- `build_single_model.py [label]`: a single model, no ensembling (default `v46`). Modular: works
  for any member, e.g. `python final/build_single_model.py v60_w8`.
  
### `outputs/`: weights & submissions

`outputs/weights/` holds every trained checkpoint; the two CSVs submitted to Kaggle live in
`outputs/solutions/`:
- `solution_ensemble_6model.csv`: the 6-model ensemble (ADE 2.954).
- `solution_v46_full.csv`: a single v46 (100 % data, 4-flip TTA, no ensemble).

---
 