# NBA Trajectory Prediction — NML 2026

Multi-agent trajectory forecasting for 11 NBA players.  
Given 8 context frames, predict x,y positions over the next 12 timesteps (0.4s ahead).  
Scored by MSE in original court coordinates (feet). Lower is better.

---

## Best Results

| Submission | Kaggle MSE | Notes |
|---|---|---|
| **v38 solo TTA** | **3.68213** | **Current best** |
| v31 + v38 TTA ensemble | 3.78456 | |
| 6-model TTA (v19–v31 + v38) | 3.94295 | Older models dilute v38 |
| 6-model TTA (v19–v31 + v33) | 4.04023 | Pre-NLL ensemble |
| 5-model TTA (v19–v31) | 4.05180 | Previous best |

**Key insight:** v38 (GRU + Gaussian NLL) is so much stronger than all earlier models that ensembling with them hurts. v38 solo TTA is the best submission.

---

## Architecture — v38 (Breakthrough)

- **Encoder:** GRU over 8 context frames, per-player
- **Social context:** K=1 nearest teammate + K=1 nearest opponent relative pos/vel; ball-relative features; type-aware attention bias
- **Decoder:** Autoregressive GRU rollout, 12 steps
- **Loss:** Diagonal Gaussian NLL — outputs (μ_x, μ_y, log σ_x, log σ_y); inference returns only μ
- **Augmentation:** Random x-flip (p=0.5) and y-flip (p=0.5) for court symmetry
- **Data split:** Proper train/val split via `train_files.txt` (3974) / `val_files.txt` (994)
- **Training:** 35 epochs, lr=1e-3, CosineAnnealingLR, batch=32

The Gaussian NLL loss lets the model express per-step uncertainty instead of hedging toward the mean on hard timesteps. Train loss goes negative (e.g. −2.19 at epoch 35) — this is correct behavior.

---

## Model History

| Date | Model | Local MSE | Kaggle MSE | Key change |
|---|---|---|---|---|
| 2026-05-01 | Baseline GRU | 54.01 | — | Naive GRU, no interactions |
| 2026-05-01 | v2 dist-GAT | 9.26 | — | Distance-biased multi-head attention |
| 2026-05-01 | v3 | 6.03 | — | +Velocity features |
| 2026-05-01 | v6 | 4.36 | 4.64 | +Ball-relative + typed attention bias |
| 2026-05-01 | v12 | 4.16 | 4.46 | +K=1 nearest social features |
| 2026-05-01 | v15 | 4.04 | 4.36 | +K=2 nearest social features |
| 2026-05-02 | v19 | 3.84 | 4.22 | v15 × 100 epochs |
| 2026-05-02 | v23 | 3.77 | — | v15 × 150 epochs |
| 2026-05-02 | v26 | 3.79 | — | +Court symmetry augmentation |
| 2026-05-02 | v31 | 3.73 | 4.12 | v26 × 200 epochs |
| 2026-05-05 | v33 | 3.75 | — | v26 × 250 epochs (plateau) |
| 2026-05-05 | v37 | 3.86 | — | Non-autoregressive Transformer + NLL |
| **2026-05-05** | **v38** | **3.44** | **3.68** | **GRU + Gaussian NLL — breakthrough** |
| 2026-05-05 | v39 | 6.37 | — | Bigger GRU (192d, 3L) + NLL — late bloomer, needs 150ep |
| 2026-05-05 | v40 | 3.92 | — | Transformer + NLL |

---

## What Doesn't Work

- **Dropout in autoregressive rollout** (v32): corrupts GRU hidden state, +29% error
- **Teacher forcing** (v5): train/inference gap, score collapsed to 11.95
- **Displacement prediction** (v8): error accumulation worse than absolute position
- **Very large models at 35 epochs** (v27, v39): late bloomers — need 150+ epochs
- **Epoch scaling past 200ep with MSE** (v33): diminishing returns, got slightly worse
- **More features without capacity** (v13): centroid + ball-align features added noise

---

## Running Experiments

```bash
# Train a model (epochs, batch_size)
python run_one.py models/nba_v38.py 35 32

# Generate TTA submission
python make_tta_v38.py

# Evaluate
python evaluate.py
```

All new models must use the proper data split:
```python
TRAIN_TXT = _DIR.parent / 'train_files.txt'   # 3974 files
VAL_TXT   = _DIR.parent / 'val_files.txt'     # 994 files
```

---

## Next Experiments

1. **v39 × 150 epochs** — bigger GRU (192d, 3L) + NLL, needs time to converge
2. **v40 × 100 epochs** — Transformer + NLL, may also be a late bloomer
3. **v41** — v38 arch + timestep-weighted NLL (linearly increasing weights)
4. **v42** — v38 arch + K=2 nearest neighbors instead of K=1
