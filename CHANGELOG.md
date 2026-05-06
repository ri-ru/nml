# Changelog

## [2026-05-06] — Gaussian NLL breakthrough confirmed on Kaggle

Submitted v38 (GRU + Gaussian NLL) and confirmed it obliterates all prior models on Kaggle.

**Kaggle scores:**
- `solution_tta_v38_solo.csv` → **3.68213** ← new best (was 4.05180)
- `solution_tta_v31_v38.csv` → **3.78456**
- `solution_tta_ensemble_v19_v23_v24_v26_v31_v38.csv` → **3.94295**
- `solution_tta_ensemble_v19_v23_v24_v26_v31_v33.csv` → **4.04023**

**Key finding:** Ensembling v38 with older MSE models *hurts* — they drag down the predictions. v38 solo with 4-variant TTA is the best approach. The local→Kaggle gap for v38 is only 0.24 (3.44 local → 3.68 Kaggle), much tighter than the old ~0.35 gap.

---

## [2026-05-05] — Gaussian NLL loss, proper train/val split, v37–v40

### Breakthrough: Gaussian NLL loss (v38)
Replaced MSE loss with diagonal Gaussian NLL. Model outputs (μ_x, μ_y, log σ_x, log σ_y) per timestep. Only μ is used at inference. Result: **3.4355 local MSE at just 35 epochs** vs v31's 3.7319 at 200 epochs. A 0.29 improvement with 6× fewer training steps.

### Proper data split
Previous models (v1–v36) were evaluated on the same data they trained on — contaminated. Introduced `train_files.txt` (3974) / `val_files.txt` (994) for honest evaluation. All v38+ use this split.

### Models this session
- **v33**: v26 arch × 250 epochs → 3.7471 (worse than 200ep, MSE scaling plateau confirmed)
- **v37**: Non-autoregressive Transformer + NLL → 3.8641 (Transformer slower to converge)
- **v38**: GRU (128d, 2L) + Gaussian NLL × 35ep → **3.4355** (breakthrough)
- **v39**: GRU (192d, 3L) + NLL × 35ep → 6.3680 (late bloomer, needs 150ep)
- **v40**: Transformer + NLL × 35ep → 3.9194

### TTA pipeline for Gaussian models
`make_tta_v38.py` — 4 TTA variants (original, x-flip, y-flip, xy-flip). Gaussian models in `eval()` mode return only μ, so TTA works identically to MSE models.

---

## [2026-05-02] — Epoch scaling, court symmetry augmentation, v17–v34

### Epoch scaling (v19, v23, v31)
Longer training consistently improved MSE models: v15 (35ep) → v19 (100ep) → v23 (150ep) → v31 (200ep). Gains were real. 250ep (v33) showed plateau.

### Court symmetry augmentation (v26, v30)
Added random x-flip (p=0.5) and y-flip (p=0.5) during training. Improved v24 → v26 by ~0.9% at same epoch count. Kept in all subsequent models.

### Notable dead ends
- **v22**: Decoder self-attention over prior steps — redundant, GRU hidden state already captures history
- **v27**: Scaled model (256d, 3L) × 35ep → 6.94 (late bloomer)
- **v28**: Explicit acceleration features → no gain, GRU differentiates implicitly
- **v29**: 16-frame context (0.8s) → worse, longer context degrades GRU hidden state quality
- **v32**: Dropout (p=0.15) in decoder → 4.82 (+29% regression), corrupts autoregressive rollout
- **v34**: EMA shadow weights → superseded by NLL breakthrough

### Best model this phase
**v31**: v26 arch (K=1 social + ball-rel + augmentation) × 200 epochs → **3.7319 local, 4.11950 Kaggle**

---

## [2026-05-01] — Feature engineering, social context, v1–v16

### Starting point
Baseline GRU: 54.01. Each player predicted independently, no interactions.

### Social graph (v6, v12, v15)
- Ball-relative features + typed attention bias (v6): 4.36 — biggest single jump
- K=1 nearest teammate + opponent (v12): 4.16
- K=2 nearest (v15): 4.04 — social context is the main signal

### Dead ends
- **v5**: Teacher forcing → 11.95 (train/inference gap fatal)
- **v7**: Temporal Transformer context encoder → worse (GRU handles 8-frame context fine)
- **v8**: Displacement prediction → worse (absolute position easier to train)
- **v13**: All features combined → worse (more features without capacity = noise)
- **v14, v16**: Directional attention → NaN (gradient explosion at dist→0)

### First Kaggle submission
v6 → 4.63615. Confirmed local→Kaggle correlation.
