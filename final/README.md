# final/ — the final submission (Kaggle ADE **2.954**), self-contained

The final model is a **6-member, single-architecture ensemble**: one graph-attention network
(`NBAv46Model`) trained six ways and combined by a **2-group (ball | players)** weighting. There is
no dependency on any other folder — the 6 checkpoints, the architecture, the weights, and the build
script are all here.

## The 6 members and their weights

All six share the **same** network (`models/nba_v46.py`), differing only in training. Three are
**ball specialists** (ball-weighted NLL loss); three are **player generalists** (standard loss,
different seeds). The 2-group fit routes the ball to the specialists and the players to the
generalists — a partition predicted by our per-entity error analysis (the ball is ~6× harder than a
player; see the report).

| Model | Role | Ball weight | Player weight |
|---|---|---:|---:|
| `v60_w8`    | ball specialist (loss ×8)        | **0.606** | 0.000 |
| `v60_w5_s1` | ball specialist (loss ×5, seed 1)| **0.246** | 0.000 |
| `v60_w8_s2` | ball specialist (loss ×8, seed 2)| **0.099** | 0.000 |
| `v46_s3`    | player generalist (seed 3)       | 0.048 | 0.285 |
| `v46`       | player generalist (baseline)     | 0.000 | 0.328 |
| `v46_s7`    | player generalist (seed 7)       | 0.000 | **0.387** |

The ball weights sit entirely on the three `v60` specialists; the player weights entirely on the
three `v46` generalists (with `v46_s3` the sole cross-over). Each member is evaluated under 4-flip
test-time augmentation on its 100 %-data (`_full`) checkpoint.

## Contents

```
build_submission.py     load 6 checkpoints -> 4-flip TTA on test -> 2-group weighting -> solution.csv
ensemble_weights.json   the pool order + the two fitted weight vectors (ball / players)
models/nba_v46.py       the architecture (all 6 members are NBAv46Model)
weights/                the 6 _full (100 %-data) checkpoints
solution.csv            the submitted Kaggle file (ADE 2.954)
```

## Reproduce

```bash
export NML_DATA=/path/to/nml-2026     # folder with train/train/*.pt and test/test/*.pt
python final/build_submission.py      # GPU recommended; writes final/solution.csv
```

`solution.csv` is committed as the authoritative artifact; the script reproduces it up to the
precision of the published weights in `ensemble_weights.json`.
