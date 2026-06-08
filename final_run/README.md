# final_run/ — the FINAL submission (Kaggle 2.95986)

The honest 80/20 pipeline in [`../ensemble`](../ensemble) leaves 20 % of the labelled data unused.
For the final submission we **retrain every ensemble member on 100 % of the labelled data**
(`train + val` = `train_files_full.txt`, 4968 sequences) and reuse the 2-group weights already fit on
the 80/20 split → [`solutions/solution_100pct_full_2grp_NEW.csv`](solutions/solution_100pct_full_2grp_NEW.csv).

This is a thin layer on top of `ensemble/` — it shares the architecture files and the fitted weights,
and only swaps the checkpoints:

```
weights/     The 14 *_full (100%-data) checkpoints (model_<label>_full.pth).
solutions/   solution_100pct_full_2grp_NEW.csv  (the 2.95986 final submission).
_config.py   Paths; reuses ../ensemble/models + ../ensemble/val_preds.
```

The 14 retrained members are `v46, v46_s3, v46_s5, v46_s7, v58, v58_s1/s2/s3` and the six ball
specialists `v60_w3/w5/w5_s1/w8/w8_s1/w8_s2`. Members not retrained (the SWA snapshots) fall back to
their 80 % test predictions — `build_full_candidate.py` handles the prefer-`_full`-else-80 % logic.

> **Why no local number here:** once val is folded into training, the local val score is optimistic
> and meaningless. These models are judged **only on Kaggle**. The 80/20 result (2.98086) is the
> honest, reportable metric; 2.95986 is the leaderboard entry it produced.

## Reproduce (2.95986)

```bash
export NML_DATA=/path/to/nml-2026

# 0. The ensemble/ val + test prediction caches must exist first (the 2-group weights are fit on the
#    80% val preds, and non-retrained members reuse the 80% test preds):
python ensemble/save_val_tta_preds.py
python ensemble/save_test_tta_preds.py

# 1. Retrain each member on 100% of the labelled data -> model_<tag>_full.pth here in weights/.
#    (Checkpoints are committed; this step rebuilds them.)
python final_run/retrain_full.py ensemble/models/nba_v46.py    v46    300
python final_run/retrain_full.py ensemble/models/nba_v60_w8.py v60_w8 300
#    ... repeat for the 14 members listed above.

# 2. TTA test predictions for the _full models -> test_preds/<label>_full.pt
python final_run/save_test_full.py

# 3. Fit 2-group weights on the 80% val preds, apply to the _full test preds, write the CSV.
python final_run/build_full_candidate.py
```
