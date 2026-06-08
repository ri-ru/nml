"""Paths for the FINAL 100%-data submission run.

This run produces ``solution_100pct_full_2grp_NEW.csv`` (Kaggle **2.95986**).

It is a thin layer on top of ``../ensemble`` (the honest 80/20 pipeline):
  * it REUSES the architecture files in ``ensemble/models`` and the cached 80% val
    predictions in ``ensemble/val_preds`` (to fit the 2-group ball|player weights);
  * but every ensemble member is RETRAINED on 100% of the labeled data (train + val),
    producing ``model_<label>_full.pth`` checkpoints kept here in ``final_run/weights``.

Set ``NML_DATA`` to the dataset root (see ``../ensemble/_config.py``).
"""
import os
import sys
from pathlib import Path

FINAL = Path(__file__).resolve().parent
ROOT = FINAL.parent
ENSEMBLE = ROOT / "ensemble"

# --- reused from the 80/20 ensemble run ---
MODELS = ENSEMBLE / "models"          # architecture .py files (shared)
VAL_DIR_80 = ENSEMBLE / "val_preds"   # 80% val preds -> fit the 2-group weights
TEST_DIR_80 = ENSEMBLE / "test_preds"  # 80% test preds -> fallback for non-retrained members
sys.path.insert(0, str(MODELS))       # so `import nba_v46` resolves
sys.path.insert(0, str(FINAL))        # so `from _config import ...` resolves

# --- owned by the 100%-data run ---
WEIGHTS = FINAL / "weights"           # _full (100%-data) checkpoints
TEST_DIR = FINAL / "test_preds"       # _full test preds (gitignored, regenerable)
SOL_DIR = FINAL / "solutions"
for _d in (WEIGHTS, TEST_DIR, SOL_DIR):
    _d.mkdir(exist_ok=True)

TRAIN_TXT_FULL = ROOT / "train_files_full.txt"  # 4968 = 3974 train + 994 val

_DEFAULT_DATA = r"C:\Users\jan\.cache\kagglehub\competitions\nml-2026"
DATA = Path(os.environ.get("NML_DATA", _DEFAULT_DATA))
TRAIN = str(DATA / "train" / "train")
TEST = str(DATA / "test" / "test")
