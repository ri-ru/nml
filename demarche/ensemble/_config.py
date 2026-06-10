"""Canonical paths for the ensemble pipeline.

Run scripts from inside ``demarche/ensemble/`` (or as ``python demarche/ensemble/<script>.py``).

Point this at the dataset once, e.g. on the machine that has the data::

    export NML_DATA=/path/to/nml-2026          # macOS / Linux
    $env:NML_DATA = "C:\\path\\to\\nml-2026"   # PowerShell

``NML_DATA`` must be the folder that contains ``train/train/*.pt`` and
``test/test/*.pt`` (the kagglehub competition layout).
"""
import os
from pathlib import Path

ENSEMBLE = Path(__file__).resolve().parent     # demarche/ensemble
ROOT = ENSEMBLE.parent.parent                  # repo root

ARCH_DIR = ROOT / "demarche"            # single architecture source: demarche/nba_v46.py (no copies)
WEIGHTS = ROOT / "outputs" / "weights"  # all *.pth checkpoints live under outputs/

# The 6 ensemble members ARE the one demarche/nba_v46.py architecture, trained differently.
# label -> (ball_w, seed). ball_w == 1.0 is a plain generalist; > 1.0 is a ball specialist.
# This registry replaces per-member files: train any one with `python demarche/ensemble/train_member.py <label>`.
MEMBERS = {
    "v60_w8":    (8.0, 0),   # ball specialists
    "v60_w5_s1": (5.0, 1),
    "v60_w8_s2": (8.0, 2),
    "v46_s3":    (1.0, 3),   # player generalists
    "v46":       (1.0, 0),
    "v46_s7":    (1.0, 7),
}
VAL_DIR = ENSEMBLE / "val_preds"  # cached per-model val TTA tensors (gitignored)
TEST_DIR = ENSEMBLE / "test_preds"  # cached per-model test TTA tensors (gitignored)
SOL_DIR = ROOT / "outputs" / "solutions"  # submission CSVs live under outputs/
for _d in (VAL_DIR, TEST_DIR, SOL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# train/val split files live at the repo root
TRAIN_TXT = ROOT / "train_files.txt"
VAL_TXT = ROOT / "val_files.txt"

# Dataset root: NML_DATA env var, else the conventional ./data/nml-2026 in the repo (see docstring).
DATA = Path(os.environ.get("NML_DATA") or ROOT / "data" / "nml-2026")
if not (DATA / "train" / "train").exists():
    raise SystemExit(
        f"Dataset not found at {DATA}. Set NML_DATA to the dataset root (folder with "
        "train/train/*.pt and test/test/*.pt) or place it at ./data/nml-2026."
    )
TRAIN = str(DATA / "train" / "train")
TEST = str(DATA / "test" / "test")
