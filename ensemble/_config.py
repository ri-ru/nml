"""Canonical paths for the ensemble pipeline.

Run scripts from inside ``ensemble/`` (or as ``python ensemble/<script>.py``).

Point this at the dataset once, e.g. on the machine that has the data::

    export NML_DATA=/path/to/nml-2026          # macOS / Linux
    $env:NML_DATA = "C:\\path\\to\\nml-2026"   # PowerShell

``NML_DATA`` must be the folder that contains ``train/train/*.pt`` and
``test/test/*.pt`` (the kagglehub competition layout).
"""
import os
from pathlib import Path

ENSEMBLE = Path(__file__).resolve().parent
ROOT = ENSEMBLE.parent

ARCH_DIR = ROOT / "demarche"      # single architecture source: demarche/nba_v46.py (no copies)
WEIGHTS = ENSEMBLE / "weights"    # *.pth checkpoints
VAL_DIR = ENSEMBLE / "val_preds"  # cached per-model val TTA tensors (gitignored)
TEST_DIR = ENSEMBLE / "test_preds"  # cached per-model test TTA tensors (gitignored)
SOL_DIR = ENSEMBLE / "solutions"  # submission CSVs
for _d in (VAL_DIR, TEST_DIR, SOL_DIR):
    _d.mkdir(exist_ok=True)

# train/val split files live at the repo root
TRAIN_TXT = ROOT / "train_files.txt"
VAL_TXT = ROOT / "val_files.txt"

# Dataset root — must be set via the NML_DATA env var (see module docstring).
if "NML_DATA" not in os.environ:
    raise SystemExit(
        "Set NML_DATA to the dataset root (folder with train/train/*.pt and test/test/*.pt), "
        "e.g.  export NML_DATA=/path/to/nml-2026"
    )
DATA = Path(os.environ["NML_DATA"])
TRAIN = str(DATA / "train" / "train")
TEST = str(DATA / "test" / "test")
