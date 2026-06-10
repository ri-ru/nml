"""Ensemble member v60_w8 — ball specialist, ball loss x8, seed 0.
Thin wrapper: the architecture + trainer live once in demarche/. Retrain with:
    python core/run_one.py ensemble/models/nba_v60_w8.py 300 32   -> ensemble/weights/model_v60_w8.pth
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'demarche'))  # the single architecture
from nba_v60_base import build_trainer_class  # noqa: E402

_WEIGHTS = Path(__file__).resolve().parents[1] / 'weights'
Trainer = build_trainer_class(8.0, 'model_v60_w8', seed=0, weights_dir=_WEIGHTS)
