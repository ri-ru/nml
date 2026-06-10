"""Ensemble member v46_s3 — player generalist, seed 3 (ensemble diversity).
Thin wrapper: the architecture + trainer live once in demarche/. Retrain with:
    python core/run_one.py ensemble/models/nba_v46_s3.py 300 32   -> ensemble/weights/model_v46_s3.pth
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'demarche'))  # the single architecture
from nba_v60_base import build_trainer_class  # noqa: E402

_WEIGHTS = Path(__file__).resolve().parents[1] / 'weights'
Trainer = build_trainer_class(1.0, 'model_v46_s3', seed=3, weights_dir=_WEIGHTS)  # ball_w=1.0 == plain MoG
