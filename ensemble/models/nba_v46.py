"""Ensemble member v46 — player generalist, seed 0. One architecture (model/), trained via config.
    python core/run_one.py ensemble/models/nba_v46.py 300 32   -> model_v46_300ep.pth
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `model`
from model.variants import build_member  # noqa: E402

Trainer = build_member(seed=0)
