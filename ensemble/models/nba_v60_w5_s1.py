"""Ensemble member v60_w5_s1 — ball specialist, ball loss x5, seed 1.
    python core/run_one.py ensemble/models/nba_v60_w5_s1.py 300 32   -> model_v60_w5_s1_300ep.pth
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `model`
from model.variants import build_member  # noqa: E402

Trainer = build_member(seed=1, ball_w=5.0)
