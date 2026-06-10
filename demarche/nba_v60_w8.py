"""Milestone 4 — ball-weighting: the v46 architecture trained with the ball's loss up-weighted
8x. Error analysis showed the ball is ~6x harder than a player, so we train ball specialists for
the ensemble. Same network as v46; only the loss changes (see model/variants.py).

    python core/run_one.py demarche/nba_v60_w8.py 300 32
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `model`
from model.variants import build_member  # noqa: E402

Trainer = build_member(seed=0, ball_w=8.0)  # run_one discovers `Trainer`
