"""Milestone 3 — the canonical model: mixture-of-2-Gaussians head + timestep-weighted NLL,
trained 300 epochs (seed 0). Local val ADE 2.86.

The architecture itself lives in `model/architecture.py` (single source of truth); this file
is the milestone entry point so `core/run_one.py` can train it:

    python core/run_one.py demarche/nba_v46.py 300 32
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `model`
from model.architecture import NBAv46Model, Trainer  # noqa: F401,E402  (run_one discovers Trainer)
