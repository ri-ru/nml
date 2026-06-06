"""v60_w5_s1: ball-weighted (5x, best-overall) seed variant for ensemble diversity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nba_v60_base import build_trainer_class

Trainer = build_trainer_class(5.0, 'model_v60_w5_s1', seed=1)
