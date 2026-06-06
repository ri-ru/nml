"""v71 sweep variant: gamma=0.5 (softer tail concentration), base_ball_w=8.
See nba_v71.py for the full loss derivation. Same mean ball weight as v60_w8.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nba_v71  # noqa: F401  (imports trunk + loss)

Trainer = nba_v71.build_trainer_class(base_ball_w=8.0, gamma=0.5, save_stem='model_v71_g05')
