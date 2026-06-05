"""v60_w3: v46 arch + ball-weighted MoG NLL, ball entity weighted 3x. See nba_v60_base.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nba_v60_base import build_trainer_class

BALL_W = 3.0
Trainer = build_trainer_class(BALL_W, 'model_v60_w3')
