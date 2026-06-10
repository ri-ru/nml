"""v60_w8: v46 arch + ball-weighted MoG NLL, ball entity weighted 8x. See nba_v60_base.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nba_v60_base import build_trainer_class

BALL_W = 8.0
Trainer = build_trainer_class(BALL_W, 'model_v60_w8')
