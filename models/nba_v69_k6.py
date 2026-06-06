"""v69_k6: split head, ball K_ball=6, players K=2, BALL_W=1.0 (clean A/B). See nba_v69.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nba_v69 import build_trainer_class

K_BALL = 6
BALL_W = 1.0
Trainer = build_trainer_class(k_ball=K_BALL, ball_w=BALL_W, save_stem='model_v69_k6')
