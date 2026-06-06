"""v69_k6_w4: split head ball K_ball=6 + ball-weighting BALL_W=4.0 (combo). See nba_v69.py.

Tests whether more ball modes COMBINES with the validated ball-upweighting lever (v60).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nba_v69 import build_trainer_class

K_BALL = 6
BALL_W = 4.0
Trainer = build_trainer_class(k_ball=K_BALL, ball_w=BALL_W, save_stem='model_v69_k6_w4')
