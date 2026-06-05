"""v63: v46 arch + loss, but training data gets random rotation augmentation (+/-30 deg) on top of
the x/y flips. Goal: push the model toward rotation-equivariance (findings: rotation TTA fails
because v46 learned court-aligned features). See nba_v6_common.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nba_v6_common import make_trainer
from nba_v46 import NBAv46Model

Trainer = make_trainer(NBAv46Model, 'model_v63', rotate=True)
