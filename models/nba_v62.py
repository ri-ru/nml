"""v62: v46 arch, but the MoG head predicts a residual over a CONSTANT-VELOCITY extrapolation
(pos + vel + delta). The easy inertial part is free; capacity focuses on deviations.
See nba_v6_common.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nba_v6_common import make_trainer, NBAv62Model

Trainer = make_trainer(NBAv62Model, 'model_v62')
