"""v61: v46 arch, but the MoG head predicts DISPLACEMENT (pos + delta) instead of absolute position.
Rationale: displacement targets are lower-variance/better-conditioned and reduce rollout drift.
See nba_v6_common.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nba_v6_common import make_trainer, NBAv61Model

Trainer = make_trainer(NBAv61Model, 'model_v61')
