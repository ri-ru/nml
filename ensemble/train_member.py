"""Train one ensemble member by name (replaces the old per-member files):

    python ensemble/train_member.py v46_s3 300

All members are the single demarche/nba_v46.py architecture; they differ only by
(ball_w, seed), defined once in _config.MEMBERS. The checkpoint is saved to
ensemble/weights/model_<label>_<epochs>ep.pth.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demarche'))  # the single architecture
from nba_v60_base import build_trainer_class  # noqa: E402
from _config import MEMBERS, WEIGHTS, TRAIN  # noqa: E402


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MEMBERS:
        raise SystemExit("usage: python ensemble/train_member.py <label> [epochs] [batch]\n"
                         f"labels: {', '.join(MEMBERS)}")
    label = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    ball_w, seed = MEMBERS[label]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Trainer = build_trainer_class(ball_w, f'model_{label}_{epochs}ep', seed=seed, weights_dir=WEIGHTS)
    trainer = Trainer(train_path=TRAIN, val_path=TRAIN, device=device, epochs=epochs, batch_size=batch)
    trainer.train()
    print(f'saved {WEIGHTS}/model_{label}_{epochs}ep.pth')


if __name__ == '__main__':
    main()
