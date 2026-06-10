"""Train one ensemble member with it's name 
    python demarche/ensemble/train_member.py v46_s3 300            # 80% split  -> model_v46_s3_300ep.pth
    python demarche/ensemble/train_member.py v46_s3 300 --full     # 100% data  -> model_v46_s3_full.pth

see _config.MEMBERS for seeds 

  - default : trains on the fixed 80/20 split (needs train_files.txt/val_files.txt) and saves
              model_<label>_<epochs>ep.pth  used to FIT the ensemble weights.
  - --full  : trains on 100% of the labelled data and saves model_<label>_full.pth  the final submission 
  actually loads (final/build_ensemble_6model.py).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  #  
from nba_v60_base import build_trainer_class  # noqa: E402
from _config import MEMBERS, WEIGHTS, TRAIN  # noqa: E402


def main():
    full = '--full' in sys.argv
    argv = [a for a in sys.argv if a != '--full']
    if len(argv) < 2 or argv[1] not in MEMBERS:
        raise SystemExit("usage: python demarche/ensemble/train_member.py <label> [epochs] [batch] [--full]\n"
                         f"labels: {', '.join(MEMBERS)}\n"
                         "  default: trains on the 80% split (needs train_files.txt/val_files.txt) -> _300ep.pth\n"
                         "  --full : trains on 100% of the data (the submission checkpoint)       -> _full.pth")
    label = argv[1]
    epochs = int(argv[2]) if len(argv) > 2 else 300
    batch = int(argv[3]) if len(argv) > 3 else 32
    ball_w, seed = MEMBERS[label]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    stem = f'model_{label}_full' if full else f'model_{label}_{epochs}ep'
    Trainer = build_trainer_class(ball_w, stem, seed=seed, weights_dir=WEIGHTS)
    trainer = Trainer(train_path=TRAIN, val_path=TRAIN, device=device,
                      epochs=epochs, batch_size=batch, full=full)
    trainer.train()
    print(f'saved {WEIGHTS}/{stem}.pth')


if __name__ == '__main__':
    main()
