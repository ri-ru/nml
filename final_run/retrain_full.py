"""Retrain a model on 100% data (train + val) for the FINAL submission. Oscar's '20% more data'.

Patches the training file list to train_files_full.txt (4968 = 3974 train + 994 val) before building
the Trainer, trains, and saves model_<tag>_full.pth. Normalization is recomputed from the full set
(no leakage concern -- it's all training data now). The val eval printed during training is
OPTIMISTIC (val is now in train) and should be IGNORED; these models are judged only on Kaggle.

Usage: python final_run/retrain_full.py <model_py_path> <tag> [epochs]
e.g.   python final_run/retrain_full.py ensemble/models/nba_v60_w8.py v60_w8 300
"""
import importlib.util
import sys
from pathlib import Path

import torch

from _config import WEIGHTS, ROOT, TRAIN, TRAIN_TXT_FULL as FULL

import nba_v46
nba_v46.TRAIN_TXT = FULL          # v46/v60/v72/v73 Trainers read this at __init__

model_py = sys.argv[1]
tag = sys.argv[2]
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 300

spec = importlib.util.spec_from_file_location('exp_full', ROOT / model_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if hasattr(mod, 'TRAIN_TXT'):     # models with their own TRAIN_TXT (e.g. v58 family)
    mod.TRAIN_TXT = FULL

tname = [n for n in dir(mod) if 'Trainer' in n and not n.startswith('_')][0]
TrainerClass = getattr(mod, tname)
trainer = TrainerClass(train_path=TRAIN, val_path=TRAIN, device='cuda', epochs=epochs, batch_size=32)
print(f'retrain_full: {model_py} -> model_{tag}_full.pth  (train files = {len(trainer.train_dataset.sequences)})')
trainer.train()
out = WEIGHTS / f'model_{tag}_full.pth'
torch.save(trainer.model.state_dict(), out)
print(f'SAVED {out.name}')
