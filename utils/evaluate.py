import os
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader

RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'results.md')


def evaluate_model(model, trainer, val_dl, name: str, notes: str = '') -> float:
    """Evaluate a model, print the score, and log it to results.md. for tracking"""
    loss_fn = nn.MSELoss()
    mu = trainer.mu.to(trainer.device)
    sigma = trainer.sigma.to(trainer.device)
    total = 0.0

    model.train(False)
    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.to(trainer.device), y.to(trainer.device)
            pred = model(X)
            B, T, N, F = y.shape
            y_orig = y[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
            pred_orig = pred * sigma + mu
            y_orig    = y_orig * sigma + mu
            total += loss_fn(pred_orig, y_orig).item()

    score = total / len(val_dl)
    print(f'{name}: {score:.4f}')
    _log(name, score, notes)
    return score


def _log(name: str, score: float, notes: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    row = f'| {timestamp} | {name} | {score:.4f} | {notes} |\n'

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w') as f:
            f.write('# Model Results\n\n')
            f.write('| Time | Model | Local Score | Notes |\n')
            f.write('|------|-------|-------------|-------|\n')

    with open(RESULTS_FILE, 'a') as f:
        f.write(row)


class _Sampler:
    def __init__(self, max_start, batch_size):
        self.max_start = max_start

    def __iter__(self):
        for i, ms in enumerate(self.max_start):
            for start in (0, ms // 2, ms):
                yield i, start

    def __len__(self):
        return len(self.max_start) * 3


def make_val_dl(trainer, batch_size: int = 64):
    sampler = _Sampler(trainer.val_dataset.max_start, batch_size)
    return DataLoader(trainer.val_dataset, batch_size=batch_size, sampler=sampler)
