"""Local val score (overall + ball) for every cached single-model TTA pred + recap of ensemble pools."""
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).parent.parent
VAL = BASE / 'models' / 'val_preds'
Y = torch.load(VAL / 'Y_real.pt').numpy()  # [N,12,11,2]

rows = []
for f in sorted(VAL.glob('*.pt')):
    if f.name == 'Y_real.pt':
        continue
    P = torch.load(f).numpy()
    if P.shape != Y.shape:
        continue
    overall = float(((P - Y) ** 2).mean())
    ball = float(((P[:, :, 0, :] - Y[:, :, 0, :]) ** 2).mean())
    rows.append((f.stem, overall, ball))

rows.sort(key=lambda r: r[1])
print(f'{"model":<16}{"overall":>9}{"ball":>9}')
print('-' * 34)
for name, ov, ball in rows:
    print(f'{name:<16}{ov:>9.4f}{ball:>9.3f}')
