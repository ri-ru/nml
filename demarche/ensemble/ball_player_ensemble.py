"""Build the 2-group ensemble: fit ONE constrained weight vector for the BALL (entity 0) and ONE
shared vector for ALL players (entities 1-10), by least-squares on the honest 80% validation
predictions. These two vectors ARE the ensemble — paste them into final/ensemble_weights.json
(the final submission applies them to the 100%-data test predictions).

Regularized middle ground between a global blend (1 vector, underfits the ball) and full per-entity
(11 vectors, overfits the players): keeps the ball-specialist routing while cutting params 11->2,
which shrinks the local->Kaggle overfit gap.

Reads the cached val predictions written by save_val_tta_preds.py (val_preds/<label>.pt + Y_real.pt).
"""
import numpy as np
import torch
from scipy.optimize import minimize

from _config import VAL_DIR as VAL, MEMBERS
HOR, NE = 12, 11
POOL = list(MEMBERS)   # the 6 members, from the single registry in _config

Y = torch.load(VAL / 'Y_real.pt').numpy()
labs = [l for l in POOL if (VAL / f'{l}.pt').exists()]
Pv = np.stack([torch.load(VAL / f'{l}.pt').numpy() for l in labs])   # [M, N, 12, 11, 2]
M = len(labs)
print(f'{M} models: {", ".join(labs)}')


def fit(Pe, Ye):
    """Constrained least squares: weights >= 0, sum to 1 (a convex blend)."""
    res = minimize(lambda w: float(np.mean((w @ Pe - Ye) ** 2)), np.ones(M) / M, method='SLSQP',
                   bounds=[(0, 1)] * M, constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1},),
                   options={'maxiter': 500, 'ftol': 1e-10})
    return res.x


# ball group = entity 0 ; player group = entities 1..10 (pooled)
wb = fit(Pv[:, :, :, 0, :].reshape(M, -1), Y[:, :, 0, :].reshape(-1))
wp = fit(Pv[:, :, :, 1:, :].reshape(M, -1), Y[:, :, 1:, :].reshape(-1))

# val MSE under the fitted 2-group blend
out = np.zeros_like(Y)
out[:, :, 0, :] = (wb @ Pv[:, :, :, 0, :].reshape(M, -1)).reshape(Y.shape[0], HOR, 2)
for e in range(1, NE):
    out[:, :, e, :] = (wp @ Pv[:, :, :, e, :].reshape(M, -1)).reshape(Y.shape[0], HOR, 2)
overall = float(np.mean((out - Y) ** 2))
ball = float(np.mean((out[:, :, 0, :] - Y[:, :, 0, :]) ** 2))
print(f'2-group val MSE: overall={overall:.4f}  ball={ball:.3f}  '
      f'-> proj Kaggle ~{overall * 1.079:.3f}-{overall * 1.083:.3f}')

# the fitted weights — paste these into final/ensemble_weights.json (pool order = the printed order)
print(f"\n{'model':<14}{'BALL w':>9}{'PLAYER w':>10}")
print('-' * 33)
for i in range(M):
    print(f"{labs[i]:<14}{wb[i]:>9.3f}{wp[i]:>10.3f}")
print('-' * 33)
print(f'\n  "pool":           {labs}')
print(f'  "ball_weights":   {[round(float(x), 3) for x in wb]}')
print(f'  "player_weights": {[round(float(x), 3) for x in wp]}')
