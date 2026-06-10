"""Builds the 2-group ensemble.
This will fit one constrained weight vector for the ball (entity 0) 
and one shared vector for all players (1-10 entities)

w/ least squares on the 80% validation predictions.

important: 80% val predictions arent commited and must be regenerated. 
(the final submission applies them to the 100%-data test predictions).

Reads the cached val predictions written by save_val_tta_preds.py (val_preds/<label>.pt + Y_real.pt).
"""
import numpy as np
import torch
from scipy.optimize import minimize

from _config import VAL_DIR as VAL, MEMBERS
HOR, NE = 12, 11
POOL = list(MEMBERS)  #from config 

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


# ball group = entity 0 ; player group = entities 1..10  
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

# fitted weights , for the .json in outputs
print(f"\n{'model':<14}{'BALL w':>9}{'PLAYER w':>10}")
print('-' * 33)
for i in range(M):
    print(f"{labs[i]:<14}{wb[i]:>9.3f}{wp[i]:>10.3f}")
print('-' * 33)
print(f'\n  "pool":           {labs}')
print(f'  "ball_weights":   {[round(float(x), 3) for x in wb]}')
print(f'  "player_weights": {[round(float(x), 3) for x in wp]}')
