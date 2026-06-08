"""6-model variant of build_trimmed_candidate.py: drop v58 from the 7-model pool.

v58 is the only NBAv58Model (pseudo-labeled) member; the other 6 are all NBAv46Model. Dropping it
leaves a single-architecture ensemble. This is a pure refit + apply over cached predictions (no
retraining): fit the 2-group (ball|players) weights on the 80% val preds, apply to the _full
100%-data test preds. Writes solution_trimmed6_2grp.csv. Submit & compare to the 7-model 2.95815.
"""
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize

from _config import VAL_DIR_80 as VAL, TEST_DIR as TESTF, TEST_DIR_80 as TEST80, SOL_DIR as SOL
HOR, NE = 12, 11

# 7-model pool minus v58 -> 6 models, all nba_v46/NBAv46Model.
POOL = ['v60_w8', 'v60_w5_s1', 'v60_w8_s2',          # ball experts
        'v46', 'v46_s3', 'v46_s7']                   # player experts (v58 dropped)

Y = torch.load(VAL / 'Y_real.pt').numpy()
labs = [l for l in POOL if (VAL / f'{l}.pt').exists()]
missing = [l for l in POOL if l not in labs]
if missing:
    raise SystemExit(f'missing 80% val preds for {missing} -- run save_val_tta_preds.py for these first')
Pv = np.stack([torch.load(VAL / f'{l}.pt').numpy() for l in labs])   # [M,N,12,11,2]
M = len(labs)
print(f'{M} models: {labs}')

def fit(Pe, Ye):
    r = minimize(lambda w: float(np.mean((w @ Pe - Ye) ** 2)), np.ones(M) / M, method='SLSQP',
                 bounds=[(0, 1)] * M, constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1},),
                 options={'maxiter': 500, 'ftol': 1e-10})
    return r.x

wb = fit(Pv[:, :, :, 0, :].reshape(M, -1), Y[:, :, 0, :].reshape(-1))     # ball (entity 0)
wp = fit(Pv[:, :, :, 1:, :].reshape(M, -1), Y[:, :, 1:, :].reshape(-1))   # players (entities 1..10, pooled)

# test preds: prefer _full (100%-data), fall back to 80%
nfull = 0
Pt = []
for l in labs:
    fp = TESTF / f'{l}_full.pt'
    if fp.exists():
        Pt.append(torch.load(fp).numpy()); nfull += 1
    else:
        Pt.append(torch.load(TEST80 / f'{l}.pt').numpy())
Pt = np.stack(Pt)
Ntest = Pt.shape[1]
out = np.zeros((Ntest, HOR, NE, 2), dtype=np.float32)
out[:, :, 0, :] = (wb @ Pt[:, :, :, 0, :].reshape(M, -1)).reshape(Ntest, HOR, 2)
for e in range(1, NE):
    out[:, :, e, :] = (wp @ Pt[:, :, :, e, :].reshape(M, -1)).reshape(Ntest, HOR, 2)

cols = [f'entity_{i}_time_{t}_{a}' for t in range(HOR) for i in range(NE) for a in ['x', 'y']]
rows = [[i] + out[i].reshape(-1).tolist() for i in range(Ntest)]
csv = SOL / 'solution_trimmed6_2grp.csv'
pd.DataFrame(rows, columns=['id'] + cols).set_index('id').sort_index().to_csv(csv)

# local 2-group score on the 80% val preds (reference only; cannot see the _full gain)
vp = np.zeros_like(Y)
vp[:, :, 0, :] = (wb @ Pv[:, :, :, 0, :].reshape(M, -1)).reshape(Y.shape[0], HOR, 2)
for e in range(1, NE):
    vp[:, :, e, :] = (wp @ Pv[:, :, :, e, :].reshape(M, -1)).reshape(Y.shape[0], HOR, 2)
local = float(np.mean((vp - Y) ** 2))
print(f'\nTRIMMED-6 CANDIDATE (v58 dropped): {csv.name}')
print(f'  {nfull}/{M} members use _full (100%-data) test preds; rest 80%.')
print(f'  reference local 2-group (80% preds) = {local:.4f}  (7-model ref was 2.7568 -> Kaggle 2.95815)')
print(f'  ball   weights: ' + ', '.join(f'{labs[i]}={wb[i]:.3f}' for i in np.argsort(-wb) if wb[i] > 0.005))
print(f'  player weights: ' + ', '.join(f'{labs[i]}={wp[i]:.3f}' for i in np.argsort(-wp) if wp[i] > 0.005))
print('  NOTE: local cannot measure the _full gain (val is in the _full training set). Submit & compare on Kaggle.')
