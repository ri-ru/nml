"""Build 7- and 6-model candidates using the RETRAINED v46 _full test preds.

Identical method to build_trimmed_candidate.py / build_trimmed6_candidate.py (2-group weights fit on
the honest 80% val preds, applied to the _full 100%-data test preds), but the v46 _full test preds
now come from the fresh 300-epoch retrain. Writes NEW filenames so the submitted 2.95815 CSVs
(solution_trimmed7_2grp.csv, ../final_7model/solution_final_7model.csv) are never overwritten:

    solution_trimmed7_v46retrain.csv   (7-model pool, with retrained v46)
    solution_trimmed6_v46retrain.csv   (6-model pool, v58 dropped, with retrained v46)

NOTE: the 80% val preds (and thus the fitted weights + local reference MSE) are UNCHANGED -- they come
from the 80%-trained v46. Only the test predictions differ, so the local number cannot see this change;
submit and compare on Kaggle against 2.95815.
"""
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize

from _config import VAL_DIR_80 as VAL, TEST_DIR as TESTF, TEST_DIR_80 as TEST80, SOL_DIR as SOL
HOR, NE = 12, 11

POOL7 = ['v60_w8', 'v60_w5_s1', 'v60_w8_s2', 'v46', 'v58', 'v46_s3', 'v46_s7']
POOL6 = ['v60_w8', 'v60_w5_s1', 'v60_w8_s2', 'v46', 'v46_s3', 'v46_s7']   # v58 dropped

Y = torch.load(VAL / 'Y_real.pt').numpy()


def fit(Pe, Ye, M):
    r = minimize(lambda w: float(np.mean((w @ Pe - Ye) ** 2)), np.ones(M) / M, method='SLSQP',
                 bounds=[(0, 1)] * M, constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1},),
                 options={'maxiter': 500, 'ftol': 1e-10})
    return r.x


def build(pool, out_name):
    labs = [l for l in pool if (VAL / f'{l}.pt').exists()]
    missing = [l for l in pool if l not in labs]
    if missing:
        raise SystemExit(f'missing 80% val preds for {missing}')
    M = len(labs)
    Pv = np.stack([torch.load(VAL / f'{l}.pt').numpy() for l in labs])
    wb = fit(Pv[:, :, :, 0, :].reshape(M, -1), Y[:, :, 0, :].reshape(-1), M)
    wp = fit(Pv[:, :, :, 1:, :].reshape(M, -1), Y[:, :, 1:, :].reshape(-1), M)

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
    csv = SOL / out_name
    pd.DataFrame(rows, columns=['id'] + cols).set_index('id').sort_index().to_csv(csv)
    print(f'\n{out_name}: {M} models {labs}; {nfull}/{M} use _full test preds (v46 = RETRAIN).')
    print('  ball  : ' + ', '.join(f'{labs[i]}={wb[i]:.3f}' for i in np.argsort(-wb) if wb[i] > 0.005))
    print('  player: ' + ', '.join(f'{labs[i]}={wp[i]:.3f}' for i in np.argsort(-wp) if wp[i] > 0.005))


build(POOL7, 'solution_trimmed7_v46retrain.csv')
build(POOL6, 'solution_trimmed6_v46retrain.csv')
print('\nDONE. Submit & compare to 2.95815 (7-model committed v46).')
