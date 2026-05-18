"""
Per-(entity, timestep) weighted ensemble with optional bias correction.

For each (entity i, timestep t) cell, fits its own constrained weight vector
over the 21 ensemble models (w_i,t,m >= 0, sum_m w_i,t,m = 1). Then optionally
subtracts the mean val residual per (entity, timestep, axis) from the test
predictions to correct any systematic bias.

Run after save_val_tta_preds.py + save_test_tta_preds.py.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

BASE = Path(__file__).parent.parent
VAL_DIR = BASE / 'models' / 'val_preds'
TEST_DIR = BASE / 'models' / 'test_preds'
SOL_DIR = BASE / 'models' / 'solutions'
HORIZON = 12
N_ENTITIES = 11

LABELS = [
    'v46', 'v58', 'v58_swa',
    'v46_s1', 'v46_s2', 'v46_s3', 'v46_s4',
    'v46_s5', 'v46_s6', 'v46_s7', 'v46_s8',
    'v46_s1_swa', 'v46_s2_swa', 'v46_s3_swa', 'v46_s4_swa',
    'v46_s5_swa', 'v46_s6_swa', 'v46_s7_swa', 'v46_s8_swa',
    'v58_s1', 'v58_s2', 'v58_s3',
    'v58_s1_swa', 'v58_s2_swa', 'v58_s3_swa',
]


def main():
    Y = torch.load(VAL_DIR / 'Y_real.pt').numpy()  # [N, 12, 11, 2]
    N = Y.shape[0]
    print(f'Y_real: {Y.shape}')

    avail, P_val_list, P_test_list = [], [], []
    for lab in LABELS:
        vp = VAL_DIR / f'{lab}.pt'
        tp = TEST_DIR / f'{lab}.pt'
        if not vp.exists() or not tp.exists():
            continue
        avail.append(lab)
        P_val_list.append(torch.load(vp).numpy())
        P_test_list.append(torch.load(tp).numpy())
    M = len(avail)
    print(f'Loaded {M} models')

    P_val = np.stack(P_val_list, axis=0)    # [M, N, 12, 11, 2]
    P_test = np.stack(P_test_list, axis=0)  # [M, N_test, 12, 11, 2]
    N_test = P_test.shape[1]

    # --- Baseline: per-entity constrained weights ---
    print('\nFitting per-entity weights (baseline reference)...')
    W_pe = np.zeros((N_ENTITIES, M))
    w0 = np.ones(M) / M
    bounds = [(0.0, 1.0)] * M
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1.0},)
    for e in range(N_ENTITIES):
        Pe = P_val[:, :, :, e, :].reshape(M, -1)
        Ye = Y[:, :, e, :].reshape(-1)
        res = minimize(lambda w, P=Pe, Y=Ye: float(np.mean((w @ P - Y) ** 2)),
                       w0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 500, 'ftol': 1e-10})
        W_pe[e] = res.x
    pred_pe = np.zeros_like(Y)
    for e in range(N_ENTITIES):
        pred_pe[:, :, e, :] = (W_pe[e] @ P_val[:, :, :, e, :].reshape(M, -1)).reshape(N, HORIZON, 2)
    pe_score = float(np.mean((pred_pe - Y) ** 2))
    print(f'  per-entity val MSE: {pe_score:.4f}')

    # --- Per-(entity, timestep) constrained weights ---
    print('\nFitting per-(entity, timestep) weights...')
    W_pet = np.zeros((N_ENTITIES, HORIZON, M))
    for e in range(N_ENTITIES):
        for t in range(HORIZON):
            Pet = P_val[:, :, t, e, :].reshape(M, -1)  # [M, N*2]
            Yet = Y[:, t, e, :].reshape(-1)
            res = minimize(lambda w, P=Pet, Y=Yet: float(np.mean((w @ P - Y) ** 2)),
                           w0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 500, 'ftol': 1e-10})
            W_pet[e, t] = res.x
    pred_pet = np.zeros_like(Y)
    for e in range(N_ENTITIES):
        for t in range(HORIZON):
            Pet = P_val[:, :, t, e, :].reshape(M, -1)
            pred_pet[:, t, e, :] = (W_pet[e, t] @ Pet).reshape(N, 2)
    pet_score = float(np.mean((pred_pet - Y) ** 2))
    print(f'  per-(entity,timestep) val MSE: {pet_score:.4f}  (delta vs per-entity {pet_score - pe_score:+.4f})')

    # --- Residual bias correction on top of per-(entity, timestep) ---
    # Mean residual per (entity, timestep, axis): shape [12, 11, 2]
    bias = (pred_pet - Y).mean(axis=0)  # [12, 11, 2]
    pred_pet_bc = pred_pet - bias[None, :, :, :]
    pet_bc_score = float(np.mean((pred_pet_bc - Y) ** 2))
    print(f'  per-(entity,timestep) + bias correction val MSE: {pet_bc_score:.4f}  (delta vs PET {pet_bc_score - pet_score:+.4f})')
    print(f'  bias magnitude max={np.abs(bias).max():.4f}  mean={np.abs(bias).mean():.4f}  (feet)')

    # Same idea but on top of per-entity, in case per-entity-timestep overfits relative to it
    bias_pe = (pred_pe - Y).mean(axis=0)
    pred_pe_bc = pred_pe - bias_pe[None, :, :, :]
    pe_bc_score = float(np.mean((pred_pe_bc - Y) ** 2))
    print(f'  per-entity + bias correction val MSE: {pe_bc_score:.4f}  (delta vs per-entity {pe_bc_score - pe_score:+.4f})')

    print('\n=== Summary (local val MSE) ===')
    print(f'  per-entity                            {pe_score:.4f}')
    print(f'  per-entity + bias correction          {pe_bc_score:.4f}')
    print(f'  per-(entity,timestep)                 {pet_score:.4f}')
    print(f'  per-(entity,timestep) + bias correction {pet_bc_score:.4f}')

    cols = ['id'] + [f'entity_{i}_time_{t}_{axis}'
                     for t in range(HORIZON) for i in range(N_ENTITIES) for axis in ['x', 'y']]

    # Apply per-(entity, timestep) weights to test predictions.
    test_pred_pet = np.zeros((N_test, HORIZON, N_ENTITIES, 2), dtype=P_test.dtype)
    for e in range(N_ENTITIES):
        for t in range(HORIZON):
            Pet = P_test[:, :, t, e, :].reshape(M, -1)  # [M, N_test*2]
            test_pred_pet[:, t, e, :] = (W_pet[e, t] @ Pet).reshape(N_test, 2)

    # Also apply per-entity weights to test, with and without bias correction (safer fallbacks).
    test_pred_pe = np.zeros((N_test, HORIZON, N_ENTITIES, 2), dtype=P_test.dtype)
    for e in range(N_ENTITIES):
        Pe_test = P_test[:, :, :, e, :].reshape(M, -1)
        test_pred_pe[:, :, e, :] = (W_pe[e] @ Pe_test).reshape(N_test, HORIZON, 2)

    candidates = [
        ('perentity_biascor',      test_pred_pe  - bias_pe[None, :, :, :],     pe_bc_score),
        ('perentity_time',         test_pred_pet,                              pet_score),
        ('perentity_time_biascor', test_pred_pet - bias[None, :, :, :],        pet_bc_score),
    ]
    for tag, pred, score in candidates:
        rows = [[idx] + pred[idx].reshape(-1).tolist() for idx in range(N_test)]
        out = SOL_DIR / f'solution_weighted_{tag}_{M}m.csv'
        pd.DataFrame(rows, columns=cols).set_index('id').sort_index().to_csv(out)
        print(f'  wrote {out.name}  (local val MSE {score:.4f})')


if __name__ == '__main__':
    main()
