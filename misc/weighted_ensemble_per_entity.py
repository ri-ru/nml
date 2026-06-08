"""
Per-entity weighted ensemble.

Fits a separate weight vector (one per ensemble model) for each of the 11
entities (10 players + 1 ball), under the constraint w_i,m >= 0 and
sum_m w_i,m = 1 for every entity i.

Run after save_val_tta_preds.py + save_test_tta_preds.py have been run.
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
    'v60', 'v60_s1', 'v60_swa', 'v60_s1_swa',
]


def main():
    Y = torch.load(VAL_DIR / 'Y_real.pt')  # [N, 12, 11, 2]
    N = Y.shape[0]
    print(f'Y_real: {tuple(Y.shape)}')

    avail = []
    P_val_list = []
    P_test_list = []
    for lab in LABELS:
        vp = VAL_DIR / f'{lab}.pt'
        tp = TEST_DIR / f'{lab}.pt'
        if not vp.exists() or not tp.exists():
            continue
        avail.append(lab)
        P_val_list.append(torch.load(vp))
        P_test_list.append(torch.load(tp))
    M = len(avail)
    print(f'Loaded {M} models: {avail}')

    P_val = torch.stack(P_val_list, dim=0).numpy()    # [M, N, 12, 11, 2]
    P_test = torch.stack(P_test_list, dim=0)          # [M, N_test, 12, 11, 2]
    Y_np = Y.numpy()

    # --- Baseline: same constrained weights for all entities ---
    P_val_flat = P_val.reshape(M, -1)
    Y_flat = Y_np.reshape(-1)

    def loss_global(w):
        pred = w @ P_val_flat
        return float(np.mean((pred - Y_flat) ** 2))

    w0 = np.ones(M) / M
    bounds = [(0.0, 1.0)] * M
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1.0},)
    res = minimize(loss_global, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 500, 'ftol': 1e-10})
    w_global = res.x
    global_pred = w_global @ P_val_flat
    global_score = float(np.mean((global_pred - Y_flat) ** 2))
    print(f'\nBaseline constrained global weights val MSE: {global_score:.4f}')

    # --- Per-entity constrained weights ---
    print('\nFitting per-entity weights...')
    W_per_entity = np.zeros((N_ENTITIES, M))
    per_entity_scores = []
    for e in range(N_ENTITIES):
        Pe = P_val[:, :, :, e, :].reshape(M, -1)  # [M, N*12*2]
        Ye = Y_np[:, :, e, :].reshape(-1)

        def loss_e(w, Pe=Pe, Ye=Ye):
            pred = w @ Pe
            return float(np.mean((pred - Ye) ** 2))

        res = minimize(loss_e, w0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 500, 'ftol': 1e-10})
        W_per_entity[e] = res.x
        per_entity_scores.append(float(np.mean(((res.x @ Pe) - Ye) ** 2)))
        # Also baseline (global) MSE on this entity for comparison
        baseline_e = float(np.mean(((w_global @ Pe) - Ye) ** 2))
        # Top 3 weights for this entity
        top = sorted(enumerate(res.x), key=lambda t: -t[1])[:3]
        top_str = ', '.join(f'{avail[i]}={w:.3f}' for i, w in top)
        print(f'  entity {e:2d}  per-entity={per_entity_scores[-1]:.4f}  baseline={baseline_e:.4f}  top: {top_str}')

    # Overall per-entity val MSE (= mean across all positions, equivalent to overall MSE if entities are equally weighted)
    pred_val = np.zeros_like(Y_np)
    for e in range(N_ENTITIES):
        pred_val[:, :, e, :] = (W_per_entity[e] @ P_val[:, :, :, e, :].reshape(M, -1)).reshape(N, HORIZON, 2)
    pe_score = float(np.mean((pred_val - Y_np) ** 2))
    delta = pe_score - global_score
    print(f'\nPer-entity ensemble val MSE: {pe_score:.4f}  (vs baseline {global_score:.4f}, delta {delta:+.4f})')

    # --- Apply per-entity weights to test ---
    P_test_np = P_test.numpy()  # [M, N_test, 12, 11, 2]
    N_test = P_test_np.shape[1]
    test_pred = np.zeros((N_test, HORIZON, N_ENTITIES, 2), dtype=P_test_np.dtype)
    for e in range(N_ENTITIES):
        flat = P_test_np[:, :, :, e, :].reshape(M, -1)  # [M, N_test*12*2]
        test_pred[:, :, e, :] = (W_per_entity[e] @ flat).reshape(N_test, HORIZON, 2)

    cols = ['id'] + [f'entity_{i}_time_{t}_{axis}'
                     for t in range(HORIZON) for i in range(N_ENTITIES) for axis in ['x', 'y']]
    rows = [[idx] + test_pred[idx].reshape(-1).tolist() for idx in range(N_test)]
    out = SOL_DIR / f'solution_weighted_perentity_{M}m.csv'
    pd.DataFrame(rows, columns=cols).set_index('id').sort_index().to_csv(out)
    print(f'\nWrote {out.name}')


if __name__ == '__main__':
    main()
