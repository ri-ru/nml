"""
Fit per-model ensemble weights by minimizing val MSE, then apply those
weights to test predictions and write a submission CSV.

Requires:
  models/val_preds/Y_real.pt and models/val_preds/<label>.pt
  models/test_preds/<label>.pt   (per-model test TTA tensors)

Writes:
  models/solutions/solution_weighted_<tag>.csv

Run: python misc/weighted_ensemble.py
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

# Candidate pool — script will skip labels missing on disk.
LABELS = [
    'v46', 'v58', 'v58_swa',
    'v46_s1', 'v46_s2', 'v46_s3', 'v46_s4',
    'v46_s5', 'v46_s6', 'v46_s7', 'v46_s8',
    'v46_s1_swa', 'v46_s2_swa', 'v46_s3_swa', 'v46_s4_swa',
    'v46_s5_swa', 'v46_s6_swa', 'v46_s7_swa', 'v46_s8_swa',
    'v58_s1', 'v58_s2', 'v58_s3',
    'v58_s1_swa', 'v58_s2_swa', 'v58_s3_swa',
]


def mse_feet(pred, y):
    return float(((pred - y) ** 2).mean().item())


def main():
    Y_path = VAL_DIR / 'Y_real.pt'
    if not Y_path.exists():
        raise SystemExit(f'Missing {Y_path}. Run save_val_tta_preds.py first.')
    Y = torch.load(Y_path)  # [N, 12, 11, 2]
    N = Y.shape[0]
    Y_flat = Y.reshape(N, -1).numpy()  # [N, 264]
    print(f'Y_real: {tuple(Y.shape)}')

    # Load available val preds.
    avail = []
    P_val = []
    P_test = []
    test_ids = None
    for lab in LABELS:
        vp = VAL_DIR / f'{lab}.pt'
        tp = TEST_DIR / f'{lab}.pt'
        if not vp.exists():
            print(f'  skip {lab} (no val preds)')
            continue
        if not tp.exists():
            print(f'  skip {lab} (no test preds)')
            continue
        v = torch.load(vp)  # [N, 12, 11, 2]
        t = torch.load(tp)  # [N_test, 12, 11, 2]
        assert v.shape == Y.shape, f'{lab} val shape mismatch {v.shape} vs {Y.shape}'
        avail.append(lab)
        P_val.append(v.reshape(N, -1).numpy())
        P_test.append(t)
        print(f'  loaded {lab}  val={tuple(v.shape)}  test={tuple(t.shape)}')

    if len(avail) < 2:
        raise SystemExit('Need at least 2 models to weight-ensemble.')

    # Per-model solo scores (for reference).
    print('\nSolo val MSE (feet^2):')
    for lab, p in zip(avail, P_val):
        print(f'  {lab:14s} {np.mean((p - Y_flat) ** 2):.4f}')

    # Equal-weight baseline.
    eq = np.mean(np.stack(P_val), axis=0)
    eq_score = np.mean((eq - Y_flat) ** 2)
    print(f'\nEqual-weight ensemble val MSE: {eq_score:.4f}')

    # --- Fit free weights (least-squares closed form) ---
    # Stack: P[M, N, D] -> for each sample, sum_m w_m * P[m] = Y.
    # Equivalent to least-squares on P_mat (N*D, M) vs Y_vec (N*D).
    M = len(avail)
    D = P_val[0].shape[1]
    P_mat = np.stack(P_val, axis=-1).reshape(N * D, M)
    y_vec = Y_flat.reshape(N * D)
    w_free, *_ = np.linalg.lstsq(P_mat, y_vec, rcond=None)
    free_pred = (P_mat @ w_free).reshape(N, D)
    free_score = np.mean((free_pred - Y_flat) ** 2)
    print(f'\nFree-weight least-squares val MSE: {free_score:.4f}')
    print(f'  weights: {dict(zip(avail, [round(float(x), 3) for x in w_free]))}')
    print(f'  sum={float(w_free.sum()):.3f}')

    # --- Fit constrained weights on full pool: w_m >= 0, sum w_m = 1 ---
    P_stack = np.stack(P_val)  # [M, N, D]

    def loss_full(w):
        pred = np.tensordot(w, P_stack, axes=([0], [0]))
        return float(np.mean((pred - Y_flat) ** 2))

    w0 = np.ones(M) / M
    bounds = [(0.0, 1.0)] * M
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1.0},)
    res = minimize(loss_full, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 500, 'ftol': 1e-10})
    w_con = res.x
    con_pred = np.tensordot(w_con, P_stack, axes=([0], [0]))
    con_score = float(np.mean((con_pred - Y_flat) ** 2))
    print(f'\nConstrained-weight (full pool) val MSE: {con_score:.4f}')
    nz = {k: round(float(v), 3) for k, v in zip(avail, w_con) if v >= 0.01}
    print(f'  non-zero weights: {nz}')

    # --- Greedy forward selection (equal-weight) ---
    print('\nGreedy forward selection (equal-weight subset):')
    solo_scores = {lab: float(np.mean((p - Y_flat) ** 2)) for lab, p in zip(avail, P_val)}
    used = [min(avail, key=lambda l: solo_scores[l])]
    cur_score = solo_scores[used[0]]
    print(f'  start: {used[0]} solo = {cur_score:.4f}')
    while len(used) < len(avail):
        best_l, best_s = None, cur_score
        for lab in avail:
            if lab in used:
                continue
            sel_idx = [avail.index(l) for l in used + [lab]]
            avg = P_stack[sel_idx].mean(axis=0)
            s = float(np.mean((avg - Y_flat) ** 2))
            if s < best_s:
                best_s, best_l = s, lab
        if best_l is None:
            print('  no improvement; stop')
            break
        used.append(best_l)
        cur_score = best_s
        print(f'  +{best_l} -> {cur_score:.4f}')
    greedy_subset = used
    greedy_eq_score = cur_score

    # --- Re-weight the greedy subset (constrained) ---
    sub_idx = [avail.index(l) for l in greedy_subset]
    P_sub = P_stack[sub_idx]
    M_sub = len(sub_idx)

    def loss_sub(w):
        pred = np.tensordot(w, P_sub, axes=([0], [0]))
        return float(np.mean((pred - Y_flat) ** 2))

    w0_sub = np.ones(M_sub) / M_sub
    bounds_sub = [(0.0, 1.0)] * M_sub
    res_sub = minimize(loss_sub, w0_sub, method='SLSQP', bounds=bounds_sub,
                       constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1.0},),
                       options={'maxiter': 500, 'ftol': 1e-10})
    w_sub = res_sub.x
    sub_pred = np.tensordot(w_sub, P_sub, axes=([0], [0]))
    greedy_w_score = float(np.mean((sub_pred - Y_flat) ** 2))
    print(f'\nGreedy subset re-weighted val MSE: {greedy_w_score:.4f}')
    print(f'  subset weights: {dict(zip(greedy_subset, [round(float(x), 3) for x in w_sub]))}')

    # --- Write a CSV for every strategy so the user can pick what to submit ---
    candidates = [
        ('free',     w_free,  avail, free_score),
        ('cons',     w_con,   avail, con_score),
        ('greedyEq', np.ones(M_sub) / M_sub, greedy_subset, greedy_eq_score),
        ('greedyW',  w_sub,   greedy_subset, greedy_w_score),
    ]
    print('\n=== Summary ===')
    for nm, _, sub, s in candidates:
        print(f'  {nm:9s} M={len(sub):2d}  val MSE={s:.4f}')

    cols = ['id'] + [f'entity_{i}_time_{t}_{axis}'
                     for t in range(HORIZON) for i in range(11) for axis in ['x', 'y']]

    for tag, w_chosen, labels_chosen, score in candidates:
        idx_chosen = [avail.index(l) for l in labels_chosen]
        T = torch.stack([P_test[i] for i in idx_chosen], dim=0)
        M_chosen = T.shape[0]
        w_t = torch.tensor(w_chosen, dtype=T.dtype)
        test_pred = (w_t.view(M_chosen, 1, 1, 1, 1) * T).sum(dim=0)
        N_test = test_pred.shape[0]
        rows = [[idx] + test_pred[idx].reshape(-1).cpu().tolist() for idx in range(N_test)]
        out = SOL_DIR / f'solution_weighted_{tag}_{len(labels_chosen)}m.csv'
        pd.DataFrame(rows, columns=cols).set_index('id').sort_index().to_csv(out)
        print(f'  wrote {out.name}  (local val MSE {score:.4f})')

    print('\n--- Skipping the legacy single-CSV write below; all 4 have been emitted above. ---')
    return

    # Apply to test preds — only use the test tensors that match labels_chosen.
    idx_chosen = [avail.index(l) for l in labels_chosen]
    T = torch.stack([P_test[i] for i in idx_chosen], dim=0)  # [M', N_test, 12, 11, 2]
    M_chosen = T.shape[0]
    w_t = torch.tensor(w_chosen, dtype=T.dtype)
    test_pred = (w_t.view(M_chosen, 1, 1, 1, 1) * T).sum(dim=0)
    N_test = test_pred.shape[0]

    cols = ['id'] + [f'entity_{i}_time_{t}_{axis}'
                     for t in range(HORIZON) for i in range(11) for axis in ['x', 'y']]
    # IDs: assume sorted 0..N_test-1 since save_test_tta_preds saves in sorted order.
    rows = []
    for idx in range(N_test):
        rows.append([idx] + test_pred[idx].reshape(-1).cpu().tolist())

    out = SOL_DIR / f'solution_weighted_{tag}_{len(labels_chosen)}m.csv'
    pd.DataFrame(rows, columns=cols).set_index('id').sort_index().to_csv(out)
    print(f'\nWrote {out}')
    print(f'Models used: {labels_chosen}')
    print(f'Weights:     {[round(float(x), 3) for x in w_chosen]}')


if __name__ == '__main__':
    main()
