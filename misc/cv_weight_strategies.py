"""
Cross-validated honest evaluation of weighted-ensemble strategies.

Splits the val set into two halves, fits each strategy on one half, evaluates
on the other, then swaps. The mean held-out MSE is a much better Kaggle proxy
than the in-sample MSE (which the more-parameterized fits will overfit).
"""
import torch
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

BASE = Path(__file__).parent.parent
VAL_DIR = BASE / 'models' / 'val_preds'
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


def fit_cons(P, Y, M):
    w0 = np.ones(M) / M
    bounds = [(0.0, 1.0)] * M
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1.0},)
    res = minimize(lambda w, P=P, Y=Y: float(np.mean((w @ P - Y) ** 2)),
                   w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 500, 'ftol': 1e-10})
    return res.x


def strategy_cons(P_train, Y_train, P_test, Y_test, M):
    """Single weight vector, constrained."""
    P_tr = P_train.reshape(M, -1)
    Y_tr = Y_train.reshape(-1)
    w = fit_cons(P_tr, Y_tr, M)
    pred = (w @ P_test.reshape(M, -1)).reshape(Y_test.shape)
    return float(np.mean((pred - Y_test) ** 2))


def strategy_per_type(P_train, Y_train, P_test, Y_test, M):
    """Two weight vectors: one for entity 0 (ball), one for entities 1-10 (players)."""
    # Ball: entity 0
    P_ball_tr = P_train[:, :, :, 0, :].reshape(M, -1)
    Y_ball_tr = Y_train[:, :, 0, :].reshape(-1)
    w_ball = fit_cons(P_ball_tr, Y_ball_tr, M)
    # Players: entities 1..10
    P_play_tr = P_train[:, :, :, 1:, :].reshape(M, -1)
    Y_play_tr = Y_train[:, :, 1:, :].reshape(-1)
    w_play = fit_cons(P_play_tr, Y_play_tr, M)
    # Apply to test
    pred_ball = (w_ball @ P_test[:, :, :, 0, :].reshape(M, -1)).reshape(Y_test[:, :, 0, :].shape)
    pred_play = (w_play @ P_test[:, :, :, 1:, :].reshape(M, -1)).reshape(Y_test[:, :, 1:, :].shape)
    mse_ball = float(np.mean((pred_ball - Y_test[:, :, 0, :]) ** 2))
    mse_play = float(np.mean((pred_play - Y_test[:, :, 1:, :]) ** 2))
    # Weighted by entity count (1 ball, 10 players)
    return (mse_ball * 1 + mse_play * 10) / 11


def strategy_per_entity(P_train, Y_train, P_test, Y_test, M):
    """11 weight vectors, one per entity."""
    out = np.zeros_like(Y_test)
    for e in range(N_ENTITIES):
        Pe_tr = P_train[:, :, :, e, :].reshape(M, -1)
        Ye_tr = Y_train[:, :, e, :].reshape(-1)
        w = fit_cons(Pe_tr, Ye_tr, M)
        Pe_te = P_test[:, :, :, e, :].reshape(M, -1)
        out[:, :, e, :] = (w @ Pe_te).reshape(Y_test[:, :, e, :].shape)
    return float(np.mean((out - Y_test) ** 2))


def strategy_per_entity_bc(P_train, Y_train, P_test, Y_test, M):
    """Per-entity + bias correction (bias measured on the TRAINING split, not test)."""
    out_train = np.zeros_like(Y_train)
    out_test = np.zeros_like(Y_test)
    for e in range(N_ENTITIES):
        Pe_tr = P_train[:, :, :, e, :].reshape(M, -1)
        Ye_tr = Y_train[:, :, e, :].reshape(-1)
        w = fit_cons(Pe_tr, Ye_tr, M)
        out_train[:, :, e, :] = (w @ Pe_tr).reshape(Y_train[:, :, e, :].shape)
        out_test[:, :, e, :] = (w @ P_test[:, :, :, e, :].reshape(M, -1)).reshape(Y_test[:, :, e, :].shape)
    bias = (out_train - Y_train).mean(axis=0)  # [12, 11, 2]
    out_test = out_test - bias[None, :, :, :]
    return float(np.mean((out_test - Y_test) ** 2))


def strategy_per_entity_time(P_train, Y_train, P_test, Y_test, M):
    """11 x 12 weight vectors."""
    out = np.zeros_like(Y_test)
    for e in range(N_ENTITIES):
        for t in range(HORIZON):
            Pet_tr = P_train[:, :, t, e, :].reshape(M, -1)
            Yet_tr = Y_train[:, t, e, :].reshape(-1)
            w = fit_cons(Pet_tr, Yet_tr, M)
            Pet_te = P_test[:, :, t, e, :].reshape(M, -1)
            out[:, t, e, :] = (w @ Pet_te).reshape(Y_test[:, t, e, :].shape)
    return float(np.mean((out - Y_test) ** 2))


def strategy_per_entity_time_bc(P_train, Y_train, P_test, Y_test, M):
    out_train = np.zeros_like(Y_train)
    out_test = np.zeros_like(Y_test)
    for e in range(N_ENTITIES):
        for t in range(HORIZON):
            Pet_tr = P_train[:, :, t, e, :].reshape(M, -1)
            Yet_tr = Y_train[:, t, e, :].reshape(-1)
            w = fit_cons(Pet_tr, Yet_tr, M)
            out_train[:, t, e, :] = (w @ Pet_tr).reshape(Y_train[:, t, e, :].shape)
            Pet_te = P_test[:, :, t, e, :].reshape(M, -1)
            out_test[:, t, e, :] = (w @ Pet_te).reshape(Y_test[:, t, e, :].shape)
    bias = (out_train - Y_train).mean(axis=0)
    out_test = out_test - bias[None, :, :, :]
    return float(np.mean((out_test - Y_test) ** 2))


def main():
    Y = torch.load(VAL_DIR / 'Y_real.pt').numpy()
    N = Y.shape[0]
    avail, P_val_list = [], []
    for lab in LABELS:
        vp = VAL_DIR / f'{lab}.pt'
        if not vp.exists():
            continue
        avail.append(lab)
        P_val_list.append(torch.load(vp).numpy())
    M = len(avail)
    P_val = np.stack(P_val_list, axis=0)  # [M, N, 12, 11, 2]
    print(f'M={M} models, N={N} val windows')

    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    half = N // 2
    idx_a = perm[:half]
    idx_b = perm[half:]

    strategies = [
        ('equal-weight (no fit)',          lambda Pt, Yt, Pe, Ye, M:
            float(np.mean((Pe.mean(axis=0) - Ye) ** 2))),
        ('cons (21 params)',               strategy_cons),
        ('per-type (42 params)',           strategy_per_type),
        ('per-entity (231 params)',        strategy_per_entity),
        ('per-entity + biascor',           strategy_per_entity_bc),
        ('per-(entity,time) (2772)',       strategy_per_entity_time),
        ('per-(entity,time) + biascor',    strategy_per_entity_time_bc),
    ]

    print(f'\n{"strategy":<35s}  {"in-sample":>10s}  {"held-out":>10s}  {"gap":>7s}')
    print('-' * 70)
    for name, fn in strategies:
        # In-sample MSE: fit on all of val, eval on all of val (same set).
        in_sample = fn(P_val, Y, P_val, Y, M)
        # Held-out: A->B, then B->A, average.
        mse_ab = fn(P_val[:, idx_a], Y[idx_a], P_val[:, idx_b], Y[idx_b], M)
        mse_ba = fn(P_val[:, idx_b], Y[idx_b], P_val[:, idx_a], Y[idx_a], M)
        held_out = 0.5 * (mse_ab + mse_ba)
        gap = held_out - in_sample
        print(f'{name:<35s}  {in_sample:>10.4f}  {held_out:>10.4f}  {gap:>+7.4f}')


if __name__ == '__main__':
    main()
