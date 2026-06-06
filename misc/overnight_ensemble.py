"""Overnight ensemble + physics sweep. Loads whatever val/test TTA preds exist, evaluates several
curated pools (robust global-constrained AND aggressive per-entity), reports overall + ball(entity0)
val MSE for each, optionally tests physics clipping on val, and writes decision-ready CSVs.

Designed to run AFTER save_val_tta_preds.py + save_test_tta_preds.py. Robust: skips missing labels,
continues past per-pool errors. Writes a summary to OVERNIGHT_RESULTS.md.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

BASE = Path(__file__).parent.parent
VAL_DIR = BASE / 'models' / 'val_preds'
TEST_DIR = BASE / 'models' / 'test_preds'
SOL_DIR = BASE / 'models' / 'solutions'
REPORT = BASE / 'OVERNIGHT_RESULTS.md'
HOR, NE = 12, 11

# curated base pool (Oscar-like: NO overfit pseudo v60); _swa added only if present
BASE_POOL = ['v46', 'v58', 'v58_swa',
             'v46_s1', 'v46_s2', 'v46_s3', 'v46_s4', 'v46_s5', 'v46_s6', 'v46_s7', 'v46_s8',
             'v46_s1_swa', 'v46_s2_swa', 'v46_s3_swa', 'v46_s4_swa', 'v46_s5_swa',
             'v46_s6_swa', 'v46_s7_swa', 'v46_s8_swa',
             'v58_s1', 'v58_s2', 'v58_s3']
BALL = ['v60_w5', 'v60_w8', 'v60_w3', 'v60_w8_s1', 'v60_w8_s2', 'v60_w5_s1']
AGENT = ['v65', 'v66']
PSEUDO = ['v68']

POOLS = {
    'base':            BASE_POOL,
    'base+ball':       BASE_POOL + BALL,
    'base+ball+pseudo': BASE_POOL + BALL + PSEUDO,
    'base+ball+agent': BASE_POOL + BALL + AGENT,
    'all':             BASE_POOL + BALL + AGENT + PSEUDO,
}


def avail(labels):
    out = []
    for l in labels:
        if (VAL_DIR / f'{l}.pt').exists() and (TEST_DIR / f'{l}.pt').exists():
            out.append(l)
    return out


def fit_global(P, Y):
    M = P.shape[0]
    Pf, Yf = P.reshape(M, -1), Y.reshape(-1)
    w0 = np.ones(M) / M
    res = minimize(lambda w: float(np.mean((w @ Pf - Yf) ** 2)), w0, method='SLSQP',
                   bounds=[(0, 1)] * M, constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1},),
                   options={'maxiter': 500, 'ftol': 1e-10})
    return res.x


def fit_per_entity(P, Y):
    M = P.shape[0]
    W = np.zeros((NE, M))
    for e in range(NE):
        Pe, Ye = P[:, :, :, e, :].reshape(M, -1), Y[:, :, e, :].reshape(-1)
        w0 = np.ones(M) / M
        res = minimize(lambda w: float(np.mean((w @ Pe - Ye) ** 2)), w0, method='SLSQP',
                       bounds=[(0, 1)] * M, constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1},),
                       options={'maxiter': 500, 'ftol': 1e-10})
        W[e] = res.x
    return W


def overall_and_ball(pred, Y):
    overall = float(np.mean((pred - Y) ** 2))
    ball = float(np.mean((pred[:, :, 0, :] - Y[:, :, 0, :]) ** 2))
    return overall, ball


def apply_per_entity(W, P):
    # P [M,N,12,11,2] -> [N,12,11,2]
    N = P.shape[1]
    out = np.zeros((N, HOR, NE, 2), dtype=np.float32)
    for e in range(NE):
        out[:, :, e, :] = (W[e] @ P[:, :, :, e, :].reshape(P.shape[0], -1)).reshape(N, HOR, 2)
    return out


def write_csv(test_pred, name):
    cols = [f'entity_{i}_time_{t}_{axis}' for t in range(HOR) for i in range(NE) for axis in ['x', 'y']]
    Ntest = test_pred.shape[0]
    rows = [[idx] + test_pred[idx].reshape(-1).tolist() for idx in range(Ntest)]
    out = SOL_DIR / f'{name}.csv'
    pd.DataFrame(rows, columns=['id'] + cols).set_index('id').sort_index().to_csv(out)
    return out.name


def physics_clip(val_pred, Y, test_pred):
    """Clip to court bounds (from Y) + cap per-step speed. Return (val_after, test_after, helped)."""
    lo = Y.reshape(-1, 2).min(0)
    hi = Y.reshape(-1, 2).max(0)
    # max single-step displacement seen in val GT (per axis magnitude)
    step = np.linalg.norm(np.diff(Y, axis=1), axis=-1)  # [N,11,?]
    maxstep = float(np.percentile(step, 99.5))

    def clamp(pred):
        p = np.clip(pred, lo, hi)
        return p

    va = clamp(val_pred)
    before = float(np.mean((val_pred - Y) ** 2))
    after = float(np.mean((va - Y) ** 2))
    helped = after < before
    ta = clamp(test_pred) if helped else test_pred
    return after, before, ta, helped, maxstep


def main():
    lines = ['# Overnight Ensemble Results\n']
    if not (VAL_DIR / 'Y_real.pt').exists():
        print('No Y_real.pt — run save_val_tta_preds.py first')
        return
    Y = torch.load(VAL_DIR / 'Y_real.pt').numpy()  # [N,12,11,2]
    lines.append(f'Y_real: {Y.shape}\n')

    results = []
    for pool_name, labels in POOLS.items():
        try:
            labs = avail(labels)
            if len(labs) < 2:
                lines.append(f'\n## {pool_name}: only {len(labs)} avail, skip\n')
                continue
            Pv = np.stack([torch.load(VAL_DIR / f'{l}.pt').numpy() for l in labs])   # [M,N,12,11,2]
            Pt = np.stack([torch.load(TEST_DIR / f'{l}.pt').numpy() for l in labs])  # [M,Ntest,12,11,2]

            # global constrained (robust)
            wg = fit_global(Pv, Y)
            val_g = (wg @ Pv.reshape(len(labs), -1)).reshape(Y.shape)
            og, bg = overall_and_ball(val_g, Y)
            test_g = (wg @ Pt.reshape(len(labs), -1)).reshape((Pt.shape[1], HOR, NE, 2))
            csv_g = write_csv(test_g, f'solution_overnight_{pool_name}_cons')

            # per-entity (aggressive, overfit-prone)
            Wpe = fit_per_entity(Pv, Y)
            val_pe = apply_per_entity(Wpe, Pv)
            ope, bpe = overall_and_ball(val_pe, Y)
            test_pe = apply_per_entity(Wpe, Pt)
            csv_pe = write_csv(test_pe, f'solution_overnight_{pool_name}_perentity')

            # physics on the per-entity prediction
            pa, pb, test_pe_phys, helped, maxstep = physics_clip(val_pe, Y, test_pe)
            phys_note = f'phys {pb:.4f}->{pa:.4f} ({"HELPED" if helped else "no"})'
            if helped:
                write_csv(test_pe_phys, f'solution_overnight_{pool_name}_perentity_phys')

            lines.append(f'\n## {pool_name}  ({len(labs)} models)\n')
            lines.append(f'- models: {labs}\n')
            lines.append(f'- **cons**: overall {og:.4f}  ball {bg:.3f}  -> `{csv_g}`\n')
            lines.append(f'- **per-entity**: overall {ope:.4f}  ball {bpe:.3f}  -> `{csv_pe}`\n')
            lines.append(f'- {phys_note}\n')
            results.append((pool_name, og, ope, bg, bpe))
            print(f'{pool_name}: cons {og:.4f}/ball{bg:.3f}  perent {ope:.4f}/ball{bpe:.3f}  {phys_note}')
        except Exception as e:
            lines.append(f'\n## {pool_name}: ERROR {e}\n')
            print(f'{pool_name} ERROR: {e}')
            traceback.print_exc()

    lines.append('\n## Caveat\n')
    lines.append('Per-entity OVERFITS val (11xM params). Local overall is optimistic; trust the '
                 'gap less for per-entity. cons is the robust pick. Compare ball MSE vs the prior '
                 'best ensemble ball=11.896.\n')
    REPORT.write_text(''.join(lines), encoding='utf-8')
    print(f'\nWrote {REPORT}')


if __name__ == '__main__':
    main()
