"""Deep ball-error diagnostic: WHERE the ball error comes from, to find the next lever.

Uses cached val TTA preds (val_preds/<label>.pt) aligned with Y_real.pt, and reconstructs the context
windows from val_files.txt in the SAME order (3 windows/seq: starts 0, ms//2, ms). Everything in feet.

Analyses (ball = entity 0):
  A. Concentration  - is ball error a heavy tail (few catastrophic seqs) or uniform?
  B. Context speed  - is error worse when the ball is slow (about to be passed) or fast (in flight)?
  C. CV baseline    - does the model beat a constant-velocity extrapolation of the ball? where not?
  D. Hedging/bias   - does the model under-move the ball (predict less displacement than actual)?
  E. Possession     - error vs ball-to-nearest-player distance at present (held vs in flight).
  F. Per-timestep   - v60_w8 (best ball model) vs v46 baseline.

Run: python misc/analyze_ball_deep.py
"""
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).parent.parent
VAL_DIR = BASE / 'models' / 'val_preds'
VAL_TXT = BASE / 'val_files.txt'
REPORT = BASE / 'BALL_DEEP_ANALYSIS.md'
CTX, HOR, WIN = 8, 12, 20
PRIMARY = 'v60_w8'   # best single ball model
BASELINE = 'v46'


def build_windows():
    """Return ctx [N,8,11,4] and present/vel arrays, in the Y_real.pt window order."""
    ctxs = []
    for f in VAL_TXT.read_text().splitlines():
        seq = torch.load(f, weights_only=False).numpy()  # [T,11,4] feet
        ms = max(0, len(seq) - WIN)
        for s in (0, ms // 2, ms):
            ctxs.append(seq[s:s + CTX])
    return np.stack(ctxs)  # [N,8,11,4]


def ball_mse_per_window(pred, gt):
    # pred/gt [N,12,11,2]; ball=0 -> [N]
    d = pred[:, :, 0, :] - gt[:, :, 0, :]
    return (d ** 2).mean(axis=(1, 2))


def main():
    Y = torch.load(VAL_DIR / 'Y_real.pt').numpy()              # [N,12,11,2]
    P = torch.load(VAL_DIR / f'{PRIMARY}.pt').numpy()
    Pb = torch.load(VAL_DIR / f'{BASELINE}.pt').numpy()
    ctx = build_windows()                                       # [N,8,11,4]
    N = Y.shape[0]
    assert ctx.shape[0] == N, f'window mismatch {ctx.shape[0]} vs {N}'
    L = [f'# Deep Ball-Error Analysis (model={PRIMARY}, baseline={BASELINE})\n', f'N windows: {N}\n']

    bm = ball_mse_per_window(P, Y)                             # [N] ball MSE per window
    bm_base = ball_mse_per_window(Pb, Y)
    L.append(f'\nMean ball MSE: {PRIMARY}={bm.mean():.3f}  {BASELINE}={bm_base.mean():.3f}\n')

    # ---- A. concentration ----
    order = np.sort(bm)[::-1]
    tot = order.sum()
    cum = np.cumsum(order) / tot
    p_for = lambda frac: int(np.searchsorted(cum, frac)) + 1
    L.append('\n## A. Concentration (is it a heavy tail?)\n')
    for frac in (0.5, 0.8, 0.9):
        k = p_for(frac); L.append(f'- top {k} windows ({k/N*100:.1f}%) = {frac*100:.0f}% of ball error\n')
    L.append(f'- median ball MSE {np.median(bm):.3f} vs mean {bm.mean():.3f} (mean>>median => heavy tail)\n')
    L.append(f'- worst 1% mean {order[:max(1,N//100)].mean():.1f}; best 50% mean {np.sort(bm)[:N//2].mean():.3f}\n')

    # present pos/vel of the ball (feet)
    ball_ctx = ctx[:, :, 0, :2]                               # [N,8,2]
    present = ball_ctx[:, -1]                                  # [N,2]
    vel = ball_ctx[:, -1] - ball_ctx[:, -2]                   # [N,2] last-step velocity
    speed = np.linalg.norm(vel, axis=1)                       # [N]

    # ---- B. context speed bins ----
    L.append('\n## B. Ball error vs context speed (last-step |v|, feet/frame)\n')
    qs = np.quantile(speed, [0, .25, .5, .75, 1.0])
    for i in range(4):
        m = (speed >= qs[i]) & (speed <= qs[i + 1] if i == 3 else speed < qs[i + 1])
        L.append(f'- speed [{qs[i]:.2f},{qs[i+1]:.2f}) n={m.sum():4d}  ball MSE {bm[m].mean():.3f}\n')

    # ---- C. constant-velocity ball baseline ----
    steps = np.arange(1, HOR + 1)[None, :, None]              # [1,12,1]
    cv = present[:, None, :] + steps * vel[:, None, :]        # [N,12,2]
    cv_mse = ((cv - Y[:, :, 0, :]) ** 2).mean(axis=(1, 2))    # [N]
    L.append('\n## C. Model vs constant-velocity ball baseline\n')
    L.append(f'- CV ball MSE {cv_mse.mean():.3f}  vs model {bm.mean():.3f} '
             f'({"model better" if bm.mean()<cv_mse.mean() else "CV better"})\n')
    win = (bm < cv_mse).mean()
    L.append(f'- model beats CV on {win*100:.1f}% of windows; CV beats model on {100-win*100:.1f}%\n')
    # where CV wins (model loses): characterize
    closs = bm > cv_mse
    L.append(f'- on windows where CV wins: mean speed {speed[closs].mean():.2f} vs overall {speed.mean():.2f}\n')

    # ---- D. hedging / displacement bias ----
    pred_disp = np.linalg.norm(P[:, -1, 0, :] - present, axis=1)   # predicted net ball move
    actual_disp = np.linalg.norm(Y[:, -1, 0, :] - present, axis=1)
    L.append('\n## D. Hedging (does the model under-move the ball?)\n')
    L.append(f'- mean predicted net displacement {pred_disp.mean():.2f} vs actual {actual_disp.mean():.2f} '
             f'(ratio {pred_disp.mean()/actual_disp.mean():.2f}; <1 = under-moves)\n')
    big = actual_disp > np.quantile(actual_disp, 0.75)
    L.append(f'- on big-move balls (top 25% actual disp): pred {pred_disp[big].mean():.2f} vs '
             f'actual {actual_disp[big].mean():.2f} (ratio {pred_disp[big].mean()/actual_disp[big].mean():.2f})\n')

    # ---- E. possession proxy: ball-to-nearest-player dist at present ----
    players = ctx[:, -1, 1:, :2]                              # [N,10,2]
    d2p = np.linalg.norm(players - present[:, None, :], axis=2).min(axis=1)  # [N]
    L.append('\n## E. Ball error vs nearest-player distance at present (possession proxy)\n')
    qd = np.quantile(d2p, [0, .25, .5, .75, 1.0])
    for i in range(4):
        m = (d2p >= qd[i]) & (d2p < qd[i + 1] if i < 3 else d2p <= qd[i + 1])
        L.append(f'- dist [{qd[i]:.1f},{qd[i+1]:.1f}) n={m.sum():4d}  ball MSE {bm[m].mean():.3f}\n')

    # ---- F. per-timestep ball MSE: model vs baseline ----
    L.append('\n## F. Per-timestep ball MSE\n')
    L.append(f'  {"t":>3}{PRIMARY:>10}{BASELINE:>10}{"CV":>10}\n')
    for t in range(HOR):
        mt = ((P[:, t, 0, :] - Y[:, t, 0, :]) ** 2).mean()
        bt = ((Pb[:, t, 0, :] - Y[:, t, 0, :]) ** 2).mean()
        ct = ((cv[:, t, :] - Y[:, t, 0, :]) ** 2).mean()
        L.append(f'  {t+1:>3}{mt:>10.3f}{bt:>10.3f}{ct:>10.3f}\n')

    REPORT.write_text(''.join(L), encoding='utf-8')
    print(''.join(L))
    print(f'\nWrote {REPORT}')


if __name__ == '__main__':
    main()
