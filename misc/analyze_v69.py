"""Ball-vs-player decomposition + ball net-displacement hedging ratio for v69 split-head models.

v69 uses a custom split head (proj_ball K_ball, proj_player K=2), so analyze_batch.py (which
loads NBAv46Model) cannot load these weights. This mirrors the val protocol (evaluate.make_val_dl,
3 windows/seq) and the denormalized MSE.

Reports per checkpoint: overall / ball(entity0) / player MSE, ball share, ratio, and the
predicted-vs-actual ball NET DISPLACEMENT ratio (BALL_DEEP_ANALYSIS section D; <1 = under-moves).

  python misc/analyze_v69.py model_v69_k4_50ep.pth:4 model_v69_k6_50ep.pth:6 [--mode]

Each arg is <weightfile>:<k_ball>. --mode evaluates the ball with the dominant-mode readout
instead of the weighted mean.
"""
import sys
import importlib.util
from pathlib import Path

import torch

BASE = Path(__file__).parent.parent
WEIGHTS = BASE / 'models' / 'weights'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_MODE = '--mode' in sys.argv

specs = []
for a in sys.argv[1:]:
    if a.endswith('.pth') or ':' in a and a.split(':')[0].endswith('.pth'):
        name, _, k = a.partition(':')
        specs.append((name, int(k) if k else 4))
if not specs:
    specs = [('model_v69_k4_50ep.pth', 4), ('model_v69_k6_50ep.pth', 6)]


def load_v69():
    spec = importlib.util.spec_from_file_location('nba_v69', BASE / 'models' / 'nba_v69.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate_weights(mod, model, val_dl, mu, sigma):
    se = torch.zeros(12, 11, 2, device=DEVICE)
    cnt = 0
    pred_disp_sum = 0.0       # sum over windows of predicted ball net displacement
    actual_disp_sum = 0.0
    disp_n = 0
    # big-move (top 25% actual) accounting needs all values -> collect per-window
    pred_disp_all, actual_disp_all = [], []
    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)                       # [T, B*N, 2] normalized
            B, T, N, _ = y.shape
            y_n = y[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
            pr = pred * sigma + mu
            yr = y_n * sigma + mu
            s = ((pr - yr) ** 2).reshape(T, B, N, 2).sum(dim=1)   # [T,N,2]
            se += s
            cnt += B

            # ball net displacement: |pos_T - pos_present|. present = last context frame.
            pr_bn = pr.reshape(T, B, N, 2)[:, :, 0, :]   # [T, B, 2] predicted ball
            yr_bn = yr.reshape(T, B, N, 2)[:, :, 0, :]   # [T, B, 2] actual ball
            present = (X[:, -1, 0, :2] * sigma + mu)     # [B, 2] ball present pos (entity 0)
            pred_disp = (pr_bn[-1] - present).norm(dim=-1)   # [B]
            actual_disp = (yr_bn[-1] - present).norm(dim=-1)  # [B]
            pred_disp_all.append(pred_disp.cpu())
            actual_disp_all.append(actual_disp.cpu())
            pred_disp_sum += pred_disp.sum().item()
            actual_disp_sum += actual_disp.sum().item()
            disp_n += B

    total = se.sum().item() / (cnt * 11 * 12 * 2)
    ball = se[:, 0, :].sum().item() / (cnt * 12 * 2)
    players = se[:, 1:, :].sum().item() / (cnt * 10 * 12 * 2)
    ball_share = se[:, 0, :].sum().item() / se.sum().item()

    pd = torch.cat(pred_disp_all)
    ad = torch.cat(actual_disp_all)
    ratio = pd.mean().item() / ad.mean().item()
    # big moves: top 25% by actual displacement
    k = max(1, int(0.25 * len(ad)))
    top_idx = ad.topk(k).indices
    big_ratio = pd[top_idx].mean().item() / ad[top_idx].mean().item()
    return total, ball, players, ball_share, ratio, pd.mean().item(), ad.mean().item(), big_ratio


def main():
    mod = load_v69()
    trainer = mod.nba_v46.Trainer(None, None, device=DEVICE)
    mu, sigma = trainer.mu.to(DEVICE), trainer.sigma.to(DEVICE)
    sys.path.insert(0, str(BASE))
    import evaluate
    val_dl = evaluate.make_val_dl(trainer, batch_size=64)

    print(f'\nball_readout={"mode" if USE_MODE else "mean"}')
    print(f'{"checkpoint":<28}{"overall":>9}{"ball":>9}{"players":>9}{"ball%":>7}{"ratio":>7}'
          f'{"disp_ratio":>11}{"big_ratio":>11}')
    print('-' * 91)
    for name, k_ball in specs:
        fp = WEIGHTS / name
        if not fp.exists():
            print(f'{name:<28}  (missing)')
            continue
        model = mod.NBAv69Model(k_ball=k_ball).to(DEVICE)
        model.ball_readout = 'mode' if USE_MODE else 'mean'
        model.load_state_dict(torch.load(fp, weights_only=True))
        model.train(False)
        ov, ball, pl, sh, dr, pdm, adm, br = evaluate_weights(mod, model, val_dl, mu, sigma)
        print(f'{name:<28}{ov:>9.4f}{ball:>9.3f}{pl:>9.4f}{sh*100:>6.1f}%'
              f'{ball/pl:>7.2f}{dr:>11.3f}{br:>11.3f}')
    print('\nbaselines (50ep): v46 overall=3.3115 ball=13.66 | v60_w8 overall=3.285 ball=12.83')


if __name__ == '__main__':
    main()
