"""Custom per-entity error decomposition for the v69 split-head model.

v69 has a custom forward() (dict in training mode; [T,B*N,2] weighted-mean per entity in
eval mode), so it is compatible with evaluate.py's MSE loop, but we want a richer breakdown:
overall / ball(entity 0) / player MSE, plus the BALL HEDGING RATIO (predicted vs actual
ball net displacement; BALL_DEEP_ANALYSIS section D — aim toward 1.0).

Mirrors misc/analyze_error_sources.py's val protocol (evaluate.make_val_dl, 3 windows/seq,
denormalized MSE) but loads NBAv69Model with the right k_ball and supports the ball
'mean' vs 'mode' readout toggle.

Run (from repo root, with the venv python):
  python misc/eval_v69_ball.py --weights model_v69_k4_50ep.pth --k_ball 4
  python misc/eval_v69_ball.py --weights model_v69_k6_50ep.pth --k_ball 6
  python misc/eval_v69_ball.py --weights model_v69_k6_50ep.pth --k_ball 6 --readout mode
  python misc/eval_v69_ball.py --weights model_v69_k6_50ep.pth --k_ball 6 --tta
"""
import sys
import importlib.util
from pathlib import Path

import torch

BASE = Path(__file__).parent.parent
WEIGHTS = BASE / 'models' / 'weights'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


USE_TTA = '--tta' in sys.argv
WEIGHT_NAME = _arg('--weights', 'model_v69_k6_50ep.pth')
K_BALL = int(_arg('--k_ball', '6'))
READOUT = _arg('--readout', 'mean')   # 'mean' or 'mode'


def load_v69(weight_name, k_ball, readout):
    spec = importlib.util.spec_from_file_location('nba_v69', BASE / 'models' / 'nba_v69.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.NBAv69Model(k_ball=k_ball).to(DEVICE)
    model.ball_readout = readout
    model.load_state_dict(torch.load(WEIGHTS / weight_name, weights_only=True))
    model.train(False)
    return mod, model


def forward_tta(model, X, mu_d, sigma_d):
    if not USE_TTA:
        return model(X)
    preds = []
    for flip_x, flip_y in [(False, False), (True, False), (False, True), (True, True)]:
        x = X.clone()
        if flip_x:
            x[:, :, :, 0] = -x[:, :, :, 0]
        if flip_y:
            x[:, :, :, 1] = -x[:, :, :, 1]
        p = model(x)
        p_real = p * sigma_d + mu_d
        if flip_x:
            p_real[:, :, 0] = -p_real[:, :, 0] + 2 * mu_d[0]
        if flip_y:
            p_real[:, :, 1] = -p_real[:, :, 1] + 2 * mu_d[1]
        preds.append((p_real - mu_d) / sigma_d)
    return torch.stack(preds).mean(0)


def main():
    mod, model = load_v69(WEIGHT_NAME, K_BALL, READOUT)

    trainer = mod.Trainer(None, None, device=DEVICE)
    mu = trainer.mu.to(DEVICE)
    sigma = trainer.sigma.to(DEVICE)

    sys.path.insert(0, str(BASE))
    import evaluate
    val_dl = evaluate.make_val_dl(trainer, batch_size=64)

    T, N = 12, 11
    se_ten = torch.zeros(T, N, 2, device=DEVICE)
    cnt_ten = torch.zeros(T, N, 2, device=DEVICE)
    ball_se = 0.0
    ball_cnt = 0

    # hedging accumulators (denormalized feet): net displacement = ||pos[t=12] - pos[t_last_ctx]||
    # We follow BALL_DEEP_ANALYSIS D: net displacement over the 12-step horizon for the ball,
    # comparing predicted final pos vs actual final pos relative to the last context position.
    pred_disp_sum = 0.0
    actual_disp_sum = 0.0
    big_pred_disp_sum = 0.0
    big_actual_disp_sum = 0.0
    disp_records = []  # (actual_disp, pred_disp) for the big-move quartile split

    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = forward_tta(model, X, mu, sigma)  # [T, B*N, 2] normalized
            B, Ty, Ny, Fdim = y.shape
            y_n = y[:, :, :, :2].permute(1, 0, 2, 3).reshape(Ty, B * Ny, 2)
            pred_r = pred * sigma + mu
            y_r = y_n * sigma + mu
            se = (pred_r - y_r) ** 2
            se = se.reshape(Ty, B, Ny, 2)

            ball_mask = (X[:, 0, :, 2] == 0)  # [B, N]
            se_ten += se.sum(dim=1)
            cnt_ten += B
            bm = ball_mask.unsqueeze(0).unsqueeze(-1).expand(Ty, B, Ny, 2)
            ball_se += se[bm].sum().item()
            ball_cnt += bm.sum().item()

            # ---- hedging: ball net displacement over horizon ----
            # last context position of the ball, denormalized:
            last_ctx = X[:, -1, :, :2] * sigma + mu   # [B, N, 2]
            # ball is entity index 0 (always)
            last_ctx_ball = last_ctx[:, 0, :]          # [B, 2]
            pred_r_bn = pred_r.reshape(Ty, B, Ny, 2)
            y_r_bn = y_r.reshape(Ty, B, Ny, 2)
            pred_final_ball = pred_r_bn[-1, :, 0, :]   # [B, 2]
            actual_final_ball = y_r_bn[-1, :, 0, :]    # [B, 2]
            pred_disp = (pred_final_ball - last_ctx_ball).norm(dim=-1)      # [B]
            actual_disp = (actual_final_ball - last_ctx_ball).norm(dim=-1)  # [B]
            pred_disp_sum += pred_disp.sum().item()
            actual_disp_sum += actual_disp.sum().item()
            for a, p in zip(actual_disp.tolist(), pred_disp.tolist()):
                disp_records.append((a, p))

    total_se = se_ten.sum().item()
    total_cnt = cnt_ten.sum().item()
    overall_mse = total_se / total_cnt
    ball_mse = ball_se / ball_cnt
    players_se = total_se - ball_se
    players_cnt = total_cnt - ball_cnt
    players_mse = players_se / players_cnt

    n = len(disp_records)
    mean_pred_disp = pred_disp_sum / n
    mean_actual_disp = actual_disp_sum / n
    ratio = mean_pred_disp / mean_actual_disp

    # big-move quartile (top 25% by actual displacement)
    disp_records.sort(key=lambda r: r[0], reverse=True)
    top = disp_records[: max(1, n // 4)]
    big_actual = sum(r[0] for r in top) / len(top)
    big_pred = sum(r[1] for r in top) / len(top)
    big_ratio = big_pred / big_actual

    print(f'\n{"="*64}')
    print(f'v69 BALL DECOMP  weights={WEIGHT_NAME}  k_ball={K_BALL}  readout={READOUT}  TTA={USE_TTA}')
    print(f'{"="*64}')
    print(f'Overall MSE : {overall_mse:.4f}')
    print(f'Ball  MSE   : {ball_mse:.4f}   (entity 0)')
    print(f'Player MSE  : {players_mse:.4f}')
    print(f'Ball/player ratio: {ball_mse/players_mse:.2f}x   ball share of total err: '
          f'{ball_se/total_se*100:.1f}%')
    print()
    print(f'HEDGING (BALL_DEEP_ANALYSIS D) — predicted vs actual ball net displacement:')
    print(f'  all   : pred {mean_pred_disp:.2f} vs actual {mean_actual_disp:.2f}  '
          f'-> ratio {ratio:.3f}   (target 1.0; <1 = under-moves)')
    print(f'  big25%: pred {big_pred:.2f} vs actual {big_actual:.2f}  -> ratio {big_ratio:.3f}')
    print()
    print('Per-timestep ball MSE:')
    ball_t = (se_ten[:, 0, :].sum(dim=1) / cnt_ten[:, 0, :].sum(dim=1)).cpu()
    for t in range(T):
        print(f'  t={t+1:2d}: {ball_t[t]:7.4f}')

    # machine-readable summary line for easy scraping
    print(f'\nV69_SUMMARY weights={WEIGHT_NAME} k_ball={K_BALL} readout={READOUT} tta={USE_TTA} '
          f'overall={overall_mse:.4f} ball={ball_mse:.4f} player={players_mse:.4f} '
          f'hedge={ratio:.3f} hedge_big={big_ratio:.3f}')


if __name__ == '__main__':
    main()
