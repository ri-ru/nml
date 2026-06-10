"""Decompose the local validation MSE by error source.

Question: where does our error come from? Is the ball the dominant contributor?

Uses the SAME val protocol as evaluate.py (3 windows/seq via _Sampler), and the
SAME denormalized MSE. Breaks total error down by:
  - ball vs the 10 players (per-entity mean squared error)
  - per timestep (1..12)
  - per axis (x, y)
  - per individual entity index
Optionally runs 4-flip TTA to match the submission protocol.

Run:
  python ensemble/analyze_error_sources.py                              # v46_300ep, plain
  python ensemble/analyze_error_sources.py --tta                        # 4-flip TTA
  python ensemble/analyze_error_sources.py --weights model_v60_w8_300ep.pth --tta
"""
import sys
import importlib.util

import torch

from _config import WEIGHTS, ARCH_DIR as MODELS_DIR, ROOT
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_TTA = '--tta' in sys.argv


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# all v60_* variants reuse the NBAv46Model architecture; only the weight file changes.
WEIGHT_NAME = _arg('--weights', 'model_v46_300ep.pth')


def load_model(py_name, cls_name, weight_name):
    spec = importlib.util.spec_from_file_location(py_name, MODELS_DIR / f'{py_name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = getattr(mod, cls_name)().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS / weight_name, weights_only=True))
    model.train(False)
    return mod, model


def forward_tta(model, X, mu_d, sigma_d):
    """4-flip TTA in normalized space; returns [T, B*N, 2] normalized prediction."""
    if not USE_TTA:
        return model(X)
    B, C, N, Fdim = X.shape
    preds = []
    for flip_x, flip_y in [(False, False), (True, False), (False, True), (True, True)]:
        x = X.clone()
        if flip_x:
            x[:, :, :, 0] = -x[:, :, :, 0]
        if flip_y:
            x[:, :, :, 1] = -x[:, :, :, 1]
        p = model(x)  # [T, B*N, 2] normalized
        # undo flip in normalized space: norm pos flips sign around 0 only after
        # accounting for mean. Easiest: denorm, flip about mu, renorm.
        p_real = p * sigma_d + mu_d
        if flip_x:
            p_real[:, :, 0] = -p_real[:, :, 0] + 2 * mu_d[0]
        if flip_y:
            p_real[:, :, 1] = -p_real[:, :, 1] + 2 * mu_d[1]
        preds.append((p_real - mu_d) / sigma_d)
    return torch.stack(preds).mean(0)


def main():
    mod, model = load_model('nba_v46', 'NBAv46Model', WEIGHT_NAME)

    # the model file lives in ensemble/models/, so point its split lists at the repo root
    mod.TRAIN_TXT = ROOT / 'train_files.txt'
    mod.VAL_TXT = ROOT / 'val_files.txt'
    trainer = mod.Trainer(None, None, device=DEVICE)
    mu = trainer.mu.to(DEVICE)
    sigma = trainer.sigma.to(DEVICE)

    sys.path.insert(0, str(ROOT / 'core'))
    import evaluate
    val_dl = evaluate.make_val_dl(trainer, batch_size=64)

    # accumulators: squared error summed per (timestep, entity, axis), and counts
    T, N = 12, 11
    se_ten = torch.zeros(T, N, 2, device=DEVICE)   # sum of squared error
    cnt_ten = torch.zeros(T, N, 2, device=DEVICE)  # element count
    ball_se = 0.0
    ball_cnt = 0
    # determine ball index per sample via mask feature 2 == 0
    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = forward_tta(model, X, mu, sigma)  # [T, B*N, 2] normalized
            B, Ty, Ny, Fdim = y.shape
            y_n = y[:, :, :, :2].permute(1, 0, 2, 3).reshape(Ty, B * Ny, 2)
            pred_r = pred * sigma + mu
            y_r = y_n * sigma + mu
            se = (pred_r - y_r) ** 2  # [T, B*N, 2]
            se = se.reshape(Ty, B, Ny, 2)

            ball_mask = (X[:, 0, :, 2] == 0)  # [B, N]
            se_ten += se.sum(dim=1)           # sum over batch -> [T, N, 2]
            cnt_ten += B

            bm = ball_mask.unsqueeze(0).unsqueeze(-1).expand(Ty, B, Ny, 2)
            ball_se += se[bm].sum().item()
            ball_cnt += bm.sum().item()

    # ---- overall MSE (matches evaluate.py: mean over all T,B*N,2 then... ) ----
    # evaluate.py does loss_fn(pred,y) per batch (mean over all elems) then /len(dl).
    # That equals global mean over all elements since batches are equal-ish; we use
    # the exact global element mean here.
    total_se = se_ten.sum().item()
    total_cnt = cnt_ten.sum().item()
    overall_mse = total_se / total_cnt

    # per-entity MSE (mean over T and axis)
    per_entity = (se_ten.sum(dim=(0, 2)) / cnt_ten.sum(dim=(0, 2))).cpu()  # [N]

    ball_mse = ball_se / ball_cnt
    # players = all elements minus ball elements
    players_se = total_se - ball_se
    players_cnt = total_cnt - ball_cnt
    players_mse = players_se / players_cnt

    print(f'\n{"="*60}')
    print(f'ERROR DECOMPOSITION  (weights={WEIGHT_NAME}, TTA={USE_TTA})')
    print(f'{"="*60}')
    print(f'Overall MSE (uniform over 11 entities): {overall_mse:.4f}')
    print(f'  matches reported local score scale\n')

    print(f'{"Group":<14}{"MSE":>10}{"share of total err":>22}')
    print(f'{"-"*46}')
    print(f'{"BALL (1)":<14}{ball_mse:>10.4f}{ball_se/total_se*100:>20.1f}%')
    print(f'{"players (10)":<14}{players_mse:>10.4f}{players_se/total_se*100:>20.1f}%')
    print(f'\nBall MSE / player MSE ratio: {ball_mse/players_mse:.2f}x')
    print(f'Ball is 1/11=9.1% of entities but {ball_se/total_se*100:.1f}% of error\n')

    # if ball were as accurate as players, what would overall be?
    hypothetical = (players_se + players_mse * ball_cnt) / total_cnt
    print(f'If ball were as good as avg player: overall MSE -> {hypothetical:.4f} '
          f'({(overall_mse-hypothetical)/overall_mse*100:.1f}% lower)\n')

    print('Per-entity MSE (entity index order within each sample):')
    for n in range(N):
        bar = '#' * int(per_entity[n] / per_entity.max() * 40)
        print(f'  entity {n:2d}: {per_entity[n]:7.4f}  {bar}')

    # per-timestep, split ball vs players (ball is ALWAYS entity index 0)
    print('\nPer-timestep MSE (denormalized) — ball is entity 0:')
    print(f'  {"t":>3}{"overall":>10}{"ball":>10}{"players":>10}{"ball share":>12}')
    per_t = (se_ten.sum(dim=(1, 2)) / cnt_ten.sum(dim=(1, 2))).cpu()
    ball_t = (se_ten[:, 0, :].sum(dim=1) / cnt_ten[:, 0, :].sum(dim=1)).cpu()
    play_t = (se_ten[:, 1:, :].sum(dim=(1, 2)) / cnt_ten[:, 1:, :].sum(dim=(1, 2))).cpu()
    ball_share_t = (se_ten[:, 0, :].sum(dim=1) / se_ten.sum(dim=(1, 2))).cpu()
    for t in range(T):
        print(f'  {t+1:>3}{per_t[t]:>10.4f}{ball_t[t]:>10.4f}{play_t[t]:>10.4f}'
              f'{ball_share_t[t]*100:>11.1f}%')

    print('\nPer-axis MSE:')
    per_axis = (se_ten.sum(dim=(0, 1)) / cnt_ten.sum(dim=(0, 1))).cpu()
    print(f'  x: {per_axis[0]:.4f}   y: {per_axis[1]:.4f}')


if __name__ == '__main__':
    main()
