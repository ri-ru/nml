"""Batch ball-vs-player decomposition over several checkpoints (loads val data once).

All listed checkpoints reuse the NBAv46Model architecture. Prints overall / ball / player MSE
and ball-share for each. Pass --tta for 4-flip TTA. Edit WEIGHTS list or pass names as args.

  python misc/analyze_batch.py [--tta] [model_v60_w2_50ep.pth ...]
"""
import sys
import importlib.util
from pathlib import Path

import torch

BASE = Path(__file__).parent.parent
WEIGHTS = BASE / 'models' / 'weights'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_TTA = '--tta' in sys.argv

names = [a for a in sys.argv[1:] if a.endswith('.pth')]
if not names:
    names = ['model_v60_w2_50ep.pth', 'model_v60_w3_50ep.pth',
             'model_v60_w5_50ep.pth', 'model_v60_w8_50ep.pth']


def load_v46():
    spec = importlib.util.spec_from_file_location('nba_v46', BASE / 'models' / 'nba_v46.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def forward(model, X, mu_d, sigma_d):
    if not USE_TTA:
        return model(X)
    preds = []
    for fx, fy in [(False, False), (True, False), (False, True), (True, True)]:
        x = X.clone()
        if fx:
            x[:, :, :, 0] = -x[:, :, :, 0]
        if fy:
            x[:, :, :, 1] = -x[:, :, :, 1]
        p = model(x) * sigma_d + mu_d
        if fx:
            p[:, :, 0] = -p[:, :, 0] + 2 * mu_d[0]
        if fy:
            p[:, :, 1] = -p[:, :, 1] + 2 * mu_d[1]
        preds.append((p - mu_d) / sigma_d)
    return torch.stack(preds).mean(0)


def evaluate_weights(mod, model, val_dl, mu, sigma):
    se = torch.zeros(12, 11, 2, device=DEVICE)
    cnt = 0
    with torch.no_grad():
        for X, y in val_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = forward(model, X, mu, sigma)
            B, T, N, _ = y.shape
            y_n = y[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
            pr = pred * sigma + mu
            yr = y_n * sigma + mu
            s = ((pr - yr) ** 2).reshape(T, B, N, 2).sum(dim=1)  # [T,N,2]
            se += s
            cnt += B
    total = se.sum().item() / (cnt * 11 * 12 * 2)
    ball = se[:, 0, :].sum().item() / (cnt * 12 * 2)
    players = se[:, 1:, :].sum().item() / (cnt * 10 * 12 * 2)
    ball_share = se[:, 0, :].sum().item() / se.sum().item()
    return total, ball, players, ball_share


def main():
    mod = load_v46()
    trainer = mod.Trainer(None, None, device=DEVICE)
    mu, sigma = trainer.mu.to(DEVICE), trainer.sigma.to(DEVICE)
    sys.path.insert(0, str(BASE))
    import evaluate
    val_dl = evaluate.make_val_dl(trainer, batch_size=64)

    model = mod.NBAv46Model().to(DEVICE)
    print(f'\nTTA={USE_TTA}')
    print(f'{"checkpoint":<26}{"overall":>9}{"ball":>9}{"players":>9}{"ball%":>8}{"ratio":>7}')
    print('-' * 68)
    rows = []
    for nm in names:
        fp = WEIGHTS / nm
        if not fp.exists():
            print(f'{nm:<26}  (missing)')
            continue
        model.load_state_dict(torch.load(fp, weights_only=True))
        model.train(False)
        ov, ball, pl, sh = evaluate_weights(mod, model, val_dl, mu, sigma)
        rows.append((nm, ov, ball, pl, sh))
        print(f'{nm:<26}{ov:>9.4f}{ball:>9.3f}{pl:>9.4f}{sh*100:>7.1f}%{ball/pl:>7.2f}')
    if rows:
        best = min(rows, key=lambda r: r[1])
        print(f'\nbest overall: {best[0]}  ({best[1]:.4f})')
        bestball = min(rows, key=lambda r: r[2])
        print(f'best ball:    {bestball[0]}  (ball {bestball[2]:.3f})')


if __name__ == '__main__':
    main()
