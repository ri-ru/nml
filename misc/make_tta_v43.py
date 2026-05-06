"""
TTA submission for v43 x100ep (local 3.1936 -- new best).
Generates: solution_tta_v43_solo.csv (4-variant TTA: orig, x-flip, y-flip, xy-flip)
"""
import os
import sys
import torch
import pandas as pd
from pathlib import Path
import importlib.util

TRAIN   = r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\train\train'
TEST    = r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\test\test'
DEVICE  = 'cuda'
BASE    = Path(__file__).parent.parent
WEIGHTS = BASE / 'models' / 'weights'
SOLUTIONS = BASE / 'models' / 'solutions'
SOLUTIONS.mkdir(parents=True, exist_ok=True)


def compute_normalization(data_path):
    all_pos = []
    for f in [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt')]:
        seq = torch.load(f, weights_only=False)
        all_pos.append(seq[:, :, [0, 1]])
    all_pos = torch.cat(all_pos, dim=0)
    return all_pos.mean(dim=(0, 1)), all_pos.std(dim=(0, 1))


def predict_tta(model, seq_raw, mu, sigma, context_size=8, device='cuda'):
    mu_d, sigma_d = mu.to(device), sigma.to(device)
    seq_norm = seq_raw.clone()
    seq_norm[:, :, [0, 1]] = (seq_norm[:, :, [0, 1]] - mu) / sigma
    ctx = seq_norm[-context_size:].unsqueeze(0).to(device)

    preds_real = []
    for flip_x, flip_y in [(False, False), (True, False), (False, True), (True, True)]:
        x = ctx.clone()
        if flip_x:
            x[:, :, :, 0] = -x[:, :, :, 0]
        if flip_y:
            x[:, :, :, 1] = -x[:, :, :, 1]
        with torch.no_grad():
            pred_norm = model(x)  # [12, 11, 2]
        pred_real = pred_norm * sigma_d + mu_d
        if flip_x:
            pred_real[:, :, 0] = -pred_real[:, :, 0] + 2 * mu_d[0]
        if flip_y:
            pred_real[:, :, 1] = -pred_real[:, :, 1] + 2 * mu_d[1]
        preds_real.append(pred_real)

    return torch.stack(preds_real).mean(dim=0)  # [12, 11, 2]


def main():
    print('Computing normalization from train set...')
    mu, sigma = compute_normalization(TRAIN)

    print('Loading nba_v43 model (100ep weights)...')
    spec = importlib.util.spec_from_file_location('nba_v43', BASE / 'models' / 'nba_v43.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.NBAv43Model().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS / 'model_v43.pth', weights_only=True))
    model.train(False)
    print('Model loaded.')

    test_files = sorted(
        [f for f in os.listdir(TEST) if f.endswith('.pt')],
        key=lambda f: int(f.removesuffix('.pt'))
    )

    cols = ['id'] + [f'entity_{i}_time_{t}_{axis}'
                     for t in range(12) for i in range(11) for axis in ['x', 'y']]
    all_rows = []
    for idx, fname in enumerate(test_files):
        if idx % 200 == 0:
            print(f'  {idx}/{len(test_files)}')
        seq_raw = torch.load(os.path.join(TEST, fname), weights_only=False)
        pred = predict_tta(model, seq_raw, mu, sigma, device=DEVICE)
        row = [int(fname.removesuffix('.pt'))] + pred.reshape(-1).cpu().tolist()
        all_rows.append(row)

    out = SOLUTIONS / 'solution_tta_v43_solo.csv'
    pd.DataFrame(all_rows, columns=cols).set_index('id').sort_index().to_csv(out)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
