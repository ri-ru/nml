"""
Solo (no TTA) submission for v44 x250ep.
Single forward pass, original orientation only.
Compare against solution_tta_v44_250ep.csv to isolate TTA effect.
"""
import os
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


def main():
    print('Computing normalization from train set...')
    mu, sigma = compute_normalization(TRAIN)
    mu_d, sigma_d = mu.to(DEVICE), sigma.to(DEVICE)

    print('Loading nba_v44 model (250ep weights)...')
    spec = importlib.util.spec_from_file_location('nba_v44', BASE / 'models' / 'nba_v44.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.NBAv44Model().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS / 'model_v44_250ep.pth', weights_only=True))
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
        seq_norm = seq_raw.clone()
        seq_norm[:, :, [0, 1]] = (seq_norm[:, :, [0, 1]] - mu) / sigma
        ctx = seq_norm[-8:].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_norm = model(ctx)  # [12, 11, 2]
        pred_real = pred_norm * sigma_d + mu_d
        row = [int(fname.removesuffix('.pt'))] + pred_real.reshape(-1).cpu().tolist()
        all_rows.append(row)

    out = SOLUTIONS / 'solution_solo_v44_250ep.csv'
    pd.DataFrame(all_rows, columns=cols).set_index('id').sort_index().to_csv(out)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
