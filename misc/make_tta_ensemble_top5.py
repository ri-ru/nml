"""
Generate the Kaggle CSV for the greedy-best ensemble:
  v46 + v58 + v46_s1 + v46_s3 + v46_s3_swa

Loads all 5 models, runs 4-flip TTA per model on each test sample,
averages all 20 predictions per test sample, writes one CSV.
Local val score: 2.7905 (greedy best).
"""
import os
import torch
import pandas as pd
from pathlib import Path
import importlib.util

DEVICE = 'cuda'
TRAIN = r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\train\train'
TEST  = r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\test\test'
BASE = Path(__file__).parent.parent
WEIGHTS = BASE / 'models' / 'weights'
SOLUTIONS = BASE / 'models' / 'solutions'
SOLUTIONS.mkdir(parents=True, exist_ok=True)
CONTEXT = 8
HORIZON = 12

MODELS = [
    ('v46',         'nba_v46',    'NBAv46Model', 'model_v46_300ep.pth'),
    ('v58',         'nba_v58',    'NBAv58Model', 'model_v58_300ep.pth'),
    ('v46_s1',      'nba_v46_s1', 'NBAv46Model', 'model_v46_s1_300ep.pth'),
    ('v46_s3',      'nba_v46_s3', 'NBAv46Model', 'model_v46_s3_300ep.pth'),
    ('v46_s3_swa',  'nba_v46_s3', 'NBAv46Model', 'model_v46_s3_swa.pth'),
]


def compute_normalization(data_path):
    all_pos = []
    for f in [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt')]:
        seq = torch.load(f, weights_only=False)
        all_pos.append(seq[:, :, [0, 1]])
    cat = torch.cat(all_pos, dim=0)
    return cat.mean(dim=(0, 1)), cat.std(dim=(0, 1))


def load_one(model_file, model_class, weights_fname):
    spec = importlib.util.spec_from_file_location(model_file, BASE / 'models' / f'{model_file}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, model_class)
    m = cls().to(DEVICE)
    m.load_state_dict(torch.load(WEIGHTS / weights_fname, weights_only=True))
    m.train(False)
    return m


def predict_tta(model, ctx_norm, mu_d, sigma_d):
    """ctx_norm: [B=1, T_ctx, N, F]. Returns [12, 11, 2] in feet (TTA-averaged)."""
    preds_real = []
    for fx, fy in [(False, False), (True, False), (False, True), (True, True)]:
        x = ctx_norm.clone()
        if fx: x[:, :, :, 0] = -x[:, :, :, 0]
        if fy: x[:, :, :, 1] = -x[:, :, :, 1]
        with torch.no_grad():
            p_norm = model(x)
        p_real = p_norm * sigma_d + mu_d
        if fx:
            p_real = p_real.clone()
            p_real[:, :, 0] = -p_real[:, :, 0] + 2 * mu_d[0]
        if fy:
            p_real = p_real.clone()
            p_real[:, :, 1] = -p_real[:, :, 1] + 2 * mu_d[1]
        preds_real.append(p_real)
    return torch.stack(preds_real).mean(dim=0)  # [12, 11, 2]


def main():
    print('Computing normalization from train set...')
    mu, sigma = compute_normalization(TRAIN)
    mu_d, sigma_d = mu.to(DEVICE), sigma.to(DEVICE)

    print(f'Loading {len(MODELS)} models...')
    models = {}
    for label, mf, mc, wf in MODELS:
        models[label] = load_one(mf, mc, wf)
        print(f'  {label}: {wf}')

    test_files = sorted(
        [f for f in os.listdir(TEST) if f.endswith('.pt')],
        key=lambda f: int(f.removesuffix('.pt'))
    )
    print(f'Test files: {len(test_files)}')

    cols = ['id'] + [f'entity_{i}_time_{t}_{axis}'
                     for t in range(HORIZON) for i in range(11) for axis in ['x', 'y']]
    rows = []
    for idx, fname in enumerate(test_files):
        if idx % 100 == 0:
            print(f'  {idx}/{len(test_files)}')
        seq_raw = torch.load(os.path.join(TEST, fname), weights_only=False)
        seq_norm = seq_raw.clone()
        seq_norm[:, :, [0, 1]] = (seq_norm[:, :, [0, 1]] - mu) / sigma
        ctx = seq_norm[-CONTEXT:].unsqueeze(0).to(DEVICE)

        # Per-model TTA-averaged prediction in feet, then average across models
        per_model = [predict_tta(m, ctx, mu_d, sigma_d) for m in models.values()]
        avg = torch.stack(per_model).mean(dim=0)  # [12, 11, 2]

        row = [int(fname.removesuffix('.pt'))] + avg.reshape(-1).cpu().tolist()
        rows.append(row)

    out = SOLUTIONS / 'solution_tta_ensemble_top5_v46_v58_v46s1_v46s3_v46s3swa.csv'
    pd.DataFrame(rows, columns=cols).set_index('id').sort_index().to_csv(out)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
