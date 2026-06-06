"""
Generate per-model test-set TTA predictions and save as both a tensor
(.pt for ensembling) and a Kaggle CSV (for direct submission).

Run after save_val_tta_preds.py — uses the same model list, so weighted
ensemble can later combine val preds (for weight fitting) with test
preds (for the actual submission).

Outputs (under models/test_preds/):
  <label>.pt   - [N_test, 12, 11, 2] in feet
  <label>.csv  - Kaggle-format solution CSV
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
OUT_DIR = BASE / 'models' / 'test_preds'
SOL_DIR = BASE / 'models' / 'solutions'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOL_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT = 8
HORIZON = 12

MODELS = [
    ('v46',         'nba_v46',    'NBAv46Model', 'model_v46_300ep.pth'),
    ('v58',         'nba_v58',    'NBAv58Model', 'model_v58_300ep.pth'),
    ('v58_swa',     'nba_v58',    'NBAv58Model', 'model_v58_swa.pth'),
    ('v46_s1',      'nba_v46_s1', 'NBAv46Model', 'model_v46_s1_300ep.pth'),
    ('v46_s2',      'nba_v46_s2', 'NBAv46Model', 'model_v46_s2_300ep.pth'),
    ('v46_s3',      'nba_v46_s3', 'NBAv46Model', 'model_v46_s3_300ep.pth'),
    ('v46_s4',      'nba_v46_s4', 'NBAv46Model', 'model_v46_s4_300ep.pth'),
    ('v46_s5',      'nba_v46_s5', 'NBAv46Model', 'model_v46_s5_300ep.pth'),
    ('v46_s6',      'nba_v46_s6', 'NBAv46Model', 'model_v46_s6_300ep.pth'),
    ('v46_s7',      'nba_v46_s7', 'NBAv46Model', 'model_v46_s7_300ep.pth'),
    ('v46_s8',      'nba_v46_s8', 'NBAv46Model', 'model_v46_s8_300ep.pth'),
    ('v46_s1_swa',  'nba_v46_s1', 'NBAv46Model', 'model_v46_s1_swa.pth'),
    ('v46_s2_swa',  'nba_v46_s2', 'NBAv46Model', 'model_v46_s2_swa.pth'),
    ('v46_s3_swa',  'nba_v46_s3', 'NBAv46Model', 'model_v46_s3_swa.pth'),
    ('v46_s4_swa',  'nba_v46_s4', 'NBAv46Model', 'model_v46_s4_swa.pth'),
    ('v46_s5_swa',  'nba_v46_s5', 'NBAv46Model', 'model_v46_s5_swa.pth'),
    ('v46_s6_swa',  'nba_v46_s6', 'NBAv46Model', 'model_v46_s6_swa.pth'),
    ('v46_s7_swa',  'nba_v46_s7', 'NBAv46Model', 'model_v46_s7_swa.pth'),
    ('v46_s8_swa',  'nba_v46_s8', 'NBAv46Model', 'model_v46_s8_swa.pth'),
    ('v58_s1',      'nba_v58_s1', 'NBAv58Model', 'model_v58_s1_300ep.pth'),
    ('v58_s2',      'nba_v58_s2', 'NBAv58Model', 'model_v58_s2_300ep.pth'),
    ('v58_s3',      'nba_v58_s3', 'NBAv58Model', 'model_v58_s3_300ep.pth'),
    ('v58_s1_swa',  'nba_v58_s1', 'NBAv58Model', 'model_v58_s1_swa.pth'),
    ('v58_s2_swa',  'nba_v58_s2', 'NBAv58Model', 'model_v58_s2_swa.pth'),
    ('v58_s3_swa',  'nba_v58_s3', 'NBAv58Model', 'model_v58_s3_swa.pth'),
    ('v60',         'nba_v60',    'NBAv60Model', 'model_v60_200ep.pth'),
    ('v60_s1',      'nba_v60_s1', 'NBAv60Model', 'model_v60_s1_200ep.pth'),
    ('v60_swa',     'nba_v60',    'NBAv60Model', 'model_v60_swa.pth'),
    ('v60_s1_swa',  'nba_v60_s1', 'NBAv60Model', 'model_v60_s1_swa.pth'),
    # --- ball-weighted (NBAv46Model arch) + agent-centric + pseudo round-2 ---
    ('v60_w5',      'nba_v46',    'NBAv46Model', 'model_v60_w5_300ep.pth'),
    ('v60_w8',      'nba_v46',    'NBAv46Model', 'model_v60_w8_300ep.pth'),
    ('v60_w3',      'nba_v46',    'NBAv46Model', 'model_v60_w3_300ep.pth'),
    ('v65',         'nba_v65',    'NBAv65Model', 'model_v65_300ep.pth'),
    ('v66',         'nba_v66',    'NBAv66Model', 'model_v66_300ep.pth'),
    ('v68',         'nba_v46',    'NBAv46Model', 'model_v68_300ep.pth'),
    ('v60_w8_s1',   'nba_v46',    'NBAv46Model', 'model_v60_w8_s1_300ep.pth'),
    ('v60_w8_s2',   'nba_v46',    'NBAv46Model', 'model_v60_w8_s2_300ep.pth'),
    ('v60_w5_s1',   'nba_v46',    'NBAv46Model', 'model_v60_w5_s1_300ep.pth'),
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
    """ctx_norm: [1, T_ctx, N, F]. Returns [12, 11, 2] in feet (TTA-averaged)."""
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
    return torch.stack(preds_real).mean(dim=0)


def main():
    print('Computing normalization...')
    mu, sigma = compute_normalization(TRAIN)
    mu_d, sigma_d = mu.to(DEVICE), sigma.to(DEVICE)

    test_files = sorted(
        [f for f in os.listdir(TEST) if f.endswith('.pt')],
        key=lambda f: int(f.removesuffix('.pt'))
    )
    print(f'Test files: {len(test_files)}')

    # Pre-load + normalize all test sequences once.
    test_ctx = []
    test_ids = []
    for fname in test_files:
        seq_raw = torch.load(os.path.join(TEST, fname), weights_only=False)
        seq_norm = seq_raw.clone()
        seq_norm[:, :, [0, 1]] = (seq_norm[:, :, [0, 1]] - mu) / sigma
        ctx = seq_norm[-CONTEXT:]
        test_ctx.append(ctx)
        test_ids.append(int(fname.removesuffix('.pt')))

    cols = ['id'] + [f'entity_{i}_time_{t}_{axis}'
                     for t in range(HORIZON) for i in range(11) for axis in ['x', 'y']]

    for label, mf, mc, wf in MODELS:
        out_pt = OUT_DIR / f'{label}.pt'
        out_csv = SOL_DIR / f'solution_tta_{label}.csv'
        if out_pt.exists():
            print(f'  skip {label} (already saved)')
            continue
        wpath = WEIGHTS / wf
        if not wpath.exists():
            print(f'  MISSING weights for {label}: {wf} — skipping')
            continue
        print(f'  predicting {label}...')
        model = load_one(mf, mc, wf)
        all_preds = []
        rows = []
        for idx, (ctx, _id) in enumerate(zip(test_ctx, test_ids)):
            if idx % 500 == 0:
                print(f'    {idx}/{len(test_ctx)}')
            ctx_b = ctx.unsqueeze(0).to(DEVICE)
            pred = predict_tta(model, ctx_b, mu_d, sigma_d)  # [12, 11, 2]
            all_preds.append(pred.cpu())
            rows.append([_id] + pred.reshape(-1).cpu().tolist())
        full = torch.stack(all_preds, dim=0)  # [N_test, 12, 11, 2]
        torch.save(full, out_pt)
        pd.DataFrame(rows, columns=cols).set_index('id').sort_index().to_csv(out_csv)
        del model
        torch.cuda.empty_cache()
        print(f'    -> {out_pt} ({tuple(full.shape)})')
        print(f'    -> {out_csv}')

    print('Done.')


if __name__ == '__main__':
    main()
