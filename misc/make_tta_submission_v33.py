"""
TTA ensemble: v19, v23, v24, v26, v31, v33.
v33 = v26 arch x 250 epochs (best local so far). Same strategy as previous best ensemble.
"""
import os
import torch
import pandas as pd
from pathlib import Path
import sys

TRAIN = r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\train\train'
TEST  = r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\test\test'
DEVICE = 'cuda'
SOLUTIONS = Path(r'C:\Users\jan\downloads\nml_project_proposal\nml_project_proposal\models\solutions')
WEIGHTS   = Path(r'C:\Users\jan\downloads\nml_project_proposal\nml_project_proposal\models\weights')
MODELS_DIR = Path(r'C:\Users\jan\downloads\nml_project_proposal\nml_project_proposal\models')

sys.path.insert(0, str(Path(__file__).parent))


def compute_normalization(data_path):
    all_pos = []
    for f in [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt')]:
        seq = torch.load(f, weights_only=False)
        all_pos.append(seq[:, :, [0, 1]])
    all_pos = torch.cat(all_pos, dim=0)
    return all_pos.mean(dim=(0, 1)), all_pos.std(dim=(0, 1))


def predict_tta(model, seq_raw, mu, sigma, context_size=8, device='cuda'):
    mu_d = mu.to(device)
    sigma_d = sigma.to(device)

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

    return torch.stack(preds_real).mean(dim=0)


def load_model(module_name, model_class_name, weight_file,
               state_dim=128, context_size=8, horizon_size=12, num_heads=4, num_layers=2):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, MODELS_DIR / f'{module_name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ModelClass = getattr(mod, model_class_name)
    model = ModelClass(state_dim, context_size, horizon_size, num_heads, num_layers).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS / weight_file, weights_only=True))
    model.eval()
    return model


def main():
    print('Computing normalization stats...')
    mu, sigma = compute_normalization(TRAIN)

    print('Loading models...')
    models = [
        load_model('nba_v19', 'NBAv19Model', 'model_v19.pth'),
        load_model('nba_v23', 'NBAv23Model', 'model_v23.pth'),
        load_model('nba_v24', 'NBAv24Model', 'model_v24.pth'),
        load_model('nba_v26', 'NBAv26Model', 'model_v26.pth'),
        load_model('nba_v31', 'NBAv31Model', 'model_v31.pth'),
        load_model('nba_v33', 'NBAv33Model', 'model_v33.pth'),
    ]
    print(f'Loaded {len(models)} models')

    test_files = sorted([f for f in os.listdir(TEST) if f.endswith('.pt')],
                        key=lambda f: int(f.removesuffix('.pt')))

    all_traj = []
    for i, fname in enumerate(test_files):
        if i % 100 == 0:
            print(f'  {i}/{len(test_files)}')
        seq_raw = torch.load(os.path.join(TEST, fname), weights_only=False)

        model_preds = [predict_tta(m, seq_raw, mu, sigma, device=DEVICE) for m in models]
        avg_pred = torch.stack(model_preds).mean(dim=0)  # [12, 11, 2]
        traj = avg_pred.reshape(-1).cpu().tolist()
        all_traj.append([int(fname.removesuffix('.pt'))] + traj)

    cols = ['id'] + [f'entity_{i}_time_{t}_{axis}'
                     for t in range(12) for i in range(11) for axis in ['x', 'y']]
    out = SOLUTIONS / 'solution_tta_ensemble_v19_v23_v24_v26_v31_v33.csv'
    pd.DataFrame(all_traj, columns=cols).set_index('id').sort_index().to_csv(out)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
