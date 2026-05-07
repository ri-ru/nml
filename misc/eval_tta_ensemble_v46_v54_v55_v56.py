"""
Local TTA ensemble evaluation: v46(300ep) + v54(250ep) + v55(250ep) + v56(250ep).

For each val sample (3 fixed starts per sequence — same as evaluate.py):
  1. For each model, apply 4-flip TTA → per-model TTA prediction (denormalized).
  2. Average across the 4 models → ensemble TTA prediction.
  3. Accumulate MSE in feet.

Also reports per-model TTA-only score for comparison vs solo non-TTA.
"""
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import importlib.util

DEVICE = 'cuda'
BASE = Path(__file__).parent.parent
WEIGHTS = BASE / 'models' / 'weights'
TRAIN_TXT = BASE / 'train_files.txt'
VAL_TXT   = BASE / 'val_files.txt'
CONTEXT = 8
HORIZON = 12


def compute_normalization_from_files(file_list):
    all_pos = []
    for f in file_list:
        seq = torch.load(f, weights_only=False)
        all_pos.append(seq[:, :, [0, 1]])
    return torch.cat(all_pos, dim=0).mean(dim=(0, 1)), torch.cat(all_pos, dim=0).std(dim=(0, 1))


def load_model(name, model_class_name, weights_filename):
    spec = importlib.util.spec_from_file_location(name, BASE / 'models' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, model_class_name)
    model = cls().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS / weights_filename, weights_only=True))
    model.train(False)
    return model


def predict_tta_norm(model, ctx_norm):
    """ctx_norm: [B, T_ctx, N, F] already normalized.
    Returns 4-flip averaged prediction in NORMALIZED coords: [T_pred, B*N, 2]."""
    preds = []
    for flip_x, flip_y in [(False, False), (True, False), (False, True), (True, True)]:
        x = ctx_norm.clone()
        if flip_x:
            x[:, :, :, 0] = -x[:, :, :, 0]
        if flip_y:
            x[:, :, :, 1] = -x[:, :, :, 1]
        with torch.no_grad():
            p = model(x)  # [T_pred, B*N, 2] in normalized coords
        # Undo the flip on the prediction (in normalized space — symmetric around 0)
        if flip_x:
            p = p.clone()
            p[:, :, 0] = -p[:, :, 0]
        if flip_y:
            p = p.clone()
            p[:, :, 1] = -p[:, :, 1]
        preds.append(p)
    return torch.stack(preds).mean(dim=0)


def main():
    print('Loading val + train file lists...')
    train_files = TRAIN_TXT.read_text().splitlines()
    val_files   = VAL_TXT.read_text().splitlines()

    print('Computing normalization from train_files.txt (matches training)...')
    mu, sigma = compute_normalization_from_files(train_files)
    mu_d, sigma_d = mu.to(DEVICE), sigma.to(DEVICE)
    print(f'  mu={mu.tolist()}  sigma={sigma.tolist()}')

    print('Loading 4 models...')
    models = {
        'v46': load_model('nba_v46', 'NBAv46Model', 'model_v46_300ep.pth'),
        'v54': load_model('nba_v54', 'NBAv54Model', 'model_v54_250ep.pth'),
        'v55': load_model('nba_v55', 'NBAv55Model', 'model_v55_250ep.pth'),
        'v56': load_model('nba_v56', 'NBAv56Model', 'model_v56_250ep.pth'),
    }
    print(f'  loaded: {list(models.keys())}')

    # Pre-load val sequences and normalize
    print(f'Loading {len(val_files)} val sequences...')
    val_seqs = []
    for f in val_files:
        seq = torch.load(f, weights_only=False)
        seq_norm = seq.clone()
        seq_norm[:, :, [0, 1]] = (seq_norm[:, :, [0, 1]] - mu) / sigma
        val_seqs.append(seq_norm)

    # Same sampler as evaluate.py: starts (0, max_start//2, max_start) per sequence
    sample_indices = []  # list of (seq_idx, start)
    window_size = CONTEXT + HORIZON
    for i, seq in enumerate(val_seqs):
        max_start = max(0, len(seq) - window_size)
        for start in (0, max_start // 2, max_start):
            sample_indices.append((i, start))
    print(f'Total samples: {len(sample_indices)}')

    # MSE accumulators in feet (denormalized) per source: {model_name: total}, plus 'ensemble'
    sources = list(models.keys()) + ['ensemble']
    totals = {s: 0.0 for s in sources}
    n_batches = 0

    BATCH = 64
    for b_start in range(0, len(sample_indices), BATCH):
        b = sample_indices[b_start:b_start + BATCH]
        # Build batch
        Xs, Ys = [], []
        for seq_idx, start in b:
            seq = val_seqs[seq_idx]
            Xs.append(seq[start:start + CONTEXT])
            Ys.append(seq[start + CONTEXT:start + window_size])
        X = torch.stack(Xs).to(DEVICE)            # [B, T_ctx, N, F]
        Y = torch.stack(Ys).to(DEVICE)            # [B, T_pred, N, F]
        Bsize, _, N, _ = Y.shape
        Y_real = Y[:, :, :, :2].permute(1, 0, 2, 3).reshape(HORIZON, Bsize * N, 2) * sigma_d + mu_d

        # Per-model TTA predictions (denormalized)
        per_model_real = {}
        for name, m in models.items():
            pred_norm = predict_tta_norm(m, X)              # [T_pred, B*N, 2] normalized
            pred_real = pred_norm * sigma_d + mu_d          # feet
            per_model_real[name] = pred_real
            mse = F.mse_loss(pred_real, Y_real)
            totals[name] += mse.item()

        # Ensemble = average of the 4 model TTA predictions
        ens_real = torch.stack(list(per_model_real.values()), dim=0).mean(dim=0)
        totals['ensemble'] += F.mse_loss(ens_real, Y_real).item()

        n_batches += 1
        if (b_start // BATCH) % 10 == 0:
            print(f'  batch {b_start // BATCH + 1}/{(len(sample_indices) + BATCH - 1) // BATCH}')

    print('\n=== TTA Local Scores (val set, MSE in feet) ===')
    for s in sources:
        print(f'  {s:10s}: {totals[s] / n_batches:.4f}')


if __name__ == '__main__':
    main()
