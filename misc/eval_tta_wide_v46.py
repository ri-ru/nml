"""
Wider TTA eval on v46×300ep against val set.
Tests:
  - flips:    4 variants  (orig, x-flip, y-flip, xy-flip)
  - rot:      4 variants  (90° CCW of each flip)
  - noise:    4 variants  (σ=0.02 Gaussian noise on context, fixed seeds)
Reports MSE for several subsets so we can pick the best for the Kaggle CSV.
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
NOISE_STD = 0.02


def compute_normalization_from_files(file_list):
    all_pos = []
    for f in file_list:
        seq = torch.load(f, weights_only=False)
        all_pos.append(seq[:, :, [0, 1]])
    cat = torch.cat(all_pos, dim=0)
    return cat.mean(dim=(0, 1)), cat.std(dim=(0, 1))


def load_v46():
    spec = importlib.util.spec_from_file_location('nba_v46', BASE / 'models' / 'nba_v46.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.NBAv46Model().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS / 'model_v46_300ep.pth', weights_only=True))
    model.train(False)
    return model


def rot90_ccw(xy):
    """xy: [..., 2] → 90° CCW rotation: (x, y) → (-y, x)."""
    return torch.stack([-xy[..., 1], xy[..., 0]], dim=-1)


def rot90_cw(xy):
    """90° CW: (x, y) → (y, -x)."""
    return torch.stack([xy[..., 1], -xy[..., 0]], dim=-1)


def predict_variants(model, X_norm):
    """Returns dict variant_name → pred [T_pred, B*N, 2] (in normalized coords).
    Each variant transforms the context input, runs model, then UNDOES the transform on output.
    """
    out = {}
    # 4 flips
    for fx, fy in [(False, False), (True, False), (False, True), (True, True)]:
        x = X_norm.clone()
        if fx: x[:, :, :, 0] = -x[:, :, :, 0]
        if fy: x[:, :, :, 1] = -x[:, :, :, 1]
        with torch.no_grad():
            p = model(x).clone()
        if fx: p[:, :, 0] = -p[:, :, 0]
        if fy: p[:, :, 1] = -p[:, :, 1]
        out[f'flip_{int(fx)}{int(fy)}'] = p

    # 4 with 90° CCW (applied AFTER each flip — but a 90° rot+flip combo collapses,
    # so we keep simpler set: just rot-CCW of each flip variant in input).
    for fx, fy in [(False, False), (True, False), (False, True), (True, True)]:
        x = X_norm.clone()
        if fx: x[:, :, :, 0] = -x[:, :, :, 0]
        if fy: x[:, :, :, 1] = -x[:, :, :, 1]
        x[:, :, :, :2] = rot90_ccw(x[:, :, :, :2])
        with torch.no_grad():
            p = model(x).clone()
        # Undo rotation: apply 90° CW
        p = rot90_cw(p)
        if fx: p[:, :, 0] = -p[:, :, 0]
        if fy: p[:, :, 1] = -p[:, :, 1]
        out[f'rot_{int(fx)}{int(fy)}'] = p

    # 4 with small Gaussian noise on context positions (fixed seeds for determinism)
    for s in range(4):
        g = torch.Generator(device=X_norm.device).manual_seed(1000 + s)
        x = X_norm.clone()
        noise = torch.randn(x[:, :, :, :2].shape, generator=g, device=x.device) * NOISE_STD
        x[:, :, :, :2] = x[:, :, :, :2] + noise
        with torch.no_grad():
            p = model(x).clone()
        out[f'noise_{s}'] = p

    return out


def eval_subset(preds_by_batch, names, Y_real_by_batch, mu_d, sigma_d):
    """preds_by_batch: list of dicts (one per batch) variant_name → pred-norm [T, BN, 2].
    Y_real_by_batch: list of [T, BN, 2] in feet.
    Returns mean MSE in feet.
    """
    total = 0.0
    for d, Y in zip(preds_by_batch, Y_real_by_batch):
        avg = torch.stack([d[n] for n in names]).mean(dim=0)
        avg_real = avg * sigma_d + mu_d
        total += F.mse_loss(avg_real, Y).item()
    return total / len(preds_by_batch)


def main():
    print('Loading v46×300ep + val...')
    train_files = TRAIN_TXT.read_text().splitlines()
    val_files   = VAL_TXT.read_text().splitlines()
    mu, sigma = compute_normalization_from_files(train_files)
    mu_d, sigma_d = mu.to(DEVICE), sigma.to(DEVICE)
    model = load_v46()

    val_seqs = []
    for f in val_files:
        seq = torch.load(f, weights_only=False)
        seq_norm = seq.clone()
        seq_norm[:, :, [0, 1]] = (seq_norm[:, :, [0, 1]] - mu) / sigma
        val_seqs.append(seq_norm)

    sample_indices = []
    window = CONTEXT + HORIZON
    for i, seq in enumerate(val_seqs):
        ms = max(0, len(seq) - window)
        for start in (0, ms // 2, ms):
            sample_indices.append((i, start))
    print(f'Total samples: {len(sample_indices)}')

    BATCH = 64
    preds_by_batch = []
    Y_real_by_batch = []

    for b_start in range(0, len(sample_indices), BATCH):
        b = sample_indices[b_start:b_start + BATCH]
        Xs, Ys = [], []
        for i, s in b:
            seq = val_seqs[i]
            Xs.append(seq[s:s + CONTEXT])
            Ys.append(seq[s + CONTEXT:s + window])
        X = torch.stack(Xs).to(DEVICE)
        Y = torch.stack(Ys).to(DEVICE)
        Bsize, _, N, _ = Y.shape
        Y_real = Y[:, :, :, :2].permute(1, 0, 2, 3).reshape(HORIZON, Bsize * N, 2) * sigma_d + mu_d

        d = predict_variants(model, X)
        preds_by_batch.append(d)
        Y_real_by_batch.append(Y_real)
        if (b_start // BATCH) % 10 == 0:
            print(f'  batch {b_start // BATCH + 1}/{(len(sample_indices) + BATCH - 1) // BATCH}')

    flip_names  = [f'flip_{a}{b}' for a in '01' for b in '01']
    rot_names   = [f'rot_{a}{b}'  for a in '01' for b in '01']
    noise_names = [f'noise_{i}'   for i in range(4)]

    configs = {
        '4_flips_baseline':       flip_names,
        '4_rot_only':             rot_names,
        '4_noise_only':           noise_names,
        '8_flips_plus_rot':       flip_names + rot_names,
        '8_flips_plus_noise':     flip_names + noise_names,
        '8_rot_plus_noise':       rot_names + noise_names,
        '12_all':                 flip_names + rot_names + noise_names,
    }

    print('\n=== TTA config sweep on val (MSE in feet) ===')
    print(f'{"config":35s} {"score":>8s}')
    for name, sel in configs.items():
        s = eval_subset(preds_by_batch, sel, Y_real_by_batch, mu_d, sigma_d)
        print(f'{name:35s} {s:8.4f}')

    # Also per-variant individual scores (sanity check)
    print('\n=== Per-variant individual scores ===')
    for n in flip_names + rot_names + noise_names:
        s = eval_subset(preds_by_batch, [n], Y_real_by_batch, mu_d, sigma_d)
        print(f'{n:20s} {s:8.4f}')


if __name__ == '__main__':
    main()
