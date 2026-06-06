"""
Run each ensemble candidate model on the val split with 4-flip TTA.
Save per-model val predictions + ground truth to disk so we can later
optimize ensemble weights without re-running GPU inference.

Outputs (under models/val_preds/):
  Y_real.pt      - [N_windows, 12, 11, 2] ground truth in feet
  <label>.pt     - [N_windows, 12, 11, 2] TTA-averaged predictions in feet

Run: python misc/save_val_tta_preds.py
"""
import os
import torch
from pathlib import Path, PureWindowsPath
import importlib.util


def _default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = _default_device()
BASE = Path(__file__).parent.parent
WEIGHTS = BASE / 'models' / 'weights'
TRAIN_TXT = BASE / 'train_files.txt'
VAL_TXT = BASE / 'val_files.txt'
OUT_DIR = BASE / 'models' / 'val_preds'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT = 8
HORIZON = 12
BATCH = 64
DATA_DIR = os.environ.get('NML_DATA_DIR')  # optional rebase for cross-machine path lists


def _rebase(p: str) -> str:
    """Rebase Windows / foreign paths against NML_DATA_DIR if set."""
    if Path(p).exists() or DATA_DIR is None:
        return p
    name = PureWindowsPath(p).name or Path(p).name
    return str(Path(DATA_DIR) / name)


def _empty_cache():
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

# (label, source_model_file, model_class, weights_filename). Missing weights are skipped.
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
    # New: HiVT-style. Constructor needs (mu, sigma) — load_one detects this.
    ('v61',         'nba_v61',    'NBAv61Model', 'model_v61_300ep.pth'),
    # Hybrid: per-focal canonical frame + v46-style AR+MoG. Also needs (mu, sigma).
    ('v62',         'nba_v62',    'NBAv62Model', 'model_v62_300ep.pth'),
]


def compute_normalization_from_files(file_list):
    all_pos = []
    for f in file_list:
        seq = torch.load(f, weights_only=False)
        all_pos.append(seq[:, :, [0, 1]])
    cat = torch.cat(all_pos, dim=0)
    return cat.mean(dim=(0, 1)), cat.std(dim=(0, 1))


def load_one(model_file, model_class, weights_fname, mu=None, sigma=None):
    spec = importlib.util.spec_from_file_location(model_file, BASE / 'models' / f'{model_file}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, model_class)
    try:
        m = cls()
    except TypeError:
        # Models like v61 require (mu, sigma) at construction.
        m = cls(mu=mu, sigma=sigma)
    m = m.to(DEVICE)
    m.load_state_dict(torch.load(WEIGHTS / weights_fname, weights_only=True, map_location=DEVICE))
    m.train(False)
    return m


def predict_tta_norm(model, X_norm):
    """X_norm: [B, T_ctx, N, F]. Returns normalized [T_out, B*N, 2] averaged over 4 flips."""
    preds = []
    for fx, fy in [(False, False), (True, False), (False, True), (True, True)]:
        x = X_norm.clone()
        if fx: x[:, :, :, 0] = -x[:, :, :, 0]
        if fy: x[:, :, :, 1] = -x[:, :, :, 1]
        with torch.no_grad():
            p = model(x).clone()
        if fx: p[:, :, 0] = -p[:, :, 0]
        if fy: p[:, :, 1] = -p[:, :, 1]
        preds.append(p)
    return torch.stack(preds).mean(dim=0)


def main():
    print('Loading splits + computing normalization...')
    train_files = [_rebase(p) for p in TRAIN_TXT.read_text().splitlines()]
    val_files   = [_rebase(p) for p in VAL_TXT.read_text().splitlines()]
    mu, sigma = compute_normalization_from_files(train_files)
    mu_d, sigma_d = mu.to(DEVICE), sigma.to(DEVICE)

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
    N = len(sample_indices)
    print(f'Val windows: {N}')

    # Build all X, Y up front into batched tensors (per-batch lists).
    X_batches, Y_real_batches = [], []
    for b_start in range(0, N, BATCH):
        b = sample_indices[b_start:b_start + BATCH]
        Xs, Ys = [], []
        for i, s in b:
            seq = val_seqs[i]
            Xs.append(seq[s:s + CONTEXT])
            Ys.append(seq[s + CONTEXT:s + window])
        X = torch.stack(Xs).to(DEVICE)
        Y = torch.stack(Ys).to(DEVICE)
        Bsize, _, num_e, _ = Y.shape
        Y_real = Y[:, :, :, :2].permute(1, 0, 2, 3).reshape(HORIZON, Bsize * num_e, 2) * sigma_d + mu_d
        X_batches.append(X)
        Y_real_batches.append(Y_real)

    # Save Y_real once (shape [N, 12, 11, 2] in feet).
    Y_full = torch.cat(
        [Y.reshape(HORIZON, -1, 11, 2).permute(1, 0, 2, 3) for Y in Y_real_batches],
        dim=0,
    ).cpu()
    torch.save(Y_full, OUT_DIR / 'Y_real.pt')
    print(f'Saved Y_real.pt  shape={tuple(Y_full.shape)}')

    for label, mf, mc, wf in MODELS:
        out_path = OUT_DIR / f'{label}.pt'
        if out_path.exists():
            print(f'  skip {label} (already saved)')
            continue
        wpath = WEIGHTS / wf
        if not wpath.exists():
            print(f'  MISSING weights for {label}: {wf} — skipping')
            continue
        print(f'  predicting {label}...')
        model = load_one(mf, mc, wf, mu=mu, sigma=sigma)
        preds_batches = []
        for X in X_batches:
            pn = predict_tta_norm(model, X)
            pr = pn * sigma_d + mu_d  # [T_out, B*N, 2]
            Bsize = X.shape[0]
            pr = pr.reshape(HORIZON, Bsize, 11, 2).permute(1, 0, 2, 3)  # [B, 12, 11, 2]
            preds_batches.append(pr.cpu())
        full = torch.cat(preds_batches, dim=0)
        torch.save(full, out_path)
        del model
        _empty_cache()
        print(f'    -> {out_path}  shape={tuple(full.shape)}')

    print('Done.')


if __name__ == '__main__':
    main()
