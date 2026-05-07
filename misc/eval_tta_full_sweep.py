"""
Full TTA + ensemble sweep on val set.

Models:
  v46     - baseline (model_v46_300ep.pth)
  v58     - pseudo-labeled (model_v58_300ep.pth)
  v46_s1  - seed 1 final (model_v46_s1_300ep.pth)
  v46_s2  - seed 2 final
  v46_s3  - seed 3 final
  v46_s1_swa - seed 1 SWA (avg of ep 280/285/290/295/300)
  v46_s2_swa - seed 2 SWA
  v46_s3_swa - seed 3 SWA

For each: 4-flip TTA → per-model TTA prediction.
Sweep ensembles (mean of subsets), report MSE in feet.
"""
import os
import torch
import torch.nn.functional as F
from itertools import combinations
from pathlib import Path
import importlib.util

DEVICE = 'cuda'
BASE = Path(__file__).parent.parent
WEIGHTS = BASE / 'models' / 'weights'
TRAIN_TXT = BASE / 'train_files.txt'
VAL_TXT   = BASE / 'val_files.txt'
CONTEXT = 8
HORIZON = 12

# (label, source_model_file, model_class, weights_filename)
MODELS = [
    ('v46',         'nba_v46', 'NBAv46Model', 'model_v46_300ep.pth'),
    ('v58',         'nba_v58', 'NBAv58Model', 'model_v58_300ep.pth'),
    ('v46_s1',      'nba_v46_s1', 'NBAv46Model', 'model_v46_s1_300ep.pth'),
    ('v46_s2',      'nba_v46_s2', 'NBAv46Model', 'model_v46_s2_300ep.pth'),
    ('v46_s3',      'nba_v46_s3', 'NBAv46Model', 'model_v46_s3_300ep.pth'),
    ('v46_s1_swa',  'nba_v46_s1', 'NBAv46Model', 'model_v46_s1_swa.pth'),
    ('v46_s2_swa',  'nba_v46_s2', 'NBAv46Model', 'model_v46_s2_swa.pth'),
    ('v46_s3_swa',  'nba_v46_s3', 'NBAv46Model', 'model_v46_s3_swa.pth'),
]


def compute_normalization_from_files(file_list):
    all_pos = []
    for f in file_list:
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


def predict_tta_norm(model, X_norm):
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
    print('Loading val + computing normalization...')
    train_files = TRAIN_TXT.read_text().splitlines()
    val_files   = VAL_TXT.read_text().splitlines()
    mu, sigma = compute_normalization_from_files(train_files)
    mu_d, sigma_d = mu.to(DEVICE), sigma.to(DEVICE)

    print(f'Loading {len(MODELS)} models...')
    models = {}
    for label, mf, mc, wf in MODELS:
        models[label] = load_one(mf, mc, wf)
        print(f'  {label}: {wf}')

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
    # store predictions per (batch_idx, model_label)
    preds_by_batch = []
    Y_real_by_batch = []
    n_batches = (len(sample_indices) + BATCH - 1) // BATCH

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

        d = {}
        for label, m in models.items():
            pn = predict_tta_norm(m, X)
            d[label] = pn * sigma_d + mu_d
        preds_by_batch.append(d)
        Y_real_by_batch.append(Y_real)
        if (b_start // BATCH) % 10 == 0:
            print(f'  batch {b_start // BATCH + 1}/{n_batches}')

    def eval_combo(labels):
        total = 0.0
        for d, Y in zip(preds_by_batch, Y_real_by_batch):
            avg = torch.stack([d[l] for l in labels]).mean(dim=0)
            total += F.mse_loss(avg, Y).item()
        return total / len(preds_by_batch)

    labels = [m[0] for m in MODELS]

    # Per-model individual scores
    print('\n=== Per-model TTA solo scores ===')
    solo = {}
    for l in labels:
        s = eval_combo([l])
        solo[l] = s
        print(f'  {l:15s} {s:.4f}')

    # Curated ensemble combos (likely useful)
    combos = {
        'baseline_v46':          ['v46'],
        'v46+v58':               ['v46', 'v58'],
        'v46+v46_s3':            ['v46', 'v46_s3'],
        'v46+v58+v46_s3':        ['v46', 'v58', 'v46_s3'],
        'all_seeds_final':       ['v46', 'v46_s1', 'v46_s2', 'v46_s3'],
        'all_seeds_swa':         ['v46', 'v46_s1_swa', 'v46_s2_swa', 'v46_s3_swa'],
        'v46+swa+v58':           ['v46', 'v46_s1_swa', 'v46_s2_swa', 'v46_s3_swa', 'v58'],
        'best_solo_top3':        sorted(labels, key=lambda l: solo[l])[:3],
        'best_solo_top5':        sorted(labels, key=lambda l: solo[l])[:5],
        'all_8':                 labels,
    }

    print('\n=== Curated ensemble combos ===')
    print(f'{"combo":35s} {"size":>5s} {"score":>9s} {"models":40s}')
    results = []
    for name, sel in combos.items():
        s = eval_combo(sel)
        results.append((name, s, sel))
        print(f'{name:35s} {len(sel):5d} {s:9.4f} {",".join(sel)[:40]}')

    # Greedy forward selection: start from best solo, keep adding the model that helps most
    print('\n=== Greedy forward selection ===')
    used = [min(labels, key=lambda l: solo[l])]
    print(f'Start: {used[0]} solo = {solo[used[0]]:.4f}')
    while len(used) < len(labels):
        cands = [(l, eval_combo(used + [l])) for l in labels if l not in used]
        best_l, best_s = min(cands, key=lambda t: t[1])
        cur = eval_combo(used)
        if best_s < cur:
            used.append(best_l)
            print(f'  +{best_l} -> {best_s:.4f} (was {cur:.4f})')
        else:
            print(f'  no improvement; stop')
            break
    print(f'\nGreedy best: {used} -> {eval_combo(used):.4f}')


if __name__ == '__main__':
    main()
