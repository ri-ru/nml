"""Generate pseudo-training files from ANY solution CSV. Round-2+ self-training.

Usage: python misc/make_pseudo_generic.py <solution_csv_path>
Writes pseudo_train_round2/<id>.pt and train_files_pseudo_round2.txt (real train + pseudo).
"""
import os
import sys
from pathlib import Path

import pandas as pd
import torch

BASE = Path(__file__).parent.parent
from _config import TEST as _TEST
TEST_DIR = Path(_TEST)
PSEUDO_DIR = BASE / 'pseudo_train_round2'
PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_TXT = BASE / 'train_files.txt'
OUT_TXT = BASE / 'train_files_pseudo_round2.txt'
CONTEXT, HORIZON, N_AGENTS = 8, 12, 11


def main():
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not csv or not csv.exists():
        print(f'CSV not found: {csv}')
        sys.exit(1)
    print(f'Loading {csv.name}...')
    df = pd.read_csv(csv, index_col='id')
    assert len(df.columns) == HORIZON * N_AGENTS * 2, f'bad cols {len(df.columns)}'

    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.pt')],
                        key=lambda f: int(f.removesuffix('.pt')))
    pseudo_paths, skipped = [], 0
    for fname in test_files:
        tid = int(fname.removesuffix('.pt'))
        if tid not in df.index:
            skipped += 1; continue
        seq = torch.load(TEST_DIR / fname, weights_only=False)
        if len(seq) < CONTEXT:
            skipped += 1; continue
        ctx = seq[-CONTEXT:].clone()
        pred_xy = torch.tensor(df.loc[tid].values, dtype=torch.float32).reshape(HORIZON, N_AGENTS, 2)
        meta = ctx[0:1, :, 2:4].expand(HORIZON, -1, -1).clone()
        future = torch.cat([pred_xy, meta], dim=-1)
        full = torch.cat([ctx, future], dim=0)
        out_path = PSEUDO_DIR / f'{tid}.pt'
        torch.save(full, out_path)
        pseudo_paths.append(str(out_path))

    combined = TRAIN_TXT.read_text().splitlines() + pseudo_paths
    OUT_TXT.write_text('\n'.join(combined))
    print(f'Wrote {len(pseudo_paths)} pseudo (skipped {skipped}); {OUT_TXT.name} = {len(combined)} total')


if __name__ == '__main__':
    main()
