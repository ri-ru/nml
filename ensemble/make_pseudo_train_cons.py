"""
Regenerate pseudo training files from the current best ensemble
(solution_weighted_cons_21m.csv, Kaggle 3.00676) rather than the older
v46x300ep solo. Outputs to pseudo_train_cons/ and writes
train_files_pseudo_cons.txt = train_files.txt + new pseudo files.

Run once before training models that use train_files_pseudo_cons.txt.
"""
import os
import pandas as pd
import torch
from pathlib import Path

BASE = Path(__file__).parent.parent
SOLUTIONS = BASE / 'models' / 'solutions'
TEST_DIR = Path(r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\test\test')
PSEUDO_DIR = BASE / 'pseudo_train_cons'
PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_TXT = BASE / 'train_files.txt'
TRAIN_TXT_PSEUDO = BASE / 'train_files_pseudo_cons.txt'

CONTEXT = 8
HORIZON = 12
N_AGENTS = 11

CSV = SOLUTIONS / 'solution_weighted_cons_21m.csv'


def main():
    print(f'Loading predictions from {CSV.name}...')
    df = pd.read_csv(CSV, index_col='id')
    print(f'  {len(df)} test predictions, {len(df.columns)} cols (expected 264)')
    assert len(df.columns) == HORIZON * N_AGENTS * 2

    test_files = sorted(
        [f for f in os.listdir(TEST_DIR) if f.endswith('.pt')],
        key=lambda f: int(f.removesuffix('.pt'))
    )
    print(f'Test files: {len(test_files)}')

    pseudo_paths = []
    skipped = 0
    for fname in test_files:
        tid = int(fname.removesuffix('.pt'))
        if tid not in df.index:
            skipped += 1
            continue
        seq = torch.load(TEST_DIR / fname, weights_only=False)
        if len(seq) < CONTEXT:
            skipped += 1
            continue
        ctx = seq[-CONTEXT:].clone()
        pred_xy = torch.tensor(df.loc[tid].values, dtype=torch.float32).reshape(HORIZON, N_AGENTS, 2)
        meta = ctx[0:1, :, 2:4].expand(HORIZON, -1, -1).clone()
        future = torch.cat([pred_xy, meta], dim=-1)
        full = torch.cat([ctx, future], dim=0)
        out_path = PSEUDO_DIR / f'{tid}.pt'
        torch.save(full, out_path)
        pseudo_paths.append(str(out_path))

    print(f'Wrote {len(pseudo_paths)} pseudo files; skipped {skipped}')

    train_paths = TRAIN_TXT.read_text().splitlines()
    combined = train_paths + pseudo_paths
    TRAIN_TXT_PSEUDO.write_text('\n'.join(combined))
    print(f'Wrote {TRAIN_TXT_PSEUDO} ({len(train_paths)} real + {len(pseudo_paths)} pseudo = {len(combined)} total)')


if __name__ == '__main__':
    main()
