"""
Generate pseudo training files from v46×300ep TTA predictions on test set.
Reads solution_tta_v46_300ep.csv (predictions in feet, denormalized).

For each test_id.pt:
  - Take last 8 frames (real context)
  - Append 12 predicted frames (positions only; copy entity_type & team from context)
  - Save to pseudo_train/<id>.pt as [20, 11, 4] tensor in raw feet

Build train_files_pseudo.txt = train_files.txt + all pseudo_train/<id>.pt paths.
"""
import os
import pandas as pd
import torch
from pathlib import Path

BASE = Path(__file__).parent.parent
from _config import SOL_DIR as SOLUTIONS, TEST as _TEST
TEST_DIR = Path(_TEST)
PSEUDO_DIR = BASE / 'pseudo_train'
PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_TXT = BASE / 'train_files.txt'
TRAIN_TXT_PSEUDO = BASE / 'train_files_pseudo.txt'

CONTEXT = 8
HORIZON = 12
N_AGENTS = 11

CSV = SOLUTIONS / 'solution_tta_v46_300ep.csv'


def main():
    print(f'Loading predictions from {CSV.name}...')
    df = pd.read_csv(CSV, index_col='id')
    print(f'  {len(df)} test predictions, {len(df.columns)} cols (expected 264)')
    assert len(df.columns) == HORIZON * N_AGENTS * 2, f'unexpected cols: {len(df.columns)}'

    test_files = sorted(
        [f for f in os.listdir(TEST_DIR) if f.endswith('.pt')],
        key=lambda f: int(f.removesuffix('.pt'))
    )
    print(f'Test files: {len(test_files)}')

    # Reshape pred row → [T_pred, N, 2]
    # CSV column order is: entity_{i}_time_{t}_{axis}, with t outer, i inner per script:
    # cols are constructed as: [for t in range(12): for i in range(11): for axis in xy]
    # so the flat order is t-major, then i, then axis.
    # df row reshape: [T, N, 2]
    pseudo_paths = []
    skipped = 0
    for fname in test_files:
        tid = int(fname.removesuffix('.pt'))
        if tid not in df.index:
            skipped += 1
            continue
        seq = torch.load(TEST_DIR / fname, weights_only=False)  # [M, N, 4] raw feet
        if len(seq) < CONTEXT:
            skipped += 1
            continue
        ctx = seq[-CONTEXT:].clone()  # [8, 11, 4]

        # Build predicted frames with same entity_type (col 2) and team (col 3) as context
        pred_xy = torch.tensor(df.loc[tid].values, dtype=torch.float32).reshape(HORIZON, N_AGENTS, 2)
        # Use first context frame's entity_type/team labels (constant across time)
        meta = ctx[0:1, :, 2:4].expand(HORIZON, -1, -1).clone()  # [12, 11, 2]
        future = torch.cat([pred_xy, meta], dim=-1)  # [12, 11, 4]

        full = torch.cat([ctx, future], dim=0)  # [20, 11, 4]
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
