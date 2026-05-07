"""
Ensemble TTA: average of v46×300ep + v54×250ep + v55×250ep + v56×250ep TTA CSVs.
All four are MoG K=2 + GRU + type-aware attention variants — they cluster tightly
in solo eval (v46=2.8628, v54=2.9235, v55=2.9212, v56=2.9206) so an ensemble
hopefully decorrelates their residuals.
Output: solution_tta_ensemble_v46_v54_v55_v56.csv
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
SOLUTIONS = BASE / 'models' / 'solutions'

paths = {
    'v46': SOLUTIONS / 'solution_tta_v46_300ep.csv',
    'v54': SOLUTIONS / 'solution_tta_v54_250ep.csv',
    'v55': SOLUTIONS / 'solution_tta_v55_250ep.csv',
    'v56': SOLUTIONS / 'solution_tta_v56_250ep.csv',
}

dfs = {k: pd.read_csv(p, index_col='id') for k, p in paths.items()}
print(f'Loaded {len(dfs)} CSVs, all with {len(next(iter(dfs.values())))} rows')

# Verify column alignment
cols0 = list(dfs['v46'].columns)
for k, df in dfs.items():
    assert list(df.columns) == cols0, f'{k} column mismatch'
    assert df.index.equals(dfs['v46'].index), f'{k} index mismatch'

ensemble = sum(dfs.values()) / len(dfs)
out = SOLUTIONS / 'solution_tta_ensemble_v46_v54_v55_v56.csv'
ensemble.to_csv(out)
print(f'Saved: {out}  ({len(ensemble)} rows)')
