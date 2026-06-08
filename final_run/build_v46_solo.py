"""Single-model submission: just the retrained v46 (_full, 4-flip TTA), no ensemble.

Dumps final_run/test_preds/v46_full.pt (already TTA-averaged, real coords) straight to a Kaggle CSV
so we can see where one v46 model lands on its own vs the ensemble's 2.95815.
"""
from pathlib import Path
import pandas as pd, torch
from _config import TEST_DIR as TESTF, SOL_DIR as SOL
HOR, NE = 12, 11

P = torch.load(TESTF / 'v46_full.pt').numpy()   # [Ntest, 12, 11, 2]
Ntest = P.shape[0]
cols = [f'entity_{i}_time_{t}_{a}' for t in range(HOR) for i in range(NE) for a in ['x', 'y']]
rows = [[i] + P[i].reshape(-1).tolist() for i in range(Ntest)]
csv = SOL / 'solution_v46_solo.csv'
pd.DataFrame(rows, columns=['id'] + cols).set_index('id').sort_index().to_csv(csv)
print(f'{csv.name}: single retrained v46 (4-flip TTA), {Ntest} rows, no ensemble.')
