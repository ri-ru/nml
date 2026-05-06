"""
Ensemble TTA: average of v43x250ep TTA + v44x250ep TTA.
Both CSVs already generated. This just averages them.
Gaussian (v43) + MoG (v44) have different uncertainty representations
so their residual errors should be less correlated.
Output: solution_tta_ensemble_v43_v44_250ep.csv
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
SOLUTIONS = BASE / 'models' / 'solutions'

v43 = pd.read_csv(SOLUTIONS / 'solution_tta_v43_250ep.csv', index_col='id')
v44 = pd.read_csv(SOLUTIONS / 'solution_tta_v44_250ep.csv', index_col='id')

ensemble = (v43 + v44) / 2
out = SOLUTIONS / 'solution_tta_ensemble_v43_v44_250ep.csv'
ensemble.to_csv(out)
print(f'Saved: {out}  ({len(ensemble)} rows)')
