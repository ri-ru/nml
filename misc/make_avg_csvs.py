"""
CPU-only equal-weight averaging of existing TTA solution CSVs.
Produces several candidate ensemble submissions ready to upload to Kaggle.

Run: python misc/make_avg_csvs.py
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
SOL = BASE / 'models' / 'solutions'

# Current best (3.01420 on Kaggle) is the top5 TTA ensemble of v46+v58+v46_s1+v46_s3+v46_s3_swa.
# Available solo TTA CSVs (with their Kaggle scores):
#   solution_tta_v44_250ep.csv  -> 3.11593
#   solution_tta_v45_300ep.csv  -> 3.24342
#   solution_tta_v46_300ep.csv  -> 3.03702  (best solo)
#   solution_tta_v54_250ep.csv  -> 3.10415
#   solution_tta_v55_250ep.csv  -> 3.12138
#   solution_tta_v56_250ep.csv  -> 3.13162
#   solution_tta_ensemble_top5_v46_v58_v46s1_v46s3_v46s3swa.csv -> 3.01420 (current best ensemble)

COMBOS = {
    # Current best + 3 newer v46-line models; cheapest way to add diversity to the winning recipe.
    'avg_top5_v54_v55_v56.csv': [
        'solution_tta_ensemble_top5_v46_v58_v46s1_v46s3_v46s3swa.csv',
        'solution_tta_v54_250ep.csv',
        'solution_tta_v55_250ep.csv',
        'solution_tta_v56_250ep.csv',
    ],
    # Solo mix of the 4 best non-v58 models; tests whether ensemble *without* v58 generalizes.
    'avg_v46_v54_v55_v56.csv': [
        'solution_tta_v46_300ep.csv',
        'solution_tta_v54_250ep.csv',
        'solution_tta_v55_250ep.csv',
        'solution_tta_v56_250ep.csv',
    ],
    # Current best + 4 strong solos; broadest weighting of the v46 line in particular.
    'avg_top5_v46_v54_v55_v56.csv': [
        'solution_tta_ensemble_top5_v46_v58_v46s1_v46s3_v46s3swa.csv',
        'solution_tta_v46_300ep.csv',
        'solution_tta_v54_250ep.csv',
        'solution_tta_v55_250ep.csv',
        'solution_tta_v56_250ep.csv',
    ],
    # Kitchen sink: best + every halfway-decent solo we have.
    'avg_kitchen_sink.csv': [
        'solution_tta_ensemble_top5_v46_v58_v46s1_v46s3_v46s3swa.csv',
        'solution_tta_v44_250ep.csv',
        'solution_tta_v46_300ep.csv',
        'solution_tta_v54_250ep.csv',
        'solution_tta_v55_250ep.csv',
        'solution_tta_v56_250ep.csv',
    ],
}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).set_index('id').sort_index()
    return df


def main() -> None:
    for out_name, files in COMBOS.items():
        print(f'--- {out_name} ({len(files)} sources) ---')
        dfs = []
        for f in files:
            p = SOL / f
            if not p.exists():
                print(f'  MISSING: {f} — skipping this combo')
                dfs = None
                break
            df = load_csv(p)
            dfs.append(df)
            print(f'  loaded {f}  shape={df.shape}')
        if dfs is None:
            continue

        # Sanity: all share id index + columns
        idx = dfs[0].index
        cols = list(dfs[0].columns)
        for df in dfs[1:]:
            assert (df.index == idx).all(), 'id mismatch'
            assert list(df.columns) == cols, 'column mismatch'

        avg = sum(dfs) / len(dfs)
        out = SOL / out_name
        avg.to_csv(out)
        print(f'  -> {out}\n')


if __name__ == '__main__':
    main()
