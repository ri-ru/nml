"""
Create an 80/20 train/val split of the training data by file index. 
"""
import os, random
from pathlib import Path

OUT  = Path(__file__).resolve().parent.parent   # repo root: writes train_files.txt / val_files.txt
DATA = Path(os.environ.get('NML_DATA') or OUT / 'data' / 'nml-2026')
if not (DATA / 'train' / 'train').exists():
    raise SystemExit(f'Dataset not found at {DATA}. Set NML_DATA to the dataset root or place it '
                     'at ./data/nml-2026 (needs train/train/*.pt and test/test/*.pt).')
TRAIN = str(DATA / 'train' / 'train')

files = sorted([os.path.join(TRAIN, f) for f in os.listdir(TRAIN) if f.endswith('.pt')],
               key=lambda p: int(Path(p).stem))

random.seed(42)
random.shuffle(files)

split = int(0.8 * len(files))
train_files = files[:split]
val_files   = files[split:]

(OUT / 'train_files.txt').write_text('\n'.join(train_files))
(OUT / 'val_files.txt').write_text('\n'.join(val_files))

print(f'Total: {len(files)} | Train: {len(train_files)} | Val: {len(val_files)}')
print(f'Saved train_files.txt and val_files.txt to {OUT}')
