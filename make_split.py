"""
Create an 80/20 train/val split of the training data by file index.
Writes train_files.txt and val_files.txt — lists of absolute paths.
Models can load these instead of scanning a directory to get a clean split.
"""
import os, random
from pathlib import Path

TRAIN = r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\train\train'
OUT   = Path(r'C:\Users\jan\downloads\nml_project_proposal\nml_project_proposal')

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
