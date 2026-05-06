"""Sequential 35-epoch exploration runner. Runs v22, v27, v28, v29 one after another."""
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
BASE = Path(__file__).parent
MODELS = BASE / 'models'

runs = [
    (MODELS / 'nba_v22.py', 35),   # G: decoder self-attention
    (MODELS / 'nba_v27.py', 35),   # D: large capacity (state_dim=256, 3 layers)
    (MODELS / 'nba_v28.py', 35),   # H: acceleration features
    (MODELS / 'nba_v29.py', 35),   # K: context_size=16
]

for model_path, epochs in runs:
    print(f'\n{"="*60}')
    print(f'STARTING: {model_path.name} ({epochs} epochs)')
    print(f'{"="*60}\n')
    result = subprocess.run(
        [PYTHON, str(BASE / 'run_one.py'), str(model_path), str(epochs), '32'],
        check=False,
    )
    print(f'\nDONE: {model_path.name} — exit code {result.returncode}')

print('\nAll exploration runs complete.')
