"""Run v33, v34, v35, v36 sequentially. Logs to run_queue.log."""
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
PYTHON = str(BASE / '.venv' / 'Scripts' / 'python.exe')
RUNNER = str(BASE / 'nml_project_proposal' / 'run_one.py')
MODELS = BASE / 'nml_project_proposal' / 'models'

queue = [
    ('nba_v33.py', 250, 32),
    ('nba_v34.py', 200, 32),
    ('nba_v35.py', 300, 32),
    ('nba_v36.py', 250, 32),
]

log = open(BASE / 'run_queue.log', 'w', buffering=1)

for fname, epochs, bs in queue:
    model_path = str(MODELS / fname)
    cmd = [PYTHON, RUNNER, model_path, str(epochs), str(bs)]
    print(f'\n=== Starting {fname} ({epochs} epochs) ===', flush=True)
    print(f'\n=== Starting {fname} ({epochs} epochs) ===', file=log, flush=True)
    result = subprocess.run(cmd, cwd=str(BASE), stdout=log, stderr=log)
    status = 'OK' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
    print(f'=== {fname} finished: {status} ===', flush=True)
    print(f'=== {fname} finished: {status} ===', file=log, flush=True)

log.close()
print('All done.')
