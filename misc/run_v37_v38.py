import subprocess, sys
from pathlib import Path
BASE = Path(r"C:\Users\jan\downloads\nml_project_proposal")
PYTHON = str(BASE / ".venv" / "Scripts" / "python.exe")
RUNNER = str(BASE / "nml_project_proposal" / "run_one.py")
MODELS = BASE / "nml_project_proposal" / "models"
for fname, epochs, bs in [("nba_v37.py", 35, 32), ("nba_v38.py", 35, 32)]:
    model_path = str(MODELS / fname)
    cmd = [PYTHON, RUNNER, model_path, str(epochs), str(bs)]
    print(f"=== Starting {fname} ({epochs} epochs) ===", flush=True)
    result = subprocess.run(cmd, cwd=str(BASE))
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"=== {fname} finished: {status} ===", flush=True)
print("v37+v38 queue done.")
