# NBA Trajectory Prediction (nml-2026 Kaggle / EPFL EE-452)

Trajectory prediction over 11 entities (10 players + 1 ball). Model takes 8 context frames, predicts 12 future positions.

## Memory (Claude auto-memory)

Persistent context for this project lives at:
```
C:\Users\jan\.claude\projects\C--Users-jan-Downloads-nml-project-proposal\memory\
```

Key files:
- `MEMORY.md` — index
- `next_steps.md` — current experiment queue and contingency plans
- `findings.md` — key architectural insights (MoG breakthrough, late bloomer pattern, etc.)
- `project_state.md` — current best scores and what's been tried
- `user_prefs.md` — how the user likes to work
- `feedback_commits.md` — no Co-Authored-By Claude line in commits

Read these at session start.

## Project layout

- `models/nba_vXX.py` — model versions (each is self-contained: Dataset, Model, Trainer)
- `models/weights/` — saved checkpoints (`model_vXX_<epochs>ep.pth` for tagged runs)
- `models/logs/` — training logs (`<tag>_train.log` / `<tag>_err.log`)
- `models/solutions/` — Kaggle submission CSVs
- `misc/make_tta_*.py` — TTA inference scripts (4-variant: orig, x-flip, y-flip, xy-flip averaged)
- `results.md` — append-only results table (auto-appended raw rows from evaluate.py need manual cleanup)
- `run_one.py` — runs a model file: `run_one.py <model.py> <epochs> <batch>`; saves both `model_vXX.pth` and epoch-tagged `model_vXX_NNep.pth`
- `evaluate.py` — computes local score, auto-appends to results.md

## How to launch a training run (PowerShell)

```powershell
$proj = "C:\Users\jan\Downloads\nml_project_proposal\nml_project_proposal"
$py = "C:\Users\jan\Downloads\nml_project_proposal\.venv\Scripts\python.exe"

Start-Process -WindowStyle Hidden -FilePath $py `
  -ArgumentList "$proj\run_one.py $proj\models\nba_vXX.py EPOCHS 32" `
  -RedirectStandardOutput "$proj\models\logs\<tag>_train.log" `
  -RedirectStandardError  "$proj\models\logs\<tag>_err.log" `
  -WorkingDirectory $proj
```

GPU can handle multiple concurrent training jobs.

## Conventions

- Model interface: `model(X)` in eval mode returns `[T, B*N, 2]` normalized positions; `evaluate.py` denormalizes and computes MSE
- Loss schedule: `CosineAnnealingLR` with `T_max=epochs` — must match exact epoch count
- Bold threshold in results.md: only bold rows that beat the current best
- Local→Kaggle gap: ~7% (stable)

## Git

- Tool: `C:\Program Files\Git\cmd\git.exe`
- **NEVER include `Co-Authored-By: Claude` line in commit messages** — user instruction
- Always `git push` after `git commit`
- Remote: https://github.com/ri-ru/nml.git (branch: main)
