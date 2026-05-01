# Experiment Agent Design

**Date:** 2026-05-01  
**Project:** NBA trajectory prediction (NML project)  
**Goal:** Build an AI agent that autonomously iterates on ML model architectures until validation MSE ≤ 1.0

---

## Overview

A single Python script (`agent.py`) loops indefinitely, using the Claude API to propose and implement new model architectures, train them, evaluate them, and repeat — stopping when the score hits the target or the iteration cap is reached.

---

## Loop

Each iteration:

1. Read `results.md` + all `models/nba_*.py` files
2. Call Claude API with full history → receive JSON with new model code + config
3. Write `models/nba_vN.py` (N = auto-incremented by scanning existing `models/nba_v*.py` filenames)
4. Instantiate the new `Trainer`, call `.train()` — saves weights to `models/weights/`
5. Evaluate with `evaluate.py` → get validation score
6. Append row to `results.md` (with Claude's hypothesis in the Notes column)
7. Log full iteration detail to `agent_log.md`
8. If score ≤ 1.0 → stop. Else → repeat.

**Stop conditions:**
- Score ≤ 1.0 (target reached)
- 20 iterations completed (hard cap)

---

## Claude API Prompt

**System prompt** (sent once, not per iteration):

> You are an ML researcher working on NBA player trajectory prediction. Your goal is to minimize validation MSE. Each iteration you will see the full experiment history and all prior model code. You must propose a new architecture, explain your hypothesis in 1–2 sentences, and return a complete Python model file following the Trainer convention. Respond with valid JSON only.

**User message** (rebuilt each iteration):

```
Current results:
<contents of results.md>

Prior model code:
<nba_v1.py: ...>
<nba_v2.py: ...>
...

Write a new model. Return JSON:
{
  "filename": "nba_vN.py",
  "hypothesis": "one or two sentences explaining what you're trying and why",
  "code": "full Python file contents",
  "epochs": N,
  "batch_size": N,
  "notes": "any extra notes for results.md"
}
```

**Model file convention:** Each generated file must export a `Trainer` class with:
- `__init__(self, train_path, val_path, device, epochs, batch_size, ...)` 
- `.train()` that saves weights to `models/weights/model_<name>.pth`
- `.mu`, `.sigma`, `.val_dataset` attributes (required by `evaluate.py`)

Claude may write the dataset, model, and trainer however it wants — full creative control — as long as it follows this interface.

---

## Error Handling

- **Invalid JSON response:** log error to `agent_log.md`, include the error in the next prompt so Claude can self-correct, retry
- **Syntax error in generated code:** catch `SyntaxError` on import, log, continue to next iteration
- **Training crash (OOM, runtime error):** catch exception, log with traceback, continue
- **No improvement for 5 straight iterations:** include this signal in the prompt so Claude knows to try something more different

---

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Main loop script (project root) |
| `agent_log.md` | Verbose per-iteration log (hypotheses, errors, decisions) |
| `results.md` | Clean score table (unchanged format, Claude's hypothesis goes in Notes) |
| `models/nba_vN.py` | Each generated model file, kept permanently |

---

## CLI

```bash
ANTHROPIC_API_KEY=sk-... python agent.py \
  --train-path /path/to/train \
  --val-path /path/to/val \
  --target-score 1.0 \
  --device cuda
```

Optional flags:
- `--max-iters 20` (default: 20)
- `--model claude-opus-4-7` (default: claude-opus-4-7 for best reasoning)

---

## Console Output

```
[Iter 1] Hypothesis: add temporal attention between timesteps
[Iter 1] Training nba_v3.py for 100 epochs...
[Iter 1] Score: 7.83 (target: 1.0) — continuing

[Iter 2] Hypothesis: use graph convolution with team-aware edges
[Iter 2] Training nba_v4.py for 150 epochs...
[Iter 2] Score: 5.21 (target: 1.0) — continuing
...
[Iter N] Score: 0.94 (target: 1.0) — TARGET REACHED. Stopping.
```

---

## Out of Scope

- Notebook cell manipulation
- Parallel training runs
- wandb sweep integration (agent replaces this)
- Human-in-the-loop approval before each experiment
