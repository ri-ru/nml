# Experiment Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `agent.py` — a script that uses Claude API to autonomously propose, implement, train, and evaluate new NBA trajectory model architectures until validation MSE ≤ 1.0.

**Architecture:** Single loop script at project root. Each iteration reads `results.md` + all `models/nba_v*.py` files, calls Claude API to get a new model file as JSON, writes it to `models/`, trains it, evaluates it with `evaluate.py`, logs to `agent_log.md`, and repeats. Stops when score ≤ target or iteration cap reached.

**Tech Stack:** Python 3.10+, `anthropic` SDK, PyTorch, existing `evaluate.py`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `agent.py` | Create | Main loop script — all agent logic lives here |
| `tests/test_agent.py` | Create | Unit tests for all pure functions in agent.py |
| `agent_log.md` | Auto-created at runtime | Verbose per-iteration log (hypothesis, score, errors) |

Generated model files (`models/nba_v3.py`, `models/nba_v4.py`, ...) are created at runtime by the agent.

---

## Trainer Interface Contract

Every file the agent generates **must** follow this interface (enforced via system prompt):

```python
class Trainer:
    mu: torch.Tensor        # shape [2], position mean
    sigma: torch.Tensor     # shape [2], position std
    device: str             # e.g. 'cuda' or 'cpu'
    val_dataset: Dataset    # must have .max_start: list[int]
    model: nn.Module        # set by .train() before returning

    def __init__(self, train_path, val_path, device, epochs, batch_size, **kwargs): ...
    def train(self): ...    # saves weights AND sets self.model
```

---

## Task 1: Install dependency and create skeleton files

**Files:**
- Create: `agent.py`
- Create: `tests/__init__.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Install anthropic**

```bash
pip install anthropic
python -c "import anthropic; print(anthropic.__version__)"
```
Expected: a version string like `0.40.0`

- [ ] **Step 2: Create `tests/__init__.py`** (empty file)

- [ ] **Step 3: Create `agent.py` skeleton**

```python
import argparse
import importlib.util
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import anthropic

MODELS_DIR = Path(__file__).parent / "models"
RESULTS_FILE = MODELS_DIR / "results.md"
ROOT_DIR = Path(__file__).parent


def next_version(models_dir: Path) -> int:
    raise NotImplementedError


def build_context(models_dir: Path, results_file: Path) -> str:
    raise NotImplementedError


def parse_claude_response(text: str) -> dict:
    raise NotImplementedError


def write_model_file(models_dir: Path, filename: str, code: str) -> Path:
    raise NotImplementedError


def load_and_train(model_path: Path, train_path: str, val_path: str,
                   device: str, epochs: int, batch_size: int):
    raise NotImplementedError


def run_evaluation(trainer, version: int) -> float:
    raise NotImplementedError


def append_agent_log(log_file: Path, version: int, hypothesis: str,
                     score: float | None, error: str | None) -> None:
    raise NotImplementedError


def call_claude(client: anthropic.Anthropic, context: str,
                version: int, model: str) -> dict:
    raise NotImplementedError


def agent_loop(client: anthropic.Anthropic, train_path: str, val_path: str,
               device: str, target_score: float, max_iters: int,
               claude_model: str) -> None:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `tests/test_agent.py` skeleton**

```python
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import agent
```

- [ ] **Step 5: Verify imports work**

```bash
cd /mnt/c/Users/jan/Downloads/nml_project_proposal/nml_project_proposal
python -c "import agent; print('ok')"
```
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/__init__.py tests/test_agent.py
git commit -m "feat: scaffold agent.py and test structure"
```

---

## Task 2: Implement `next_version()`

**Files:**
- Modify: `agent.py` — implement `next_version()`
- Modify: `tests/test_agent.py` — add tests

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
import tempfile
import os

def test_next_version_with_existing_files(tmp_path):
    (tmp_path / "nba_v2.py").touch()
    (tmp_path / "nba_v3.py").touch()
    assert agent.next_version(tmp_path) == 4

def test_next_version_no_versioned_files(tmp_path):
    (tmp_path / "nba_baseline.py").touch()
    (tmp_path / "nba_gat.py").touch()
    assert agent.next_version(tmp_path) == 3

def test_next_version_empty_dir(tmp_path):
    assert agent.next_version(tmp_path) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py::test_next_version_with_existing_files -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `next_version()`**

Replace the stub in `agent.py`:

```python
def next_version(models_dir: Path) -> int:
    versions = []
    for f in models_dir.glob("nba_v*.py"):
        m = re.match(r"nba_v(\d+)\.py", f.name)
        if m:
            versions.append(int(m.group(1)))
    return max(versions, default=2) + 1
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -k "next_version" -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: implement next_version()"
```

---

## Task 3: Implement `build_context()`

**Files:**
- Modify: `agent.py` — implement `build_context()`
- Modify: `tests/test_agent.py` — add tests

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_build_context_includes_results_and_code(tmp_path):
    results = tmp_path / "results.md"
    results.write_text("| score | model |\n| 9.26 | v2 |")
    (tmp_path / "nba_v2.py").write_text("class NBAv2Trainer: pass")

    ctx = agent.build_context(tmp_path, results)

    assert "9.26" in ctx
    assert "nba_v2.py" in ctx
    assert "NBAv2Trainer" in ctx

def test_build_context_missing_results(tmp_path):
    ctx = agent.build_context(tmp_path, tmp_path / "results.md")
    assert "(no results yet)" in ctx

def test_build_context_no_model_files(tmp_path):
    (tmp_path / "results.md").write_text("| score |")
    ctx = agent.build_context(tmp_path, tmp_path / "results.md")
    assert "score" in ctx
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -k "build_context" -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `build_context()`**

```python
def build_context(models_dir: Path, results_file: Path) -> str:
    results = results_file.read_text() if results_file.exists() else "(no results yet)"
    model_files = sorted(models_dir.glob("nba_v*.py"))
    code_sections = "\n\n".join(
        f"=== {f.name} ===\n{f.read_text()}" for f in model_files
    )
    return (
        f"Current results:\n{results}\n\n"
        f"Prior model code:\n{code_sections if code_sections else '(none yet)'}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -k "build_context" -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: implement build_context()"
```

---

## Task 4: Implement `parse_claude_response()`

**Files:**
- Modify: `agent.py` — implement `parse_claude_response()`
- Modify: `tests/test_agent.py` — add tests

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_parse_claude_response_valid():
    payload = {
        "filename": "nba_v3.py",
        "hypothesis": "Add dropout",
        "code": "class Trainer: pass",
        "epochs": 50,
        "batch_size": 32,
        "notes": "test run"
    }
    result = agent.parse_claude_response(json.dumps(payload))
    assert result["filename"] == "nba_v3.py"
    assert result["epochs"] == 50

def test_parse_claude_response_missing_field():
    payload = {"filename": "nba_v3.py"}  # missing required fields
    with pytest.raises(ValueError, match="Missing required field"):
        agent.parse_claude_response(json.dumps(payload))

def test_parse_claude_response_invalid_json():
    with pytest.raises(ValueError, match="Invalid JSON"):
        agent.parse_claude_response("not json at all {{{")

def test_parse_claude_response_strips_markdown_fences():
    payload = {"filename": "nba_v3.py", "hypothesis": "x", "code": "y",
               "epochs": 10, "batch_size": 16, "notes": "z"}
    wrapped = f"```json\n{json.dumps(payload)}\n```"
    result = agent.parse_claude_response(wrapped)
    assert result["filename"] == "nba_v3.py"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -k "parse_claude" -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `parse_claude_response()`**

```python
REQUIRED_FIELDS = {"filename", "hypothesis", "code", "epochs", "batch_size", "notes"}

def parse_claude_response(text: str) -> dict:
    text = text.strip()
    # strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Claude: {e}") from e
    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"Missing required field(s): {missing}")
    return data
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -k "parse_claude" -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: implement parse_claude_response()"
```

---

## Task 5: Implement `write_model_file()`

**Files:**
- Modify: `agent.py` — implement `write_model_file()`
- Modify: `tests/test_agent.py` — add tests

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_write_model_file_creates_file(tmp_path):
    code = "class Trainer:\n    pass\n"
    path = agent.write_model_file(tmp_path, "nba_v3.py", code)
    assert path == tmp_path / "nba_v3.py"
    assert path.read_text() == code

def test_write_model_file_returns_path(tmp_path):
    path = agent.write_model_file(tmp_path, "nba_v5.py", "x = 1")
    assert path.name == "nba_v5.py"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -k "write_model" -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `write_model_file()`**

```python
def write_model_file(models_dir: Path, filename: str, code: str) -> Path:
    path = models_dir / filename
    path.write_text(code)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -k "write_model" -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: implement write_model_file()"
```

---

## Task 6: Implement `load_and_train()`

**Files:**
- Modify: `agent.py` — implement `load_and_train()`
- Modify: `tests/test_agent.py` — add tests

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_load_and_train_returns_trainer(tmp_path):
    import torch
    trainer_code = '''
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class FakeDataset(Dataset):
    def __init__(self):
        self.max_start = [0, 0]
    def __len__(self): return 2
    def __getitem__(self, idx): return torch.zeros(8, 11, 4), torch.zeros(12, 11, 4)

class Trainer:
    def __init__(self, train_path, val_path, device, epochs, batch_size, **kwargs):
        self.mu = torch.zeros(2)
        self.sigma = torch.ones(2)
        self.device = device
        self.val_dataset = FakeDataset()
        self.model = None

    def train(self):
        self.model = nn.Linear(1, 1)
'''
    model_path = tmp_path / "nba_v99.py"
    model_path.write_text(trainer_code)

    trainer = agent.load_and_train(model_path, "fake/train", "fake/val", "cpu", 1, 2)
    assert trainer.model is not None
    assert hasattr(trainer, "mu")
    assert hasattr(trainer, "val_dataset")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py::test_load_and_train_returns_trainer -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `load_and_train()`**

```python
def load_and_train(model_path: Path, train_path: str, val_path: str,
                   device: str, epochs: int, batch_size: int):
    spec = importlib.util.spec_from_file_location("agent_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer_names = [n for n in dir(module) if "Trainer" in n and not n.startswith("_")]
    if not trainer_names:
        raise ImportError(f"No Trainer class found in {model_path.name}")
    TrainerClass = getattr(module, trainer_names[0])
    trainer = TrainerClass(
        train_path=train_path, val_path=val_path,
        device=device, epochs=epochs, batch_size=batch_size,
    )
    trainer.train()
    return trainer
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py::test_load_and_train_returns_trainer -v
```
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: implement load_and_train()"
```

---

## Task 7: Implement `run_evaluation()`

**Files:**
- Modify: `agent.py` — implement `run_evaluation()`
- Modify: `tests/test_agent.py` — add tests

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_run_evaluation_returns_float():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset

    class FakeDataset(Dataset):
        def __init__(self):
            self.max_start = [0]
        def __len__(self): return 1
        def __getitem__(self, idx): return torch.zeros(8, 11, 4), torch.zeros(12, 11, 4)

    class FakeModel(nn.Module):
        def forward(self, x):
            B, _, N, _ = x.shape
            return torch.zeros(12, B * N, 2)

    class FakeTrainer:
        mu = torch.zeros(2)
        sigma = torch.ones(2)
        device = "cpu"
        val_dataset = FakeDataset()
        model = FakeModel()

    with patch("agent.evaluate_model") as mock_eval, \
         patch("agent.make_val_dl") as mock_dl:
        mock_dl.return_value = []
        mock_eval.return_value = 7.5
        score = agent.run_evaluation(FakeTrainer(), 3)

    assert score == 7.5
    mock_eval.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py::test_run_evaluation_returns_float -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Add imports to `agent.py` for evaluate functions**

Add near the top of `agent.py`, after the existing imports:

```python
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import evaluate_model, make_val_dl
```

- [ ] **Step 4: Implement `run_evaluation()`**

```python
def run_evaluation(trainer, version: int) -> float:
    val_dl = make_val_dl(trainer)
    name = f"agent_v{version}"
    score = evaluate_model(
        trainer.model, trainer, val_dl, name=name,
        notes=f"generated by agent (iter {version - 2})"
    )
    return score
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_agent.py::test_run_evaluation_returns_float -v
```
Expected: PASSED

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: implement run_evaluation()"
```

---

## Task 8: Implement `append_agent_log()`

**Files:**
- Modify: `agent.py` — implement `append_agent_log()`
- Modify: `tests/test_agent.py` — add tests

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_append_agent_log_success(tmp_path):
    log_file = tmp_path / "agent_log.md"
    agent.append_agent_log(log_file, version=3, hypothesis="Try residual connections",
                           score=6.42, error=None)
    content = log_file.read_text()
    assert "nba_v3.py" in content
    assert "Try residual connections" in content
    assert "6.4200" in content

def test_append_agent_log_error(tmp_path):
    log_file = tmp_path / "agent_log.md"
    agent.append_agent_log(log_file, version=4, hypothesis="x", score=None, error="OOM")
    content = log_file.read_text()
    assert "OOM" in content
    assert "Score" not in content

def test_append_agent_log_appends(tmp_path):
    log_file = tmp_path / "agent_log.md"
    agent.append_agent_log(log_file, version=3, hypothesis="first", score=5.0, error=None)
    agent.append_agent_log(log_file, version=4, hypothesis="second", score=4.0, error=None)
    content = log_file.read_text()
    assert "first" in content and "second" in content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -k "append_agent_log" -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `append_agent_log()`**

```python
def append_agent_log(log_file: Path, version: int, hypothesis: str,
                     score: float | None, error: str | None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n## [{timestamp}] nba_v{version}.py\n",
             f"**Hypothesis:** {hypothesis}\n"]
    if score is not None:
        lines.append(f"**Score:** {score:.4f}\n")
    if error:
        lines.append(f"**Error:** {error}\n")
    with open(log_file, "a") as f:
        f.writelines(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -k "append_agent_log" -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: implement append_agent_log()"
```

---

## Task 9: Implement `call_claude()`

**Files:**
- Modify: `agent.py` — implement `call_claude()`
- Modify: `tests/test_agent.py` — add tests (mocked API)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_call_claude_returns_parsed_dict():
    payload = {
        "filename": "nba_v3.py",
        "hypothesis": "Use transformer encoder",
        "code": "class Trainer: pass",
        "epochs": 80,
        "batch_size": 32,
        "notes": "first transformer attempt"
    }
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=json.dumps(payload))]
    )
    result = agent.call_claude(mock_client, "context text", version=3, model="claude-opus-4-7")
    assert result["filename"] == "nba_v3.py"
    assert result["epochs"] == 80
    mock_client.messages.create.assert_called_once()

def test_call_claude_passes_version_in_prompt():
    payload = {"filename": "nba_v5.py", "hypothesis": "x", "code": "y",
               "epochs": 10, "batch_size": 16, "notes": "z"}
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=json.dumps(payload))]
    )
    agent.call_claude(mock_client, "ctx", version=5, model="claude-opus-4-7")
    call_kwargs = mock_client.messages.create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    user_content = str(call_kwargs)
    assert "nba_v5.py" in user_content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -k "call_claude" -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `call_claude()`**

```python
SYSTEM_PROMPT = """\
You are an ML researcher working on NBA player trajectory prediction (forecasting x,y positions
of 11 players over 12 future timesteps given 8 context timesteps).

Your goal: minimize validation MSE. Each iteration you see the full experiment history and all
prior model code. You must propose a genuinely new architecture, explain your hypothesis in 1-2
sentences, and return a complete Python model file.

REQUIRED Trainer interface — your Trainer class MUST have:
  __init__(self, train_path, val_path, device, epochs, batch_size, **kwargs)
  train(self)   <- must set self.model before returning, must save weights to models/weights/
  self.mu       <- torch.Tensor shape [2], position mean
  self.sigma    <- torch.Tensor shape [2], position std
  self.device   <- str
  self.val_dataset  <- Dataset with .max_start attribute (list of ints)
  self.model    <- nn.Module set by train()

Do NOT use wandb (it causes issues in agent mode). Save weights with:
  torch.save(self.model.state_dict(), Path(__file__).parent / 'weights' / 'model_vN.pth')

Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:
{
  "filename": "nba_vN.py",
  "hypothesis": "1-2 sentences on what you're trying and why",
  "code": "complete Python file as a string",
  "epochs": <int>,
  "batch_size": <int>,
  "notes": "brief notes for results table"
}
"""


def call_claude(client: anthropic.Anthropic, context: str,
                version: int, model: str) -> dict:
    user_message = (
        f"{context}\n\n"
        f"Write a new model file named nba_v{version}.py. "
        f"Return JSON with filename, hypothesis, code, epochs, batch_size, notes."
    )
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return parse_claude_response(response.content[0].text)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -k "call_claude" -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: implement call_claude() with system prompt"
```

---

## Task 10: Implement `agent_loop()` with error handling and stagnation detection

**Files:**
- Modify: `agent.py` — implement `agent_loop()`
- Modify: `tests/test_agent.py` — add tests

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`:

```python
def test_agent_loop_stops_at_target(tmp_path):
    with patch("agent.MODELS_DIR", tmp_path / "models"), \
         patch("agent.ROOT_DIR", tmp_path), \
         patch("agent.next_version", return_value=3), \
         patch("agent.build_context", return_value="ctx"), \
         patch("agent.call_claude", return_value={
             "filename": "nba_v3.py", "hypothesis": "test",
             "code": "class Trainer: pass", "epochs": 1,
             "batch_size": 1, "notes": "x"
         }), \
         patch("agent.write_model_file", return_value=tmp_path / "nba_v3.py"), \
         patch("agent.load_and_train", return_value=MagicMock()), \
         patch("agent.run_evaluation", return_value=0.8), \
         patch("agent.append_agent_log"):
        client = MagicMock()
        agent.agent_loop(client, "train", "val", "cpu",
                         target_score=1.0, max_iters=10, claude_model="claude-opus-4-7")
        # should stop after 1 iteration since 0.8 < 1.0

def test_agent_loop_stops_at_max_iters(tmp_path):
    with patch("agent.MODELS_DIR", tmp_path / "models"), \
         patch("agent.ROOT_DIR", tmp_path), \
         patch("agent.next_version", return_value=3), \
         patch("agent.build_context", return_value="ctx"), \
         patch("agent.call_claude", return_value={
             "filename": "nba_v3.py", "hypothesis": "h",
             "code": "class Trainer: pass", "epochs": 1,
             "batch_size": 1, "notes": "n"
         }), \
         patch("agent.write_model_file", return_value=tmp_path / "nba_v3.py"), \
         patch("agent.load_and_train", return_value=MagicMock()), \
         patch("agent.run_evaluation", return_value=99.0), \
         patch("agent.append_agent_log") as mock_log:
        client = MagicMock()
        agent.agent_loop(client, "train", "val", "cpu",
                         target_score=1.0, max_iters=3, claude_model="claude-opus-4-7")
        assert mock_log.call_count == 3

def test_agent_loop_handles_training_error(tmp_path):
    with patch("agent.MODELS_DIR", tmp_path / "models"), \
         patch("agent.ROOT_DIR", tmp_path), \
         patch("agent.next_version", side_effect=[3, 4]), \
         patch("agent.build_context", return_value="ctx"), \
         patch("agent.call_claude", return_value={
             "filename": "nba_v3.py", "hypothesis": "h",
             "code": "class Trainer: pass", "epochs": 1,
             "batch_size": 1, "notes": "n"
         }), \
         patch("agent.write_model_file", return_value=tmp_path / "nba_v3.py"), \
         patch("agent.load_and_train", side_effect=[RuntimeError("OOM"), MagicMock()]), \
         patch("agent.run_evaluation", return_value=0.5), \
         patch("agent.append_agent_log"):
        client = MagicMock()
        # should not raise — error in iter 1, success in iter 2
        agent.agent_loop(client, "train", "val", "cpu",
                         target_score=1.0, max_iters=2, claude_model="claude-opus-4-7")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -k "agent_loop" -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `agent_loop()`**

```python
def agent_loop(client: anthropic.Anthropic, train_path: str, val_path: str,
               device: str, target_score: float, max_iters: int,
               claude_model: str) -> None:
    log_file = ROOT_DIR / "agent_log.md"
    scores: list[float] = []
    extra_context = ""

    for i in range(max_iters):
        version = next_version(MODELS_DIR)
        print(f"\n[Iter {i + 1}/{max_iters}] Planning nba_v{version}.py ...")

        context = build_context(MODELS_DIR, RESULTS_FILE) + extra_context

        try:
            response = call_claude(client, context, version, claude_model)
        except (ValueError, Exception) as e:
            err = f"Claude API / parse error: {e}"
            print(f"  ERROR: {err}")
            append_agent_log(log_file, version, "(no hypothesis)", None, err)
            extra_context = f"\n\nNote: previous attempt failed with: {err}. Please fix."
            continue

        hypothesis = response["hypothesis"]
        print(f"  Hypothesis: {hypothesis}")

        try:
            model_path = write_model_file(MODELS_DIR, response["filename"], response["code"])
            print(f"  Training {response['filename']} for {response['epochs']} epochs ...")
            trainer = load_and_train(
                model_path, train_path, val_path, device,
                response["epochs"], response["batch_size"],
            )
            score = run_evaluation(trainer, version)
        except Exception as e:
            err = traceback.format_exc()
            print(f"  ERROR during training/eval: {e}")
            append_agent_log(log_file, version, hypothesis, None, err)
            extra_context = f"\n\nNote: previous model nba_v{version}.py crashed: {e}. Fix this."
            continue

        scores.append(score)
        append_agent_log(log_file, version, hypothesis, score, None)
        print(f"  Score: {score:.4f} (target: {target_score}) — "
              f"{'TARGET REACHED. Stopping.' if score <= target_score else 'continuing'}")

        if score <= target_score:
            break

        # stagnation signal: no improvement in last 5 iters
        extra_context = ""
        if len(scores) >= 5 and min(scores[-5:]) >= min(scores[:-5] or [float("inf")]):
            extra_context = (
                "\n\nNote: scores have not improved in the last 5 iterations "
                f"(recent: {[f'{s:.2f}' for s in scores[-5:]]}). "
                "Try a fundamentally different architecture."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -k "agent_loop" -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: implement agent_loop() with error handling and stagnation detection"
```

---

## Task 11: Implement `main()` CLI entry point

**Files:**
- Modify: `agent.py` — implement `main()`
- Modify: `tests/test_agent.py` — add CLI test

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent.py`:

```python
def test_main_calls_agent_loop():
    with patch("agent.anthropic.Anthropic") as mock_anthropic, \
         patch("agent.agent_loop") as mock_loop:
        mock_anthropic.return_value = MagicMock()
        sys.argv = [
            "agent.py",
            "--train-path", "/data/train",
            "--val-path", "/data/val",
            "--target-score", "1.0",
            "--device", "cpu",
            "--max-iters", "5",
        ]
        agent.main()
        mock_loop.assert_called_once()
        call_kwargs = mock_loop.call_args.kwargs
        assert call_kwargs["train_path"] == "/data/train"
        assert call_kwargs["target_score"] == 1.0
        assert call_kwargs["max_iters"] == 5
        assert call_kwargs["device"] == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py::test_main_calls_agent_loop -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `main()`**

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="AI experiment agent for NBA trajectory models")
    parser.add_argument("--train-path", required=True, help="Path to training data directory")
    parser.add_argument("--val-path", required=True, help="Path to validation data directory")
    parser.add_argument("--target-score", type=float, default=1.0,
                        help="Stop when validation MSE drops to this value (default: 1.0)")
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda)")
    parser.add_argument("--max-iters", type=int, default=20,
                        help="Maximum number of iterations (default: 20)")
    parser.add_argument("--model", default="claude-opus-4-7",
                        help="Claude model ID (default: claude-opus-4-7)")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    print(f"Starting agent. Target MSE: {args.target_score} | Max iters: {args.max_iters}")
    print(f"Device: {args.device} | Claude model: {args.model}\n")

    agent_loop(
        client=client,
        train_path=args.train_path,
        val_path=args.val_path,
        device=args.device,
        target_score=args.target_score,
        max_iters=args.max_iters,
        claude_model=args.model,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py::test_main_calls_agent_loop -v
```
Expected: PASSED

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/test_agent.py -v
```
Expected: All tests PASSED

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: implement main() CLI entry point"
```

---

## Task 12: Smoke test

**Files:** No changes — validation only.

- [ ] **Step 1: Verify agent.py syntax and imports are clean**

```bash
python -c "import agent; print('all functions present:', [f for f in dir(agent) if not f.startswith('_')])"
```
Expected: prints list including `agent_loop`, `build_context`, `call_claude`, `load_and_train`, `main`, `next_version`, `parse_claude_response`, `run_evaluation`, `write_model_file`, `append_agent_log`

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/test_agent.py -v --tb=short
```
Expected: All tests PASSED, 0 errors

- [ ] **Step 3: Print the help message**

```bash
python agent.py --help
```
Expected: shows `--train-path`, `--val-path`, `--target-score`, `--device`, `--max-iters`, `--model`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: experiment agent complete — ready for use"
```

---

## Usage

```bash
ANTHROPIC_API_KEY=sk-... python agent.py \
  --train-path /path/to/train \
  --val-path /path/to/val \
  --target-score 1.0 \
  --device cuda
```
