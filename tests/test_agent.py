import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import agent


# ── Task 2: next_version ──────────────────────────────────────────────────────

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


# ── Task 3: build_context ─────────────────────────────────────────────────────

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


# ── Task 4: parse_claude_response ─────────────────────────────────────────────

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
    payload = {"filename": "nba_v3.py"}
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


# ── Task 5: write_model_file ──────────────────────────────────────────────────

def test_write_model_file_creates_file(tmp_path):
    code = "class Trainer:\n    pass\n"
    path = agent.write_model_file(tmp_path, "nba_v3.py", code)
    assert path == tmp_path / "nba_v3.py"
    assert path.read_text() == code

def test_write_model_file_returns_path(tmp_path):
    path = agent.write_model_file(tmp_path, "nba_v5.py", "x = 1")
    assert path.name == "nba_v5.py"


# ── Task 6: load_and_train ────────────────────────────────────────────────────

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


# ── Task 7: run_evaluation ────────────────────────────────────────────────────

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


# ── Task 8: append_agent_log ──────────────────────────────────────────────────

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


# ── Task 9: call_claude ───────────────────────────────────────────────────────

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
    user_content = str(mock_client.messages.create.call_args)
    assert "nba_v5.py" in user_content


# ── Task 10: agent_loop ───────────────────────────────────────────────────────

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
        agent.agent_loop(client, "train", "val", "cpu",
                         target_score=1.0, max_iters=2, claude_model="claude-opus-4-7")


# ── Task 11: main ─────────────────────────────────────────────────────────────

def test_main_calls_agent_loop():
    import sys as _sys
    with patch("agent.anthropic.Anthropic") as mock_anthropic, \
         patch("agent.agent_loop") as mock_loop:
        mock_anthropic.return_value = MagicMock()
        _sys.argv = [
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
