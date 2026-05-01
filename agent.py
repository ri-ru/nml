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
