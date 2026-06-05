"""Run training + evaluation for one generated model file. Called by the agent loop."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import evaluate_model, make_val_dl

TRAIN = r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\train\train'
VAL   = r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\train\train'


def _default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = _default_device()


def run(model_path: str, epochs: int = 30, batch_size: int = 32) -> float:
    path = Path(model_path)
    spec = importlib.util.spec_from_file_location("exp_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    trainer_names = [n for n in dir(module) if "Trainer" in n and not n.startswith("_")]
    if not trainer_names:
        raise ImportError(f"No Trainer class found in {path.name}")

    TrainerClass = getattr(module, trainer_names[0])
    trainer = TrainerClass(
        train_path=TRAIN, val_path=VAL,
        device=DEVICE, epochs=epochs, batch_size=batch_size,
    )
    trainer.train()

    # Save epoch-tagged copy — never overwrites previous checkpoints
    import torch
    weights_dir = path.parent / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    # e.g. models/weights/model_v43_250ep.pth
    version = path.stem.replace('nba_', '')  # nba_v43 -> v43
    tagged_path = weights_dir / f'model_{version}_{epochs}ep.pth'
    torch.save(trainer.model.state_dict(), tagged_path)
    print(f"Saved tagged weights: {tagged_path}")

    val_dl = make_val_dl(trainer)
    score = evaluate_model(
        trainer.model, trainer, val_dl,
        name=path.stem, notes=f"agent loop ({epochs} epochs)"
    )
    print(f"AGENT_SCORE:{score:.4f}")
    return score


if __name__ == "__main__":
    model_path = sys.argv[1]
    epochs     = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    run(model_path, epochs, batch_size)
