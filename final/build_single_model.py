"""Build a single-model Kaggle submission (the 2nd of our two final submissions).

The other final submission is the 6-model ensemble (final/build_ensemble_6model.py). This one is a
single model with no ensembling: load one 100%-data (_full) checkpoint, run 4-flip TTA on the test
set, write the Kaggle CSV. Default model is v46 (the canonical architecture trained 300 epochs on
100% of the data) -> reproduces outputs/solutions/solution_v46_full.csv.

It works for ANY member in the registry (same architecture, different checkpoint):
    python final/build_single_model.py            # v46    -> solution_v46_full.csv
    python final/build_single_model.py v60_w8     # v60_w8 -> solution_v60_w8_full.csv

Run:
    export NML_DATA=/path/to/nml-2026          # or place the dataset at ./data/nml-2026
    python final/build_single_model.py [label]
"""
import os
import sys
import importlib.util
from pathlib import Path

import torch
import pandas as pd

HERE = Path(__file__).resolve().parent
ARCH_DIR = HERE.parent / "demarche"   # single architecture source: demarche/nba_v46.py (no copy)
WEIGHTS_DIR = HERE.parent / "outputs" / "weights"
SOL_DIR = HERE.parent / "outputs" / "solutions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT, HORIZON, NE = 8, 12, 11

# Dataset root: NML_DATA env var, else the conventional ./data/nml-2026 in the repo.
DATA = Path(os.environ.get("NML_DATA") or HERE.parent / "data" / "nml-2026")
if not (DATA / "train" / "train").exists():
    raise SystemExit(
        f"Dataset not found at {DATA}. Set NML_DATA to the dataset root (a folder with "
        "train/train/*.pt and test/test/*.pt) or place it at ./data/nml-2026."
    )
TRAIN = str(DATA / "train" / "train")   # normalization stats (matches how _full models were trained)
TEST = str(DATA / "test" / "test")


def compute_normalization(data_path):
    all_pos = []
    for f in os.listdir(data_path):
        if f.endswith(".pt"):
            seq = torch.load(os.path.join(data_path, f), weights_only=False)
            all_pos.append(seq[:, :, [0, 1]])
    cat = torch.cat(all_pos, dim=0)
    return cat.mean(dim=(0, 1)), cat.std(dim=(0, 1))


def load_model(weights_file):
    """Every member shares the NBAv46Model architecture (demarche/nba_v46.py)."""
    spec = importlib.util.spec_from_file_location("nba_v46", ARCH_DIR / "nba_v46.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.NBAv46Model().to(DEVICE)
    net.load_state_dict(torch.load(WEIGHTS_DIR / weights_file, weights_only=True))
    net.train(False)
    return net


def predict_tta(net, ctx, mu_d, sig_d):
    """ctx: [1, 8, 11, F] normalized. Returns [12, 11, 2] in feet, averaged over 4 court reflections."""
    out = []
    for fx, fy in [(False, False), (True, False), (False, True), (True, True)]:
        x = ctx.clone()
        if fx: x[:, :, :, 0] = -x[:, :, :, 0]
        if fy: x[:, :, :, 1] = -x[:, :, :, 1]
        with torch.no_grad():
            p = net(x) * sig_d + mu_d
        if fx: p = p.clone(); p[:, :, 0] = -p[:, :, 0] + 2 * mu_d[0]
        if fy: p = p.clone(); p[:, :, 1] = -p[:, :, 1] + 2 * mu_d[1]
        out.append(p)
    return torch.stack(out).mean(0).reshape(HORIZON, NE, 2)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "v46"
    weights_file = f"model_{label}_full.pth"
    if not (WEIGHTS_DIR / weights_file).exists():
        raise SystemExit(
            f"Missing checkpoint {WEIGHTS_DIR / weights_file}. Train it first:\n"
            f"  python demarche/ensemble/train_member.py {label} 300 --full"
        )
    print(f"device={DEVICE}  model={label}  ({weights_file})")

    print("computing normalization from train ...")
    mu, sig = compute_normalization(TRAIN)
    mu_d, sig_d = mu.to(DEVICE), sig.to(DEVICE)

    test_files = sorted([f for f in os.listdir(TEST) if f.endswith(".pt")],
                        key=lambda f: int(f[:-3]))
    ids = [int(f[:-3]) for f in test_files]
    ctxs = []
    for f in test_files:
        s = torch.load(os.path.join(TEST, f), weights_only=False).clone()
        s[:, :, [0, 1]] = (s[:, :, [0, 1]] - mu) / sig
        ctxs.append(s[-CONTEXT:])
    print(f"test sequences: {len(ctxs)}")

    net = load_model(weights_file)
    cols = [f"entity_{i}_time_{t}_{a}" for t in range(HORIZON) for i in range(NE) for a in ["x", "y"]]
    rows = []
    for _id, ctx in zip(ids, ctxs):
        pred = predict_tta(net, ctx.unsqueeze(0).to(DEVICE), mu_d, sig_d)  # [12, 11, 2]
        rows.append([_id] + pred.reshape(-1).cpu().tolist())

    csv = SOL_DIR / f"solution_{label}_full.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id"] + cols).set_index("id").sort_index().to_csv(csv)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
