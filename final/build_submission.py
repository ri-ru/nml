"""Build the FINAL 6-model Kaggle submission (Kaggle ADE 2.954).

Single architecture: all 6 members are NBAv46Model (3 ball specialists trained with a
ball-weighted loss + 3 player generalists at different seeds). The architecture is loaded
from the one canonical file, ../demarche/nba_v46.py (no copy in this folder).

Pipeline (no training):
  1. load the 6 100%-data (_full) checkpoints from ./weights;
  2. run 4-flip TTA inference on every test sequence (last 8 frames -> next 12 positions);
  3. apply the 2-group ensemble weights in ./ensemble_weights.json
        - the BALL (entity 0) uses `ball_weights`
        - every PLAYER (entities 1..10) uses `player_weights`
  4. write ./solution.csv (Kaggle format, ids sorted).

The committed solution.csv is the submitted artifact; this script reproduces it up to the
precision of the published weights in ensemble_weights.json.

Run:
    export NML_DATA=/path/to/nml-2026          # folder with train/train/*.pt and test/test/*.pt
    python final/build_submission.py
"""
import os
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
import pandas as pd

HERE = Path(__file__).resolve().parent
ARCH_DIR = HERE.parent / "demarche"   # single architecture source: demarche/nba_v46.py (no copy)
WEIGHTS_DIR = HERE / "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT, HORIZON, NE = 8, 12, 11

if "NML_DATA" not in os.environ:
    raise SystemExit(
        "Set NML_DATA to the dataset root (the folder containing train/train/*.pt and "
        "test/test/*.pt), e.g.  export NML_DATA=/path/to/nml-2026"
    )
DATA = Path(os.environ["NML_DATA"])
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
    """All 6 members share the NBAv46Model architecture (demarche/nba_v46.py)."""
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
    cfg = json.load(open(HERE / "ensemble_weights.json"))
    pool = cfg["pool"]
    wfile = cfg["weights_file"]
    wb = np.array(cfg["ball_weights"], dtype=np.float64)
    wp = np.array(cfg["player_weights"], dtype=np.float64)
    print(f"device={DEVICE}  pool={pool}")

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

    # Pt[m] = [N_test, 12, 11, 2] for model m, in pool order.
    Pt = np.zeros((len(pool), len(ctxs), HORIZON, NE, 2), dtype=np.float32)
    for m, label in enumerate(pool):
        net = load_model(wfile[label])
        for i, ctx in enumerate(ctxs):
            Pt[m, i] = predict_tta(net, ctx.unsqueeze(0).to(DEVICE), mu_d, sig_d).cpu().numpy()
        del net
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"  predicted {label}")

    M, Ntest = len(pool), len(ctxs)
    out = np.zeros((Ntest, HORIZON, NE, 2), dtype=np.float32)
    out[:, :, 0, :] = (wb @ Pt[:, :, :, 0, :].reshape(M, -1)).reshape(Ntest, HORIZON, 2)
    for e in range(1, NE):
        out[:, :, e, :] = (wp @ Pt[:, :, :, e, :].reshape(M, -1)).reshape(Ntest, HORIZON, 2)

    cols = [f"entity_{i}_time_{t}_{a}" for t in range(HORIZON) for i in range(NE) for a in ["x", "y"]]
    rows = [[ids[i]] + out[i].reshape(-1).tolist() for i in range(Ntest)]
    csv = HERE / "solution.csv"
    pd.DataFrame(rows, columns=["id"] + cols).set_index("id").sort_index().to_csv(csv)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
