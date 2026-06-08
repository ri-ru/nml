"""One-off: re-fit the 2-group (ball|players) weights on the honest 80% val preds and dump them
full-precision to ensemble_weights.json. The values are identical to build_trimmed_candidate.py."""
import json, numpy as np, torch
from pathlib import Path
from scipy.optimize import minimize

VAL = Path(__file__).resolve().parent.parent / "ensemble" / "val_preds"
POOL = ['v60_w8', 'v60_w5_s1', 'v60_w8_s2', 'v46', 'v58', 'v46_s3', 'v46_s7']
M = len(POOL)
Y = torch.load(VAL / 'Y_real.pt', weights_only=False).numpy()
Pv = np.stack([torch.load(VAL / f'{l}.pt', weights_only=False).numpy() for l in POOL])

def fit(Pe, Ye):
    r = minimize(lambda w: float(np.mean((w @ Pe - Ye) ** 2)), np.ones(M) / M, method='SLSQP',
                 bounds=[(0, 1)] * M, constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1},),
                 options={'maxiter': 500, 'ftol': 1e-10})
    return r.x

wb = fit(Pv[:, :, :, 0, :].reshape(M, -1), Y[:, :, 0, :].reshape(-1))
wp = fit(Pv[:, :, :, 1:, :].reshape(M, -1), Y[:, :, 1:, :].reshape(-1))

# reference local 2-group score (80% preds) for documentation
vp = np.zeros_like(Y)
vp[:, :, 0, :] = (wb @ Pv[:, :, :, 0, :].reshape(M, -1)).reshape(Y.shape[0], 12, 2)
for e in range(1, 11):
    vp[:, :, e, :] = (wp @ Pv[:, :, :, e, :].reshape(M, -1)).reshape(Y.shape[0], 12, 2)
local = float(np.mean((vp - Y) ** 2))

doc = {
    "_comment": "2-group ensemble weights for the final 7-model submission. wb routes the BALL "
                "(entity 0); wp routes ALL players (entities 1-10, pooled). Fit by constrained "
                "least-squares (w>=0, sum=1) on the honest 80% validation predictions; applied to "
                "the 100%-data (_full) test predictions. Order matches `pool`.",
    "pool": POOL,
    "architecture": {l: ("nba_v58/NBAv58Model" if l == "v58" else "nba_v46/NBAv46Model") for l in POOL},
    "weights_file": {l: f"model_{l}_full.pth" for l in POOL},
    "ball_weights": [float(x) for x in wb],
    "player_weights": [float(x) for x in wp],
    "reference_local_val_mse_80pct": round(local, 4),
    "kaggle_mse": 2.95815,
    "kaggle_ref_27model": 2.95986,
}
out = Path(__file__).resolve().parent / "ensemble_weights.json"
json.dump(doc, open(out, "w"), indent=2)
print("wrote", out)
print("ball  :", {POOL[i]: round(float(wb[i]), 4) for i in range(M)})
print("player:", {POOL[i]: round(float(wp[i]), 4) for i in range(M)})
print("local 80% 2-group MSE =", round(local, 4))
