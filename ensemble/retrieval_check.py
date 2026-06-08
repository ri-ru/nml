"""Data-structure probe: do val 8-frame contexts MATCH training-game windows? If yes, the test set
is likely cut from the same games and we can RETRIEVE continuations (potentially a huge exploit).

For ~400 val contexts (last-8-frame window of each, matching how we predict), find the nearest
8-frame context window in the TRAIN set (raw court coords, feet). Report:
  - min context RMSE (feet/coord): ~0 => exact same-game overlap.
  - retrieval ADE: the matched train window's actual 12-step future vs the val context's true future,
    compared to our model (~2.78 local). Low => retrieval predicts well.
Train windows are sub-sampled (every 3rd start) to keep brute-force NN fast.
"""
from pathlib import Path
import numpy as np, torch

BASE = Path(__file__).parent.parent
CTX, HOR, WIN = 8, 12, 20
train_files = (BASE/'train_files.txt').read_text().splitlines()
val_files = (BASE/'val_files.txt').read_text().splitlines()

# build train DB (context flattened + future), raw coords
ctxs, futs = [], []
for f in train_files:
    s = torch.load(f, weights_only=False).numpy()[:, :, :2]   # [T,11,2] feet
    ms = len(s) - WIN
    for st in range(0, ms + 1, 3):
        ctxs.append(s[st:st+CTX].reshape(-1))
        futs.append(s[st+CTX:st+WIN].reshape(-1))
DB = np.stack(ctxs).astype(np.float32)          # [W,176]
FUT = np.stack(futs).astype(np.float32)         # [W,264]
print(f'train windows: {DB.shape[0]}')

# sample val contexts: last window of each val seq
vc, vf = [], []
for f in val_files:
    s = torch.load(f, weights_only=False).numpy()[:, :, :2]
    if len(s) < WIN: continue
    st = len(s) - WIN
    vc.append(s[st:st+CTX].reshape(-1)); vf.append(s[st+CTX:st+WIN].reshape(-1))
VC = np.stack(vc).astype(np.float32); VF = np.stack(vf).astype(np.float32)
VC, VF = VC[:400], VF[:400]
print(f'val contexts probed: {VC.shape[0]}')

db_sq = (DB**2).sum(1)
min_rmse, retr_ade = [], []
for i in range(VC.shape[0]):
    d = db_sq + (VC[i]**2).sum() - 2*(DB @ VC[i])   # squared L2 to all train windows
    j = int(np.argmin(d))
    min_rmse.append(np.sqrt(max(d[j],0)/VC.shape[1]))                 # ctx RMSE feet/coord
    retr_ade.append(np.sqrt(((FUT[j]-VF[i])**2).reshape(-1,2).sum(1)).mean())  # mean displacement err
min_rmse = np.array(min_rmse); retr_ade = np.array(retr_ade)

print(f'\n=== CONTEXT MATCH (RMSE feet/coord to nearest train window) ===')
for p in [1,5,25,50]:
    print(f'  {p:>2}th pct: {np.percentile(min_rmse,p):.3f}')
print(f'  frac < 0.1 ft (near-exact): {(min_rmse<0.1).mean()*100:.1f}%')
print(f'  frac < 1.0 ft: {(min_rmse<1.0).mean()*100:.1f}%')
print(f'\n=== RETRIEVAL prediction quality (mean displacement err, feet) ===')
print(f'  median retrieval ADE: {np.median(retr_ade):.3f}   (our model ~2.78 MSE; ADE not directly comparable)')
print(f'  retrieval ADE on near-exact matches (<0.1ft ctx): {retr_ade[min_rmse<0.1].mean() if (min_rmse<0.1).any() else float("nan"):.3f}')
print(f'\nINTERPRETATION: near-zero ctx RMSE on many => test likely overlaps train games => retrieval exploit. Large => no overlap, pivot to pseudo-labeling.')
