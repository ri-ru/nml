"""2-group ensemble: fit ONE constrained weight vector for the BALL (entity 0) and ONE shared
vector for ALL players (entities 1-10). This is a regularized middle ground between global-cons
(1 vector, underfits the ball) and full per-entity (11 vectors, overfits players) -- it keeps the
ball-specialist routing that won us the leaderboard while cutting params 11->2 to shrink the
local->Kaggle overfit gap.

Pool = the 6 final members: 3 ball specialists (v60) + 3 player generalists (v46), single
architecture. Writes solution_ball_player_2grp.csv + reports.
"""
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from _config import VAL_DIR as VAL, TEST_DIR as TEST, SOL_DIR as SOL
HOR, NE = 12, 11
POOL = ['v60_w8', 'v60_w5_s1', 'v60_w8_s2',   # ball specialists
        'v46_s3', 'v46', 'v46_s7']            # player generalists

Y = torch.load(VAL/'Y_real.pt').numpy()
labs = [l for l in POOL if (VAL/f'{l}.pt').exists() and (TEST/f'{l}.pt').exists()]
Pv = np.stack([torch.load(VAL/f'{l}.pt').numpy() for l in labs])   # [M,N,12,11,2]
Pt = np.stack([torch.load(TEST/f'{l}.pt').numpy() for l in labs])
M = len(labs)
print(f'{M} models')

def fit(Pe, Ye):
    res = minimize(lambda w: float(np.mean((w@Pe - Ye)**2)), np.ones(M)/M, method='SLSQP',
                   bounds=[(0,1)]*M, constraints=({'type':'eq','fun':lambda w: w.sum()-1},),
                   options={'maxiter':500,'ftol':1e-10})
    return res.x

# ball group = entity 0 ; player group = entities 1..10 (pooled)
wb = fit(Pv[:,:,:,0,:].reshape(M,-1), Y[:,:,0,:].reshape(-1))
wp = fit(Pv[:,:,:,1:,:].reshape(M,-1), Y[:,:,1:,:].reshape(-1))

def apply(P):
    out = np.zeros((P.shape[1],HOR,NE,2),dtype=np.float32)
    out[:,:,0,:] = (wb@P[:,:,:,0,:].reshape(M,-1)).reshape(P.shape[1],HOR,2)
    for e in range(1,NE):
        out[:,:,e,:] = (wp@P[:,:,:,e,:].reshape(M,-1)).reshape(P.shape[1],HOR,2)
    return out

vp = apply(Pv)
overall = float(np.mean((vp-Y)**2)); ball = float(np.mean((vp[:,:,0,:]-Y[:,:,0,:])**2))
print(f'2-group: overall={overall:.4f}  ball={ball:.3f}  -> proj Kaggle ~{overall*1.079:.3f}-{overall*1.083:.3f}')
tp = apply(Pt)
cols=[f'entity_{i}_time_{t}_{a}' for t in range(HOR) for i in range(NE) for a in ['x','y']]
rows=[[i]+tp[i].reshape(-1).tolist() for i in range(tp.shape[0])]
out=SOL/'solution_ball_player_2grp.csv'
pd.DataFrame(rows,columns=['id']+cols).set_index('id').sort_index().to_csv(out)
print(f'wrote {out.name}')
print(f'top ball weights: ' + ', '.join(f'{labs[i]}={wb[i]:.2f}' for i in np.argsort(-wb)[:4]))
