"""Build the 100%-data submission candidate: fit 2-group (ball|players) weights on the honest 80%
val preds, then APPLY them to the _full TEST preds (100%-data members) where available, falling back
to 80% test preds for non-retrained members (e.g. SWA). Writes solution_full_2grp.csv + a report."""
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize

BASE = Path(__file__).parent.parent
VAL, TEST, SOL = BASE/'models'/'val_preds', BASE/'models'/'test_preds', BASE/'models'/'solutions'
HOR, NE = 12, 11
POOL = ['v46','v58','v46_s1','v46_s2','v46_s3','v46_s4','v46_s5','v46_s6','v46_s7','v46_s8',
        'v46_s1_swa','v46_s2_swa','v46_s3_swa','v46_s4_swa','v46_s5_swa','v46_s6_swa','v46_s7_swa','v46_s8_swa',
        'v58_s1','v58_s2','v58_s3','v60_w5','v60_w8','v60_w3','v60_w8_s1','v60_w8_s2','v60_w5_s1']

Y = torch.load(VAL/'Y_real.pt').numpy()
labs = [l for l in POOL if (VAL/f'{l}.pt').exists() and (TEST/f'{l}.pt').exists()]
Pv = np.stack([torch.load(VAL/f'{l}.pt').numpy() for l in labs]); M = len(labs)

def fit(Pe, Ye):
    r = minimize(lambda w: float(np.mean((w@Pe-Ye)**2)), np.ones(M)/M, method='SLSQP',
                 bounds=[(0,1)]*M, constraints=({'type':'eq','fun':lambda w:w.sum()-1},),
                 options={'maxiter':500,'ftol':1e-10}); return r.x
wb = fit(Pv[:,:,:,0,:].reshape(M,-1), Y[:,:,0,:].reshape(-1))
wp = fit(Pv[:,:,:,1:,:].reshape(M,-1), Y[:,:,1:,:].reshape(-1))

# test preds: prefer _full, else 80%
nfull = 0
Pt = []
for l in labs:
    fp = TEST/f'{l}_full.pt'
    if fp.exists(): Pt.append(torch.load(fp).numpy()); nfull += 1
    else: Pt.append(torch.load(TEST/f'{l}.pt').numpy())
Pt = np.stack(Pt)
Ntest = Pt.shape[1]
out = np.zeros((Ntest,HOR,NE,2),dtype=np.float32)
out[:,:,0,:] = (wb@Pt[:,:,:,0,:].reshape(M,-1)).reshape(Ntest,HOR,2)
for e in range(1,NE):
    out[:,:,e,:] = (wp@Pt[:,:,:,e,:].reshape(M,-1)).reshape(Ntest,HOR,2)

cols=[f'entity_{i}_time_{t}_{a}' for t in range(HOR) for i in range(NE) for a in ['x','y']]
rows=[[i]+out[i].reshape(-1).tolist() for i in range(Ntest)]
csv = SOL/'solution_full_2grp.csv'
pd.DataFrame(rows,columns=['id']+cols).set_index('id').sort_index().to_csv(csv)

# local val score of the (80%-pred) 2-group for reference
vp = np.zeros_like(Y)
vp[:,:,0,:] = (wb@Pv[:,:,:,0,:].reshape(M,-1)).reshape(Y.shape[0],HOR,2)
for e in range(1,NE):
    vp[:,:,e,:] = (wp@Pv[:,:,:,e,:].reshape(M,-1)).reshape(Y.shape[0],HOR,2)
local = float(np.mean((vp-Y)**2))
msg = (f"100%-DATA CANDIDATE: solution_full_2grp.csv\n"
       f"  {nfull}/{M} members use _full (100%-data) test preds; rest 80%.\n"
       f"  reference local 2-group (80% preds) = {local:.4f}; submitted 2-group was 2.7563 -> Kaggle 2.98086.\n"
       f"  NOTE: local can't measure the _full gain (val is in their training set). Submit and compare on Kaggle.")
print(msg)
(BASE/'MORNING_SUMMARY.md').open('a', encoding='utf-8').write('\n## 100% candidate\n'+msg+'\n')
