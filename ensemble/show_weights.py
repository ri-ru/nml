"""Print the full fitted 2-group ensemble weight vectors (ball | players) for all pool models."""
import numpy as np, torch
from scipy.optimize import minimize

from _config import VAL_DIR as VAL
POOL = ['v46','v58','v46_s1','v46_s2','v46_s3','v46_s4','v46_s5','v46_s6','v46_s7','v46_s8',
        'v46_s1_swa','v46_s2_swa','v46_s3_swa','v46_s4_swa','v46_s5_swa','v46_s6_swa','v46_s7_swa','v46_s8_swa',
        'v58_s1','v58_s2','v58_s3','v60_w5','v60_w8','v60_w3','v60_w8_s1','v60_w8_s2','v60_w5_s1']
Y = torch.load(VAL/'Y_real.pt').numpy()
labs=[l for l in POOL if (VAL/f'{l}.pt').exists()]
P=np.stack([torch.load(VAL/f'{l}.pt').numpy() for l in labs]); M=len(labs)

def fit(Pe,Ye):
    r=minimize(lambda w: float(np.mean((w@Pe-Ye)**2)), np.ones(M)/M, method='SLSQP',
               bounds=[(0,1)]*M, constraints=({'type':'eq','fun':lambda w:w.sum()-1},),
               options={'maxiter':500,'ftol':1e-10}); return r.x
wb=fit(P[:,:,:,0,:].reshape(M,-1), Y[:,:,0,:].reshape(-1))
wp=fit(P[:,:,:,1:,:].reshape(M,-1), Y[:,:,1:,:].reshape(-1))

print(f"{'model':<14}{'BALL w':>9}{'PLAYER w':>10}")
print('-'*33)
for i in np.argsort(-(wb+wp)):
    print(f"{labs[i]:<14}{wb[i]:>9.3f}{wp[i]:>10.3f}")
print('-'*33)
print(f"{'(sum)':<14}{wb.sum():>9.3f}{wp.sum():>10.3f}")
print(f"\nball:    nonzero models = {(wb>0.005).sum()}")
print(f"players: nonzero models = {(wp>0.005).sum()}")
