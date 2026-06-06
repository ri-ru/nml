"""Compute 4-flip TTA TEST predictions for the 100%-data (_full) models. Saves test_preds/<label>_full.pt.
Normalization from the full labeled train dir (matches how _full models were normalized)."""
import os, importlib.util
from pathlib import Path
import torch

DEVICE = 'cuda'
TRAIN = r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\train\train'
TEST  = r'C:\Users\jan\.cache\kagglehub\competitions\nml-2026\test\test'
BASE = Path(__file__).parent.parent
WEIGHTS = BASE / 'models' / 'weights'
OUT = BASE / 'models' / 'test_preds'
CONTEXT, HORIZON = 8, 12

# (label, module, class) — weights are model_<label>_full.pth
MODELS = [
    ('v46','nba_v46','NBAv46Model'), ('v46_s3','nba_v46','NBAv46Model'),
    ('v46_s5','nba_v46','NBAv46Model'), ('v46_s7','nba_v46','NBAv46Model'),
    ('v58','nba_v58','NBAv58Model'), ('v58_s1','nba_v58','NBAv58Model'),
    ('v58_s2','nba_v58','NBAv58Model'), ('v58_s3','nba_v58','NBAv58Model'),
    ('v60_w5','nba_v46','NBAv46Model'), ('v60_w8','nba_v46','NBAv46Model'), ('v60_w3','nba_v46','NBAv46Model'),
    ('v60_w8_s1','nba_v46','NBAv46Model'), ('v60_w8_s2','nba_v46','NBAv46Model'), ('v60_w5_s1','nba_v46','NBAv46Model'),
]


def norm():
    a = []
    for f in os.listdir(TRAIN):
        if f.endswith('.pt'):
            a.append(torch.load(os.path.join(TRAIN, f), weights_only=False)[:, :, [0, 1]])
    c = torch.cat(a, 0); return c.mean((0, 1)), c.std((0, 1))


def load(mod, cls, w):
    spec = importlib.util.spec_from_file_location(mod, BASE / 'models' / f'{mod}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    net = getattr(m, cls)().to(DEVICE)
    net.load_state_dict(torch.load(WEIGHTS / w, weights_only=True)); net.train(False); return net


def tta(net, ctx, mu_d, sig_d):
    out = []
    for fx, fy in [(False,False),(True,False),(False,True),(True,True)]:
        x = ctx.clone()
        if fx: x[:,:,:,0] = -x[:,:,:,0]
        if fy: x[:,:,:,1] = -x[:,:,:,1]
        with torch.no_grad(): p = net(x) * sig_d + mu_d
        if fx: p = p.clone(); p[:,:,0] = -p[:,:,0] + 2*mu_d[0]
        if fy: p = p.clone(); p[:,:,1] = -p[:,:,1] + 2*mu_d[1]
        out.append(p)
    return torch.stack(out).mean(0)


def main():
    mu, sig = norm(); mu_d, sig_d = mu.to(DEVICE), sig.to(DEVICE)
    files = sorted([f for f in os.listdir(TEST) if f.endswith('.pt')], key=lambda f: int(f[:-3]))
    ctxs = []
    for f in files:
        s = torch.load(os.path.join(TEST, f), weights_only=False).clone()
        s[:, :, [0,1]] = (s[:, :, [0,1]] - mu) / sig
        ctxs.append(s[-CONTEXT:])
    for label, mod, cls in MODELS:
        w = WEIGHTS / f'model_{label}_full.pth'
        outp = OUT / f'{label}_full.pt'
        if outp.exists(): print(f'skip {label}_full'); continue
        if not w.exists(): print(f'MISSING {label}_full weights'); continue
        net = load(mod, cls, w.name)
        preds = []
        for ctx in ctxs:
            preds.append(tta(net, ctx.unsqueeze(0).to(DEVICE), mu_d, sig_d).reshape(HORIZON,11,2).cpu())
        torch.save(torch.stack(preds), outp)
        del net; torch.cuda.empty_cache()
        print(f'saved {label}_full ({len(preds)})')


if __name__ == '__main__':
    main()
