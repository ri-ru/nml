"""Generic ball decomposition: load ANY model class + weights, report overall/ball/player MSE
on the val set (evaluate.make_val_dl protocol). Works for custom-arch models (v70 etc.).

  python misc/eval_ball_generic.py <model_module> <ClassName> <weights_filename>
e.g. python misc/eval_ball_generic.py nba_v70 NBAv70Model model_v70_50ep.pth
"""
import sys
import importlib.util
from pathlib import Path
import torch

BASE = Path(__file__).parent.parent
WEIGHTS = BASE / 'models' / 'weights'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

mod_name, cls_name, wname = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location(mod_name, BASE / 'models' / f'{mod_name}.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = getattr(mod, cls_name)().to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS / wname, weights_only=True))
model.train(False)

trainer = mod.Trainer(None, None, device=DEVICE)
mu, sigma = trainer.mu.to(DEVICE), trainer.sigma.to(DEVICE)
sys.path.insert(0, str(BASE))
import evaluate
val_dl = evaluate.make_val_dl(trainer, batch_size=64)

se = torch.zeros(12, 11, 2, device=DEVICE); cnt = 0
with torch.no_grad():
    for X, y in val_dl:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        B, T, N, _ = y.shape
        y_n = y[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        se += ((pred * sigma + mu - (y_n * sigma + mu)) ** 2).reshape(T, B, N, 2).sum(1)
        cnt += B
overall = se.sum().item() / (cnt * 11 * 12 * 2)
ball = se[:, 0, :].sum().item() / (cnt * 12 * 2)
players = se[:, 1:, :].sum().item() / (cnt * 10 * 12 * 2)
print(f'{wname}: overall={overall:.4f} ball={ball:.3f} player={players:.4f} ball%={se[:,0,:].sum().item()/se.sum().item()*100:.1f}')
