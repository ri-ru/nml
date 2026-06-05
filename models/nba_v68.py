"""v68 — pseudo-label round 2. v46 architecture, trained on real + round-2 pseudo labels generated
from the BEST current ensemble (not the old 3.00676 one). The orchestrator writes
train_files_pseudo_round2.txt (real train + fresh pseudo) before this launches.

Normalization is computed from REAL train only (no leakage from pseudo). Falls back to the existing
train_files_pseudo_cons.txt, then plain train_files.txt, if the round-2 file is absent.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import nba_v46
from nba_v46 import NBADataset, NBAv46Model, MoGNLLLoss, MultiStepMSE, WEIGHTS_DIR, NBASampler

_DIR = Path(__file__).parent
_PROJ = _DIR.parent
REAL_TRAIN = _PROJ / 'train_files.txt'
VAL_TXT = _PROJ / 'val_files.txt'
_CANDIDATES = [
    _PROJ / 'train_files_pseudo_round2.txt',
    _PROJ / 'train_files_pseudo_cons.txt',
    REAL_TRAIN,
]


class Trainer(nba_v46.Trainer):
    def __init__(self, train_path, val_path, device='cuda', epochs=35, batch_size=32, **kwargs):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

        pseudo_txt = next(p for p in _CANDIDATES if p.exists())
        print(f'v68 training file list: {pseudo_txt.name}')
        train_files = pseudo_txt.read_text().splitlines()
        val_files = VAL_TXT.read_text().splitlines()

        # normalization from REAL train only (avoid pseudo leakage)
        real_files = REAL_TRAIN.read_text().splitlines()
        mu, sigma = self._compute_normalization(real_files)
        self.mu, self.sigma = mu, sigma
        self.train_dataset = NBADataset(train_files, 8, 12, mu, sigma, augment=True)
        self.val_dataset = NBADataset(val_files, 8, 12, mu, sigma, augment=False)

    def train(self):
        import random
        torch.manual_seed(0); random.seed(0)
        generator = torch.Generator('cpu').manual_seed(0)
        train_sampler = NBASampler(self.batch_size, self.train_dataset.max_start, shuffle=True)
        val_sampler = NBASampler(self.batch_size, self.val_dataset.max_start, shuffle=False)
        from torch.utils.data import DataLoader
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, generator=generator)
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=val_sampler, generator=generator)

        model = NBAv46Model().to(self.device)
        train_loss_fn = MoGNLLLoss(); val_loss_fn = MultiStepMSE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            model.train(); self._run_epoch(train_dl, train_loss_fn, model, optimizer, 'train')
            model.train(False)
            with torch.no_grad():
                self._run_epoch(val_dl, val_loss_fn, model, None, 'val')
            train_sampler.set_epoch(epoch); val_sampler.set_epoch(epoch); scheduler.step()
        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v68.pth')
        self.model = model
